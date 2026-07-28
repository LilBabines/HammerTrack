"""
track_postprocess.py
====================

Numeric post-processing of the per-individual tracks exported by the GUI.

**Merging is no longer done here.** The GUI owns identity: it groups tracker
fragments into individuals and resolves frame collisions
(``IndividualStore.export_individuals`` / ``merge_detections``), then writes one
JSON per animal in ``<export_dir>/individuals/``. This script only refines
those trajectories numerically, and re-emits them under the *same filename
stem* so identity survives:

    individuals/<name>.json   ->   postp_tracks/<name>.json

Passes, in order (each can be disabled):

  1. **Outlier removal** — drop detections that jump implausibly far from the
     previous one. CMC-aware when ``--cmc`` is given, so camera motion is not
     mistaken for animal motion.
  2. **Interpolation** — fill every missing frame between the first and last
     detection. Done in the CMC reference frame when available, so interpolated
     centroids follow the animal rather than the camera.
  3. **Smoothing** — Savitzky-Golay (or moving average) on the centroids.

Typical invocation:

    python scripts/track_postprocess.py \\
        --tracks     projects/<project>/export/<clip_id>/individuals/ \\
        --output-dir projects/<project>/export/<clip_id>/postp_tracks/ \\
        --cmc        projects/<project>/export/<clip_id>/cmc_transforms.json

Add ``--video`` and ``--render-video`` to also render an overlay for QC.
Run ``python scripts/track_postprocess.py --help`` for all options.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

import track_io


# Per-detection keys dropped on export. ``cmc_affine`` duplicates
# cmc_transforms.json on every single detection and bloats the files for no
# gain; pass --keep-cmc-affine to preserve it.
DEFAULT_DROPPED_KEYS = ("cmc_affine",)

# Keys mapped onto explicit Detection fields; everything else is passed through
# untouched via Detection.extra.
KNOWN_DETECTION_KEYS = (
    "frame", "centroid", "bbox", "confidence", "class_id",
    "obb", "source_track_id", "interpolated",
)


# =============================================================================
# Data model
# =============================================================================

@dataclass
class Detection:
    frame: int
    centroid: List[float]
    bbox: List[float]
    confidence: float
    class_id: int
    obb: Optional[list] = None
    source_track_id: Optional[int] = None
    interpolated: bool = False
    # Any extra keys carried by the input (e.g. obb_xywhr), preserved verbatim.
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Track:
    """One individual. ``id`` is the filename stem — see track_io.track_id."""
    id: str
    detections: Dict[int, Detection] = field(default_factory=dict)  # frame -> Detection
    meta: Dict[str, Any] = field(default_factory=dict)              # uid/name/notes/...
    history: List[str] = field(default_factory=list)                # applied passes

    @property
    def frames(self) -> List[int]:
        return sorted(self.detections.keys())


# =============================================================================
# 1. Load
# =============================================================================

def track_from_json(data: dict) -> Track:
    """Build a :class:`Track` from a loaded track JSON (any of the schemas)."""
    track = Track(id=data["id"], meta=track_io.identity_payload(data))

    for det in data["detections"]:
        frame = int(det["frame"])
        extra = {k: v for k, v in det.items() if k not in KNOWN_DETECTION_KEYS}
        confidence = float(det.get("confidence", 0.0))
        track.detections[frame] = Detection(
            frame=frame,
            centroid=list(det["centroid"]),
            bbox=list(det["bbox"]),
            confidence=confidence,
            class_id=int(det.get("class_id", 0)),
            obb=det.get("obb"),
            source_track_id=det.get("source_track_id"),
            # The GUI writes the flag explicitly; older files only encoded it
            # as confidence == 0.0.
            interpolated=bool(det.get("interpolated", confidence == 0.0)),
            extra=extra,
        )
    return track


def load_individual_tracks(tracks_arg: str, pattern: str = "*.json") -> List[Track]:
    """Load every individual JSON as a :class:`Track`, identity from the stem."""
    return [track_from_json(d) for d in track_io.load_tracks(tracks_arg, pattern)]


# =============================================================================
# 2. Outlier removal
# =============================================================================

def remove_outliers(
    track: Track,
    max_jump_px: float = 150.0,
    z_threshold: float = 3.0,
    method: str = "jump",
    cum_affines: Optional[Dict[int, np.ndarray]] = None,
) -> Track:
    """Drop detections whose displacement from the previous one is implausible.

    With ``cum_affines``, displacements are measured in the CMC reference frame,
    so a fast camera pan no longer looks like a teleporting shark.
    """
    if method == "none":
        return track

    frames = track.frames
    if len(frames) < 3:
        return track

    use_cmc = cum_affines is not None and all(f in cum_affines for f in frames)
    if use_cmc:
        centroids = np.array([
            warp_point_to_ref(track.detections[f].centroid, cum_affines[f])
            for f in frames
        ])
    else:
        centroids = np.array([track.detections[f].centroid for f in frames])

    if method == "jump":
        keep = {frames[0]}
        for i in range(1, len(frames)):
            if np.linalg.norm(centroids[i] - centroids[i - 1]) <= max_jump_px:
                keep.add(frames[i])

    elif method == "zscore":
        diffs = np.linalg.norm(np.diff(centroids, axis=0), axis=1)
        mu, sigma = diffs.mean(), diffs.std() + 1e-9
        z = (diffs - mu) / sigma
        keep = {frames[0]}
        for i, zi in enumerate(z):
            if abs(zi) <= z_threshold:
                keep.add(frames[i + 1])
    else:
        raise ValueError(f"Unknown outlier method: {method}")

    n_before = len(track.detections)
    track.detections = {f: d for f, d in track.detections.items() if f in keep}
    space = "ref space" if use_cmc else "image space"
    track.history.append(
        f"outliers[{method}, {space}]: -{n_before - len(track.detections)}"
    )
    return track


# =============================================================================
# 3. Interpolation
# =============================================================================

def interpolate_missing(
    track: Track,
    method: str = "linear",
    cum_affines: Optional[Dict[int, np.ndarray]] = None,
) -> Track:
    """Fill every missing frame between the first and last detection.

    With ``cum_affines``, the interpolation happens in the reference frame and
    the result is warped back, so camera motion does not corrupt the filled
    centroids. Filled detections get ``confidence = 0.0``,
    ``interpolated = True`` and no OBB.
    """
    if method == "none":
        return track

    frames = track.frames
    if len(frames) < 2:
        return track

    missing = sorted(set(range(frames[0], frames[-1] + 1)) - set(frames))
    if not missing:
        return track

    use_cmc = cum_affines is not None and all(f in cum_affines for f in frames)
    if use_cmc:
        centroids = np.array([
            warp_point_to_ref(track.detections[f].centroid, cum_affines[f])
            for f in frames
        ])
    else:
        centroids = np.array([track.detections[f].centroid for f in frames])

    bboxes = np.array([track.detections[f].bbox for f in frames])
    frames_arr = np.array(frames, dtype=float)
    missing_arr = np.array(missing, dtype=float)

    if method == "linear":
        interp_cx = np.interp(missing_arr, frames_arr, centroids[:, 0])
        interp_cy = np.interp(missing_arr, frames_arr, centroids[:, 1])
        interp_bbox = np.column_stack([
            np.interp(missing_arr, frames_arr, bboxes[:, i]) for i in range(4)
        ])

    elif method == "cubic":
        from scipy.interpolate import CubicSpline
        interp_cx = CubicSpline(frames_arr, centroids[:, 0])(missing_arr)
        interp_cy = CubicSpline(frames_arr, centroids[:, 1])(missing_arr)
        interp_bbox = np.column_stack([
            CubicSpline(frames_arr, bboxes[:, i])(missing_arr) for i in range(4)
        ])
    else:
        raise ValueError(f"Unknown interpolation method: {method}")

    class_id = track.detections[frames[0]].class_id

    for i, f in enumerate(missing):
        c_ref = np.array([float(interp_cx[i]), float(interp_cy[i])])
        if use_cmc and f in cum_affines:
            c_img = warp_point_from_ref(c_ref, cum_affines[f])
        else:
            c_img = c_ref

        track.detections[f] = Detection(
            frame=f,
            centroid=[float(c_img[0]), float(c_img[1])],
            bbox=[float(v) for v in interp_bbox[i]],
            confidence=0.0,
            class_id=class_id,
            obb=None,
            source_track_id=None,
            interpolated=True,
        )

    space = "ref space" if use_cmc else "image space"
    track.history.append(f"interpolate[{method}, {space}]: +{len(missing)}")
    return track


# =============================================================================
# 4. Smoothing
# =============================================================================

def smooth_centroids(
    track: Track,
    method: str = "savgol",
    window: int = 7,
    polyorder: int = 2,
    smooth_interpolated: bool = True,
) -> Track:
    """Smooth the centroid series.

    Operates on detections in frame order. Run this *after* interpolation:
    on a gappy track the filter treats consecutive samples as adjacent in time,
    which distorts the trajectory across the gaps.
    """
    if method == "none":
        return track

    frames = track.frames
    if len(frames) < max(window, polyorder + 2):
        return track

    centroids = np.array([track.detections[f].centroid for f in frames])

    if method == "savgol":
        from scipy.signal import savgol_filter
        w = min(window, len(centroids))
        if w % 2 == 0:
            w -= 1
        w = max(w, polyorder + 2)
        if w % 2 == 0:
            w += 1
        smoothed = savgol_filter(centroids, window_length=w, polyorder=polyorder, axis=0)

    elif method == "moving_avg":
        kernel = np.ones(window) / window
        smoothed = np.column_stack([
            np.convolve(centroids[:, 0], kernel, mode="same"),
            np.convolve(centroids[:, 1], kernel, mode="same"),
        ])
    else:
        raise ValueError(f"Unknown smoothing method: {method}")

    for i, f in enumerate(frames):
        det = track.detections[f]
        if det.interpolated and not smooth_interpolated:
            continue
        det.centroid = [float(smoothed[i][0]), float(smoothed[i][1])]

    track.history.append(f"smooth[{method}, w={window}, p={polyorder}]")
    return track


# =============================================================================
# 5. Export
# =============================================================================

def export_track(track: Track, output_path: str, dropped_keys=DEFAULT_DROPPED_KEYS):
    """Write one post-processed track JSON, identity metadata preserved.

    The schema is a superset of the GUI's: identity keys pass through untouched,
    so ``cohesion.py`` / ``angle.py`` / ``merge_csv_per_track.py`` read the
    ``individuals/`` and ``postp_tracks/`` files indifferently.
    """
    frames = track.frames
    detections = []
    for f in frames:
        d = track.detections[f]
        det: Dict[str, Any] = {
            "frame": d.frame,
            "centroid": [round(d.centroid[0], 2), round(d.centroid[1], 2)],
            "bbox": [round(v, 2) for v in d.bbox],
            "confidence": d.confidence,
            "class_id": d.class_id,
            "interpolated": d.interpolated,
        }
        if d.obb is not None:
            det["obb"] = d.obb
        if d.source_track_id is not None:
            det["source_track_id"] = d.source_track_id
        for k, v in d.extra.items():
            if k not in dropped_keys:
                det[k] = v
        detections.append(det)

    out = {
        **track.meta,                       # uid, name, notes, color, merged_track_ids
        "id": track.id,                     # = filename stem, the pipeline identity
        "num_detections": len(detections),
        "first_frame": frames[0] if frames else None,
        "last_frame": frames[-1] if frames else None,
        "postprocess": track.history,       # provenance of the numeric passes
        "detections": detections,
    }
    Path(output_path).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return len(detections)


# =============================================================================
# CMC helpers
# =============================================================================

def load_cmc(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def build_cum_affine(cmc: dict, start: int, end: int) -> Dict[int, np.ndarray]:
    """Cumulative affine from frame ``start`` to each frame in [start, end)."""
    cum: Dict[int, np.ndarray] = {}
    M_acc = np.eye(3, dtype=np.float64)
    for f in range(start, end):
        entry = cmc.get(str(f))
        if entry is not None:
            A = np.eye(3, dtype=np.float64)
            A[:2, :] = np.array(entry)
            M_acc = A @ M_acc
        cum[f] = M_acc[:2, :].copy()
    return cum


def warp_point_to_ref(pt, cum_affine_f):
    A = np.eye(3, dtype=np.float64)
    A[:2, :] = cum_affine_f
    p = np.linalg.inv(A) @ np.array([pt[0], pt[1], 1.0])
    return p[:2]


def warp_point_from_ref(pt, cum_affine_f):
    return cum_affine_f @ np.array([pt[0], pt[1], 1.0])


# =============================================================================
# 6. Optional QC rendering
# =============================================================================

TRACK_COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),
    (0, 128, 255), (128, 255, 0), (255, 0, 128), (0, 255, 128),
    (200, 200, 0), (200, 0, 200), (0, 200, 200), (100, 100, 255),
    (255, 100, 100), (100, 255, 100),
]


def render_tracks_on_video(
    video_path: str,
    tracks: List[Track],
    output_path: str,
    trail_length: int = 10,
    draw_bbox: bool = False,
    draw_obb: bool = True,
    draw_centroid: bool = True,
    draw_trail: bool = True,
    draw_label: bool = True,
    codec: str = "mp4v",
):
    """Overlay the post-processed tracks on the clip. Labels = track IDs."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*codec), fps, (w, h))

    frame_index: Dict[int, list] = defaultdict(list)
    for tidx, track in enumerate(tracks):
        for f, det in track.detections.items():
            frame_index[f].append((tidx, det))

    print(f"Rendering {total_frames} frames -> {output_path}")
    trails: Dict[int, list] = defaultdict(list)
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for tidx, det in frame_index.get(frame_num, []):
            color = TRACK_COLORS[tidx % len(TRACK_COLORS)]
            cx, cy = int(det.centroid[0]), int(det.centroid[1])

            trails[tidx].append((cx, cy))
            if len(trails[tidx]) > trail_length:
                trails[tidx] = trails[tidx][-trail_length:]

            if draw_trail and len(trails[tidx]) > 1:
                pts = trails[tidx]
                for i in range(1, len(pts)):
                    thick = max(1, int((i / len(pts)) * 3))
                    cv2.line(frame, pts[i - 1], pts[i], color, thick, cv2.LINE_AA)

            if draw_bbox and det.bbox and len(det.bbox) == 4:
                x1, y1, x2, y2 = map(int, det.bbox)
                if det.interpolated:
                    _draw_dashed_rect(frame, (x1, y1), (x2, y2), color, 1, 8)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            if draw_obb and det.obb is not None and len(det.obb) >= 4:
                pts_obb = np.array(det.obb, dtype=np.float32)
                if pts_obb.ndim == 2 and pts_obb.shape[0] >= 4:
                    cv2.polylines(frame, [pts_obb[:4].astype(np.int32)],
                                  True, color, 2, cv2.LINE_AA)

            if draw_centroid:
                cv2.circle(frame, (cx, cy), 3 if det.interpolated else 5,
                           color, -1, cv2.LINE_AA)

            if draw_label:
                label = tracks[tidx].id
                if det.interpolated:
                    label += " (interp)"
                cv2.putText(frame, label, (cx + 8, cy - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        out.write(frame)
        frame_num += 1
        if frame_num % 500 == 0:
            print(f"  {frame_num}/{total_frames} frames rendered")

    cap.release()
    out.release()
    print(f"Done. Output: {output_path}")


def _draw_dashed_rect(img, pt1, pt2, color, thickness=1, dash_len=8):
    import cv2
    edges = [
        (pt1, (pt2[0], pt1[1])),
        ((pt2[0], pt1[1]), pt2),
        (pt2, (pt1[0], pt2[1])),
        ((pt1[0], pt2[1]), pt1),
    ]
    for (x1, y1), (x2, y2) in edges:
        dist = int(np.hypot(x2 - x1, y2 - y1))
        if dist == 0:
            continue
        dx, dy = (x2 - x1) / dist, (y2 - y1) / dist
        for i in range(0, dist, dash_len * 2):
            s, e = i, min(i + dash_len, dist)
            cv2.line(img,
                     (int(x1 + dx * s), int(y1 + dy * s)),
                     (int(x1 + dx * e), int(y1 + dy * e)),
                     color, thickness, cv2.LINE_AA)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Numeric post-processing of the per-individual tracks exported by "
            "the GUI (individuals/). Merging is done by the GUI; this script "
            "removes outliers, interpolates gaps and smooths centroids, then "
            "writes one JSON per individual under the same filename stem."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- I/O ---
    p.add_argument("--tracks", required=True,
                   help="Directory OR glob of per-individual track JSONs "
                        "(typically <export_dir>/individuals/).")
    p.add_argument("--output-dir", required=True,
                   help="Output directory for the post-processed JSONs "
                        "(typically <export_dir>/postp_tracks/).")
    p.add_argument("--pattern", default="*.json",
                   help="Glob pattern used when --tracks is a directory.")
    p.add_argument("--cmc", default=None,
                   help="Camera-motion-compensation JSON (<export_dir>/"
                        "cmc_transforms.json). Enables CMC-aware outlier "
                        "detection and interpolation. Strongly recommended.")
    p.add_argument("--cmc-end-frame", type=int, default=None,
                   help="Last frame for the cumulative affine table. "
                        "Default: max frame seen in the tracks, +1.")
    p.add_argument("--keep-cmc-affine", action="store_true",
                   help="Keep the per-detection 'cmc_affine' key on export "
                        "(redundant with --cmc and much larger files).")

    # --- Pass 1: outliers ---
    p.add_argument("--outlier-method", choices=["jump", "zscore", "none"],
                   default="jump",
                   help="Outlier rule. 'none' disables the pass.")
    p.add_argument("--max-jump-px", type=float, default=150.0,
                   help="Max centroid displacement between consecutive "
                        "detections, for --outlier-method jump.")
    p.add_argument("--z-threshold", type=float, default=3.0,
                   help="Z-score cutoff, for --outlier-method zscore.")

    # --- Pass 2: interpolation ---
    p.add_argument("--interp-method", choices=["linear", "cubic", "none"],
                   default="linear",
                   help="Gap-filling rule. 'none' disables the pass.")

    # --- Pass 3: smoothing ---
    p.add_argument("--smooth-method", choices=["savgol", "moving_avg", "none"],
                   default="savgol",
                   help="Centroid smoothing rule. 'none' disables the pass.")
    p.add_argument("--smooth-window", type=int, default=7,
                   help="Smoothing window, in frames.")
    p.add_argument("--smooth-poly", type=int, default=2,
                   help="Savitzky-Golay polynomial order.")
    p.add_argument("--no-smooth-interpolated", action="store_true",
                   help="Leave interpolated centroids untouched by the "
                        "smoothing pass.")

    # --- Optional QC render ---
    p.add_argument("--render-video", default=None,
                   help="Also render an overlay video to this path. "
                        "Requires --video.")
    p.add_argument("--video", default=None,
                   help="Source clip, for --render-video.")
    p.add_argument("--trail-length", type=int, default=10,
                   help="Centroid trail length in the rendered video.")
    p.add_argument("--draw-bbox", action="store_true",
                   help="Draw axis-aligned bboxes in the rendered video.")
    p.add_argument("--codec", default="mp4v",
                   help="FourCC codec for the rendered video.")
    return p.parse_args()


def main():
    args = parse_args()

    if args.render_video and not args.video:
        sys.exit("--render-video requires --video")

    tracks = load_individual_tracks(args.tracks, args.pattern)
    print(f"Loaded {len(tracks)} individuals: {[t.id for t in tracks]}\n")

    # --- Cumulative affines, shared by every track ---
    cum_affines = None
    if args.cmc:
        cmc = load_cmc(args.cmc)
        max_frame = max(
            (f for t in tracks for f in t.detections), default=0
        )
        end = args.cmc_end_frame if args.cmc_end_frame is not None else max_frame + 1
        cum_affines = build_cum_affine(cmc, start=0, end=end)
        print(f"CMC: {len(cmc)} matrices, cumulative table over frames 0..{end - 1}\n")
    else:
        print("No --cmc given: outlier detection and interpolation run in "
              "image space, so camera motion will bias both.\n")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dropped = () if args.keep_cmc_affine else DEFAULT_DROPPED_KEYS

    total = 0
    for track in tracks:
        n_in = len(track.detections)

        remove_outliers(
            track,
            max_jump_px=args.max_jump_px,
            z_threshold=args.z_threshold,
            method=args.outlier_method,
            cum_affines=cum_affines,
        )
        interpolate_missing(
            track, method=args.interp_method, cum_affines=cum_affines,
        )
        smooth_centroids(
            track,
            method=args.smooth_method,
            window=args.smooth_window,
            polyorder=args.smooth_poly,
            smooth_interpolated=not args.no_smooth_interpolated,
        )

        out_path = out_dir / f"{track.id}.json"
        n_out = export_track(track, str(out_path), dropped_keys=dropped)
        total += n_out

        n_interp = sum(1 for d in track.detections.values() if d.interpolated)
        print(f"{track.id}: {n_in} -> {n_out} dets ({n_interp} interpolated)")
        for step in track.history:
            print(f"    {step}")

    print(f"\nDone. {len(tracks)} tracks, {total} detections -> {out_dir}")

    if args.render_video:
        render_tracks_on_video(
            video_path=args.video,
            tracks=tracks,
            output_path=args.render_video,
            trail_length=args.trail_length,
            draw_bbox=args.draw_bbox,
            draw_obb=True,
            draw_centroid=True,
            draw_trail=True,
            draw_label=True,
            codec=args.codec,
        )


if __name__ == "__main__":
    main()