"""
Post-processing pipeline for object tracker outputs.
- Merge multiple tracks (same individual) — Kalman-tail aware
- Deduplicate frames (keep highest confidence)
- Remove outlier detections
- Smooth centroids (Savitzky-Golay or moving average)
- Interpolate missing frames
"""

import json
import glob
import numpy as np
import cv2
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict


@dataclass
class Detection:
    frame: int
    centroid: list[float]
    bbox: list[float]
    confidence: float
    class_id: int
    obb: Optional[list] = None
    source_track_id: Optional[int] = None


@dataclass
class MergedTrack:
    track_ids: list[int]
    detections: dict[int, Detection] = field(default_factory=dict)  # frame -> Detection


# ── 1. Load & Merge ─────────────────────────────────────────────────

def load_tracks(pattern: str) -> list[dict]:
    """Load all track JSON files matching a glob pattern."""
    tracks = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            tracks.append(json.load(f))
    return tracks


def merge_tracks(
    tracks: list[dict],
    track_ids: list[int] | None = None,
    cum_affines: dict[int, np.ndarray] | None = None,
) -> MergedTrack:
    """
    Merge multiple track JSONs into one, processed **in order** of track_ids.

    If cum_affines is provided, gap interpolation is done in ref space
    (frame 0) so that camera motion doesn't corrupt interpolated centroids.
    """
    if track_ids is None:
        track_ids = [t["track_id"] for t in tracks]

    track_by_id = {t["track_id"]: t for t in tracks}
    selected = [track_by_id[tid] for tid in track_ids if tid in track_by_id]

    merged = MergedTrack(track_ids=track_ids)

    for idx, track in enumerate(selected):
        dets = sorted(track["detections"], key=lambda d: d["frame"])
        if not dets:
            continue

        real_frames = [d["frame"] for d in dets if d.get("obb") is not None]
        if not real_frames:
            continue

        last_real_frame = max(real_frames)

        next_first_frame = None
        if idx + 1 < len(selected):
            next_dets = selected[idx + 1]["detections"]
            if next_dets:
                next_first_frame = min(d["frame"] for d in next_dets)

        if next_first_frame is not None:
            cut_frame = min(last_real_frame, next_first_frame - 1)
        else:
            cut_frame = last_real_frame

        for det in dets:
            if det["frame"] > cut_frame:
                break
            d = Detection(
                frame=det["frame"],
                centroid=det["centroid"],
                bbox=det["bbox"],
                confidence=det["confidence"],
                class_id=det["class_id"],
                obb=det.get("obb"),
                source_track_id=track["track_id"],
            )
            if d.frame not in merged.detections or d.confidence > merged.detections[d.frame].confidence:
                merged.detections[d.frame] = d

        # --- Interpolation du gap vers la track suivante ----------------
        if next_first_frame is not None and next_first_frame > cut_frame + 1:
            anchor = None
            for det in reversed(dets):
                if det["frame"] <= cut_frame and det.get("obb") is not None:
                    anchor = det
                    break
            if anchor is None:
                continue

            next_det = min(
                selected[idx + 1]["detections"], key=lambda d: d["frame"]
            )

            f0 = anchor["frame"]
            f1 = next_det["frame"]
            span = f1 - f0 if f1 != f0 else 1

            use_cmc = (
                cum_affines is not None
                and f0 in cum_affines
                and f1 in cum_affines
            )

            if use_cmc:
                c0 = warp_point_to_ref(anchor["centroid"], cum_affines[f0])
                c1 = warp_point_to_ref(next_det["centroid"], cum_affines[f1])
            else:
                c0 = np.array(anchor["centroid"])
                c1 = np.array(next_det["centroid"])

            for f in range(cut_frame + 1, next_first_frame):
                if f in merged.detections:
                    continue
                t = (f - f0) / span
                c_ref = c0 + t * (c1 - c0)

                if use_cmc and f in cum_affines:
                    centroid = warp_point_from_ref(c_ref, cum_affines[f]).tolist()
                else:
                    centroid = c_ref.tolist()

                merged.detections[f] = Detection(
                    frame=f,
                    centroid=centroid,
                    bbox=list(anchor["bbox"]),
                    confidence=0.0,
                    class_id=anchor["class_id"],
                    obb=None,
                    source_track_id=None,
                )

    return merged


# ── 2. Outlier Removal ──────────────────────────────────────────────

def remove_outliers(
    merged: MergedTrack,
    max_jump_px: float = 150.0,
    z_threshold: float = 3.0,
    method: str = "jump",
) -> MergedTrack:
    frames_sorted = sorted(merged.detections.keys())
    if len(frames_sorted) < 3:
        return merged

    centroids = np.array([merged.detections[f].centroid for f in frames_sorted])

    if method == "jump":
        keep = {frames_sorted[0]}
        for i in range(1, len(frames_sorted)):
            dist = np.linalg.norm(centroids[i] - centroids[i - 1])
            if dist <= max_jump_px:
                keep.add(frames_sorted[i])

    elif method == "zscore":
        diffs = np.linalg.norm(np.diff(centroids, axis=0), axis=1)
        mu, sigma = diffs.mean(), diffs.std() + 1e-9
        z = (diffs - mu) / sigma
        keep = {frames_sorted[0]}
        for i, zi in enumerate(z):
            if abs(zi) <= z_threshold:
                keep.add(frames_sorted[i + 1])
    else:
        raise ValueError(f"Unknown method: {method}")

    merged.detections = {f: d for f, d in merged.detections.items() if f in keep}
    return merged


# ── 3. Smoothing ────────────────────────────────────────────────────

def smooth_centroids(
    merged: MergedTrack,
    method: str = "savgol",
    window: int = 5,
    polyorder: int = 2,
) -> MergedTrack:
    frames_sorted = sorted(merged.detections.keys())
    if len(frames_sorted) < window:
        return merged

    centroids = np.array([merged.detections[f].centroid for f in frames_sorted])

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
        sx = np.convolve(centroids[:, 0], kernel, mode="same")
        sy = np.convolve(centroids[:, 1], kernel, mode="same")
        smoothed = np.column_stack([sx, sy])
    else:
        raise ValueError(f"Unknown method: {method}")

    for i, f in enumerate(frames_sorted):
        merged.detections[f].centroid = smoothed[i].tolist()

    return merged


# ── 4. Interpolation ────────────────────────────────────────────────

def interpolate_missing(
    merged: MergedTrack,
    method: str = "linear",
    cum_affines: dict[int, np.ndarray] | None = None,
) -> MergedTrack:
    """
    Fill in missing frames between first and last detection.
    If cum_affines is provided, interpolation is done in ref space.
    """
    frames_sorted = sorted(merged.detections.keys())
    if len(frames_sorted) < 2:
        return merged

    f_min, f_max = frames_sorted[0], frames_sorted[-1]
    all_frames = set(range(f_min, f_max + 1))
    missing = all_frames - set(frames_sorted)

    if not missing:
        return merged

    use_cmc = (
        cum_affines is not None
        and all(f in cum_affines for f in frames_sorted)
    )

    if use_cmc:
        centroids = np.array([
            warp_point_to_ref(merged.detections[f].centroid, cum_affines[f])
            for f in frames_sorted
        ])
    else:
        centroids = np.array([merged.detections[f].centroid for f in frames_sorted])

    bboxes = np.array([merged.detections[f].bbox for f in frames_sorted])
    frames_arr = np.array(frames_sorted, dtype=float)
    missing_arr = np.array(sorted(missing), dtype=float)

    if method == "linear":
        interp_cx = np.interp(missing_arr, frames_arr, centroids[:, 0])
        interp_cy = np.interp(missing_arr, frames_arr, centroids[:, 1])
        interp_bbox = np.column_stack([
            np.interp(missing_arr, frames_arr, bboxes[:, i]) for i in range(4)
        ])

    elif method == "cubic":
        from scipy.interpolate import CubicSpline
        cs_cx = CubicSpline(frames_arr, centroids[:, 0])
        cs_cy = CubicSpline(frames_arr, centroids[:, 1])
        interp_cx = cs_cx(missing_arr)
        interp_cy = cs_cy(missing_arr)
        interp_bbox = np.column_stack([
            CubicSpline(frames_arr, bboxes[:, i])(missing_arr) for i in range(4)
        ])
    else:
        raise ValueError(f"Unknown method: {method}")

    class_id = merged.detections[frames_sorted[0]].class_id

    for i, f in enumerate(missing_arr.astype(int)):
        f = int(f)
        c_ref = np.array([float(interp_cx[i]), float(interp_cy[i])])

        if use_cmc and f in cum_affines:
            c_img = warp_point_from_ref(c_ref, cum_affines[f])
        else:
            c_img = c_ref

        merged.detections[f] = Detection(
            frame=f,
            centroid=c_img.tolist(),
            bbox=interp_bbox[i].tolist(),
            confidence=0.0,
            class_id=class_id,
            source_track_id=None,
        )

    return merged


# ── 5. Export ────────────────────────────────────────────────────────

def export_merged(merged: MergedTrack, output_path: str):
    frames_sorted = sorted(merged.detections.keys())
    detections = []
    for f in frames_sorted:
        d = merged.detections[f]
        det = {
            "frame": d.frame,
            "centroid": d.centroid,
            "bbox": d.bbox,
            "confidence": d.confidence,
            "class_id": d.class_id,
            "interpolated": d.confidence == 0.0,
        }
        if d.obb is not None:
            det["obb"] = d.obb
        if d.source_track_id is not None:
            det["source_track_id"] = d.source_track_id
        detections.append(det)

    out = {
        "merged_track_ids": merged.track_ids,
        "num_detections": len(detections),
        "first_frame": frames_sorted[0] if frames_sorted else None,
        "last_frame": frames_sorted[-1] if frames_sorted else None,
        "detections": detections,
    }
    Path(output_path).write_text(json.dumps(out, indent=2))
    print(f"Exported {len(detections)} detections to {output_path}")


# ── 6. Visualization ─────────────────────────────────────────────────

TRACK_COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),
    (0, 128, 255), (128, 255, 0), (255, 0, 128), (0, 255, 128),
    (200, 200, 0), (200, 0, 200), (0, 200, 200), (100, 100, 255),
    (255, 100, 100), (100, 255, 100),
]


def render_tracks_on_video(
    video_path: str,
    all_merged: list[tuple[str, list[int], MergedTrack]],
    output_path: str,
    trail_length: int = 30,
    draw_bbox: bool = True,
    draw_obb: bool = True,
    draw_centroid: bool = True,
    draw_trail: bool = True,
    draw_label: bool = True,
    codec: str = "mp4v",
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_index: dict[int, list[tuple[int, Detection]]] = defaultdict(list)
    for tidx, (name, group, merged) in enumerate(all_merged):
        for f, det in merged.detections.items():
            frame_index[f].append((tidx, det))

    print(f"Rendering {total_frames} frames → {output_path}")

    trails: dict[int, list[tuple[int, int]]] = defaultdict(list)

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        entries = frame_index.get(frame_num, [])

        for tidx, det in entries:
            color = TRACK_COLORS[tidx % len(TRACK_COLORS)]
            is_interp = det.confidence == 0.0
            cx, cy = int(det.centroid[0]), int(det.centroid[1])

            trails[tidx].append((cx, cy))
            if len(trails[tidx]) > trail_length:
                trails[tidx] = trails[tidx][-trail_length:]

            if draw_trail and len(trails[tidx]) > 1:
                pts = trails[tidx]
                for i in range(1, len(pts)):
                    alpha = i / len(pts)
                    thick = max(1, int(alpha * 3))
                    cv2.line(frame, pts[i - 1], pts[i], color, thick, cv2.LINE_AA)

            if draw_bbox and det.bbox:
                b = det.bbox
                if len(b) == 4:
                    x1, y1, x2, y2 = map(int, b)
                    if is_interp:
                        _draw_dashed_rect(frame, (x1, y1), (x2, y2), color, 1, 8)
                    else:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            if draw_obb and det.obb is not None and len(det.obb) >= 4:
                pts_obb = np.array(det.obb, dtype=np.float32)
                if pts_obb.ndim == 2 and pts_obb.shape[0] >= 4:
                    pts_int = pts_obb[:4].astype(np.int32)
                    cv2.polylines(frame, [pts_int], True, color, 2, cv2.LINE_AA)

            if draw_centroid:
                radius = 3 if is_interp else 5
                cv2.circle(frame, (cx, cy), radius, color, -1, cv2.LINE_AA)

            if draw_label:
                # ── Utilise le nom de fichier comme label ──
                label = all_merged[tidx][0]
                if is_interp:
                    label += " (interp)"
                cv2.putText(
                    frame, label,
                    (cx + 8, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
                )

        out.write(frame)
        frame_num += 1

        if frame_num % 500 == 0:
            print(f"  {frame_num}/{total_frames} frames rendered")

    cap.release()
    out.release()
    print(f"Done. Output: {output_path}")


def _draw_dashed_rect(img, pt1, pt2, color, thickness=1, dash_len=8):
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
            s = i
            e = min(i + dash_len, dist)
            sp = (int(x1 + dx * s), int(y1 + dy * s))
            ep = (int(x1 + dx * e), int(y1 + dy * e))
            cv2.line(img, sp, ep, color, thickness, cv2.LINE_AA)


# ── CMC helpers ──────────────────────────────────────────────────────

def build_cum_affine(cmc, start, end):
    cum = {}
    M_acc = np.eye(3, dtype=np.float64)
    for f in range(start, end):
        fk = str(f)
        if fk in cmc:
            A = np.eye(3, dtype=np.float64)
            A[:2, :] = np.array(cmc[fk])
            M_acc = A @ M_acc
        cum[f] = M_acc[:2, :].copy()
    return cum


def warp_point_to_ref(pt, cum_affine_f):
    A = np.eye(3, dtype=np.float64)
    A[:2, :] = cum_affine_f
    A_inv = np.linalg.inv(A)
    p = A_inv @ np.array([pt[0], pt[1], 1.0])
    return p[:2]


def warp_point_from_ref(pt, cum_affine_f):
    A = cum_affine_f
    return A @ np.array([pt[0], pt[1], 1.0])


# ── Usage Example ────────────────────────────────────────────────────

if __name__ == "__main__":
    # 0. Load CMC & build cumulative affines
    with open("projects/hammer/exports/2021_057/cmc_transforms.json") as f:
        cmc = json.load(f)
    cum_affines = build_cum_affine(cmc, start=0, end=20000)

    # 1. Load all tracks
    tracks = load_tracks("projects/hammer/exports/2021_057/per_track/track_*.json")

    # 2. Define which tracks to merge (same individual)
    groups = [
        [2, 34, 40],
    ]

    # groups = [
    #     [2],
    #     [6],
    #     [1, 132],
    #     [19, 48, 72, 136],
    #     [139, 242, 413],
    #     [3, 50],
    #     [12,  177, 243, 256],
    #     [5, 77],
    #     [13, 116, 155],
    #     [18, 161, 186, 301, 332, 348, 384, 477],
    #     [10],
    #     [9, 120],
    #     [25, 78],
    #     [31],
    #     [23, 306, 361, 615],
    #     [17, 73, 231, 259, 281, 309, 412],
    #     [11],
    #     [7,  105, 125, 185, 246],
    #     [16, 47, 57, 82, 98, 107, 112, 152, 352, 631],
    #     [28,60],
    #     [24,51,63,67,121,128,133,142,168,252,275,364,365],
    #     [241,438],
    #     [212,395],
    #     [213,380],
    #     [38],
    #     [4,70,88],
    #     [40],
    #     [151,270],
    #     [15,89,106],
    #     [110],
    #     [134,162,174,191],
    #     [137,148],
    #     [163,172,189,285],
    #     [166,176,233],
    #     [180,235,250,254],
    #     [202,229,249,277,313],
    #     [407,414,654],
    #     [181,228],
    #     [195,253,288,340,421],
    #     [198,263,297],
    #     [224],[225,374],
    #     [240],
    #     [209,310,325],
    #     [316],
    #     [318],
    #     [319],
    #     [323],
    #     [327],
    #     [337],
    #     [345],
    #     [317],
    #     [377],
    #     [379,458],
    #     [387,417,432],
    #     [359,376,388,425,474],
    #     [386,433],
    #     [443],
    #     [462,496],
    #     [453,515],
    #     [488],
    #     [490],
    #     [491,534,548,584,597],
    #     [494],
    #     [416],
    #     [481],
    #     [484],
    #     [495,513],
    #     [498,541,543,554,660],
    #     [499],
    #     [502,520],
    #     [503,588],
    #     [383,422],
    #     [493,521,564,596,604,611,667],
    #     [392,401,402],
    #     [430,440],
    #     [435],
    #     [434,455],
    #     [492,512,518,528,533,538,542,558,562,624],
    #     [547,568],
    #     [629,669],
    #     [390,394,400,406,420],
    #     [549,559,581,658],
    #     [552,576],
    #     [578,595],
    #     [347,544,603,619,663,671],
    #     [553,587,661],
    #     [586,620],
    #     [589],
    #     [556,599],
    #     [630,655],
    #     [632,637],
    #     [618,646]
    # ]
    flat = [x for sub in groups for x in sub]
    has_duplicates = len(flat) != len(set(flat))
    from collections import Counter
    dupes = [x for x, c in Counter(flat).items() if c > 1]
    print(dupes)
    assert len(dupes) == 0

    all_merged: list[tuple[str, list[int], MergedTrack]] = []

    for num_i, group in enumerate(groups):
        if not group:
            continue

        merged = merge_tracks(tracks, track_ids=group, cum_affines=cum_affines)
        print(f"Track group {group}: {len(merged.detections)} detections after merge")

        merged = remove_outliers(merged, max_jump_px=150, method="jump")
        print(f"  After outlier removal: {len(merged.detections)}")

        merged = interpolate_missing(merged, method="linear", cum_affines=cum_affines)
        print(f"  After interpolation: {len(merged.detections)}")

        merged = smooth_centroids(merged, method="savgol", window=7, polyorder=2)

        # ── Le nom de fichier sert aussi de label dans le rendu ──
        filename = f"shark_{num_i}"
        export_merged(merged, f"projects/hammer/exports/2021_057/postp_tracks/{filename}.json")

        all_merged.append((filename, group, merged))

    # 3. Render toutes les tracks sur la vidéo
    render_tracks_on_video(
        video_path="clips/selected/SIMP2021_057-13_10-16_25.mp4",
        all_merged=all_merged,
        output_path="projects/hammer/exports/2021_057/overlay_pp.mp4",
        trail_length=10,
        draw_bbox=False,
        draw_obb=True,
        draw_centroid=True,
        draw_trail=True,
        draw_label=True,
    )