"""
Post-processing pipeline for object tracker outputs.
- Merge multiple tracks (same individual)
- Deduplicate frames (keep highest confidence)
- Remove outlier detections
- Smooth centroids (Savitzky-Golay or moving average)
- Interpolate missing frames
"""

import json
import glob
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


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


def merge_tracks(tracks: list[dict], track_ids: list[int] | None = None) -> MergedTrack:
    """
    Merge multiple track JSONs into one.
    If track_ids is None, merge ALL tracks.
    For duplicate frames, keep the detection with highest confidence.
    """
    if track_ids is None:
        track_ids = [t["track_id"] for t in tracks]

    selected = [t for t in tracks if t["track_id"] in track_ids]
    merged = MergedTrack(track_ids=track_ids)

    for track in selected:
        for det in track["detections"]:
            d = Detection(
                frame=det["frame"],
                centroid=det["centroid"],
                bbox=det["bbox"],
                confidence=det["confidence"],
                class_id=det["class_id"],
                obb=det.get("obb"),
                source_track_id=track["track_id"],
            )
            f = d.frame
            # Keep best confidence for duplicate frames
            if f not in merged.detections or d.confidence > merged.detections[f].confidence:
                merged.detections[f] = d

    return merged


# ── 2. Outlier Removal ──────────────────────────────────────────────

def remove_outliers(
    merged: MergedTrack,
    max_jump_px: float = 150.0,
    z_threshold: float = 3.0,
    method: str = "jump",
) -> MergedTrack:
    """
    Remove outlier detections.

    Methods:
      - 'jump':  remove points where the displacement from the previous
                 frame exceeds max_jump_px.
      - 'zscore': remove points whose frame-to-frame displacement is
                  > z_threshold standard deviations from the mean.
    """
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
    """
    Smooth centroid positions.

    Methods:
      - 'savgol':  Savitzky-Golay filter (needs scipy)
      - 'moving_avg': simple moving average (no extra deps)
    """
    frames_sorted = sorted(merged.detections.keys())
    if len(frames_sorted) < window:
        return merged  # not enough points

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
) -> MergedTrack:
    """
    Fill in missing frames between first and last detection.
    Interpolates centroid and bbox. Confidence is set to 0 for
    interpolated frames.

    Methods: 'linear', 'cubic' (cubic needs scipy).
    """
    frames_sorted = sorted(merged.detections.keys())
    if len(frames_sorted) < 2:
        return merged

    f_min, f_max = frames_sorted[0], frames_sorted[-1]
    all_frames = set(range(f_min, f_max + 1))
    missing = all_frames - set(frames_sorted)

    if not missing:
        return merged

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
        merged.detections[int(f)] = Detection(
            frame=int(f),
            centroid=[float(interp_cx[i]), float(interp_cy[i])],
            bbox=interp_bbox[i].tolist(),
            confidence=0.0,  # flag as interpolated
            class_id=class_id,
            source_track_id=None,
        )

    return merged


# ── 5. Export ────────────────────────────────────────────────────────

def export_merged(merged: MergedTrack, output_path: str):
    """Export the merged + processed track to JSON."""
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


# ── Usage Example ────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Load all tracks
    tracks = load_tracks("projects/hammer/exports/per_track/track_*.json")

    # 2. Define which tracks to merge (same individual)
    #    e.g. tracks 1, 5 and 12 are the same person
    groups = [
        [2, 34, 40],
        [7, 32],
        [10, 15],
        [85],
        [65]
    ]

    for group in groups:
        # Merge
        merged = merge_tracks(tracks, track_ids=group)
        print(f"Track group {group}: {len(merged.detections)} detections after merge")

        # Remove outliers
        merged = remove_outliers(merged, max_jump_px=150, method="jump")
        print(f"  After outlier removal: {len(merged.detections)}")

        # Interpolate missing frames
        merged = interpolate_missing(merged, method="linear")
        print(f"  After interpolation: {len(merged.detections)}")

        # Smooth
        merged = smooth_centroids(merged, method="savgol", window=7, polyorder=2)

        # Export
        tag = "_".join(str(i) for i in group)
        export_merged(merged, f"projects/hammer/exports/track_postprocessed/merged_{tag}.json")
