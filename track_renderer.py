"""
Render post-processed tracks onto a video and export.
"""

import cv2
import numpy as np
from typing import Optional
from collections import defaultdict


# ── Couleurs par track (cycle automatique) ───────────────────────────

PALETTE = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),
    (0, 128, 255), (128, 255, 0),
]


def get_color(track_idx: int) -> tuple:
    return PALETTE[track_idx % len(PALETTE)]


# ── Drawing helpers ──────────────────────────────────────────────────

def draw_bbox(frame, bbox, color, thickness=2):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)


def draw_obb(frame, obb, color, thickness=2):
    pts = np.array(obb, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)


def draw_centroid(frame, centroid, color, radius=4):
    cx, cy = int(centroid[0]), int(centroid[1])
    cv2.circle(frame, (cx, cy), radius, color, -1)


def draw_trail(frame, trail_points, color, thickness=2, max_len=30):
    """Draw the trajectory trail (last N points)."""
    pts = trail_points[-max_len:]
    for i in range(1, len(pts)):
        alpha = i / len(pts)  # fade: older = thinner
        t = max(1, int(thickness * alpha))
        p1 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
        p2 = (int(pts[i][0]), int(pts[i][1]))
        cv2.line(frame, p1, p2, color, t)


def draw_label(frame, text, position, color, font_scale=0.6, thickness=2):
    x, y = int(position[0]), int(position[1]) - 10
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.rectangle(frame, (x, y - h - 4), (x + w + 4, y + 4), color, -1)
    cv2.putText(frame, text, (x + 2, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), thickness)


# ── Main renderer ────────────────────────────────────────────────────

def render_tracks(
    source,                       # VideoSource instance
    merged_tracks: list,          # list of MergedTrack
    output_path: str,
    # --- display options ---
    show_bbox: bool = True,
    show_obb: bool = False,
    show_centroid: bool = True,
    show_trail: bool = True,
    show_label: bool = True,
    trail_length: int = 30,
    show_interpolated: bool = True,  # style différent pour les interpolées
    # --- video options ---
    codec: str = "mp4v",
    fps: Optional[float] = None,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
):
    """
    Render all merged tracks onto the source video and write to output_path.

    Parameters
    ----------
    source : VideoSource
        The original video source.
    merged_tracks : list[MergedTrack]
        Post-processed tracks from the pipeline.
    output_path : str
        Path for the output video (.mp4, .avi).
    """
    fps = fps or source.fps()
    end_frame = end_frame or source.count()

    # Pre-index: frame -> [(track_idx, Detection)]
    frame_index = defaultdict(list)
    for t_idx, mt in enumerate(merged_tracks):
        for f, det in mt.detections.items():
            if start_frame <= f < end_frame:
                frame_index[f].append((t_idx, det))

    # Trail history per track
    trails: dict[int, list] = defaultdict(list)

    # Init writer with first frame to get dimensions
    first_frame = source.read(start_frame)
    if first_frame is None:
        raise RuntimeError("Cannot read first frame")
    h, w = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    try:
        for f_idx in range(start_frame, end_frame):
            frame = source.read(f_idx)
            if frame is None:
                continue

            for t_idx, det in frame_index.get(f_idx, []):
                color = get_color(t_idx)
                is_interp = det.confidence == 0.0

                # Interpolated detections: dashed style / dimmer color
                if is_interp and not show_interpolated:
                    continue

                draw_color = tuple(c // 2 for c in color) if is_interp else color

                # Bounding box
                if show_bbox and det.bbox:
                    thickness = 1 if is_interp else 2
                    draw_bbox(frame, det.bbox, draw_color, thickness)

                # Oriented bounding box
                if show_obb and det.obb:
                    draw_obb(frame, det.obb, draw_color)

                # Centroid
                if show_centroid:
                    r = 3 if is_interp else 5
                    draw_centroid(frame, det.centroid, draw_color, radius=r)

                # Trail
                trails[t_idx].append(det.centroid)
                if show_trail:
                    draw_trail(frame, trails[t_idx], color,
                               thickness=2, max_len=trail_length)

                # Label
                if show_label:
                    ids = merged_tracks[t_idx].track_ids
                    tag = f"ID {ids[0]}" if len(ids) == 1 else f"IDs {ids}"
                    conf = f" {det.confidence:.0%}" if not is_interp else " (interp)"
                    draw_label(frame, f"{tag}{conf}", det.centroid, draw_color)

            writer.write(frame)

            if f_idx % 500 == 0:
                pct = (f_idx - start_frame) / (end_frame - start_frame) * 100
                print(f"  Rendering: frame {f_idx}/{end_frame} ({pct:.0f}%)")

    finally:
        writer.release()

    print(f"Done → {output_path} ({end_frame - start_frame} frames)")


# ── Usage ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from track_postprocess import load_tracks, merge_tracks, remove_outliers
    from track_postprocess import interpolate_missing, smooth_centroids
    from src.utils import VideoSource
    
    # 1. Open video
    source = VideoSource("clips/selected/SIMP2021_057-13_10-16_25.mp4")

    # 2. Load & process tracks
    tracks = load_tracks("projects/hammer/exports/per_track/track_*.json")

    groups = [
        [2, 34, 40],
        [7, 32],
        [10, 15],
        [85],
        [65]
    ]

    processed = []
    for group in groups:
        mt = merge_tracks(tracks, track_ids=group)
        mt = remove_outliers(mt, max_jump_px=150)
        mt = interpolate_missing(mt, method="linear")
        mt = smooth_centroids(mt, method="savgol", window=7)
        processed.append(mt)

    # 3. Render
    render_tracks(
        source,
        processed,
        output_path="projects/hammer/exports/SIMP2021_057-13_10-16_25_tracked_postP.mp4",
        show_bbox=True,
        show_trail=True,
        trail_length=40,
        show_obb=False,
    )

    source.close()
