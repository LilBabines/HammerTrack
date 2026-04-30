"""Tracking helpers — pure functions used by the tracking page and workers.

Contents:
* Tracker factory (``build_tracker``).
* Colour palette for track IDs (``track_color``).
* OBB ↔ AABB / xywhr conversions.
* IoU helper.
* In-place STrack Kalman patch (``update_strack_bbox``).
* Trajectory extraction from a BoxMOT tracker's internal STracks.
* Drawing routine for a tracked frame (``draw_tracked_annotations``).

Nothing here depends on Qt; all functions are CPU-side numpy/cv2.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..utils import OBBOX

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    from boxmot.trackers import BotSort
    BOXMOT_AVAILABLE = True
except ImportError:
    BOXMOT_AVAILABLE = False


# ==================== Tracker factory ====================

def build_tracker(cfg: dict):
    """Build a BoxMOT tracker from a project config dict.

    Currently always returns BoTSORT — ``cfg["tracker_type"]`` is read for
    future extension but only ``"botsort"`` is wired up. A non-botsort value
    triggers a printed warning, then falls back to BoTSORT.
    """
    if not BOXMOT_AVAILABLE:
        raise RuntimeError("boxmot is not installed. pip install boxmot")

    requested = cfg.get("tracker_type", "botsort")
    if requested != "botsort":
        print(
            f"[build_tracker] tracker_type='{requested}' not implemented — "
            f"falling back to BoTSORT."
        )

    device = "cuda:0" if (torch is not None and torch.cuda.is_available()) else "cpu"
    with_reid = bool(cfg.get("with_reid", True))
    reid_weights = cfg.get("reid_weights", "osnet_x0_25_msmt17.pt") if with_reid else ""

    return BotSort(
        reid_weights=reid_weights,
        device=device,
        half=False,
        with_reid=with_reid,
        track_high_thresh=float(cfg.get("track_high_thresh", 0.6)),
        track_low_thresh=float(cfg.get("track_low_thresh", 0.1)),
        new_track_thresh=float(cfg.get("new_track_thresh", 0.7)),
        track_buffer=int(cfg.get("track_buffer", 30)),
        match_thresh=float(cfg.get("match_thresh", 0.8)),
        proximity_thresh=float(cfg.get("proximity_thresh", 0.5)),
        appearance_thresh=float(cfg.get("appearance_thresh", 0.25)),
    )


# ==================== Colour palette ====================

def _make_palette(n: int = 64) -> List[Tuple[int, int, int]]:
    pal = []
    for i in range(n):
        h = int(180 * i / n)
        s = 200 + (i % 3) * 25
        v = 220 + (i % 2) * 35
        bgr = cv2.cvtColor(
            np.array([[[h, min(s, 255), min(v, 255)]]], dtype=np.uint8),
            cv2.COLOR_HSV2BGR,
        )[0, 0]
        pal.append(tuple(int(c) for c in bgr))
    return pal


_PALETTE = _make_palette(64)


def track_color(tid: int) -> Tuple[int, int, int]:
    if tid < 0:
        return (0, 200, 255)
    return _PALETTE[tid % len(_PALETTE)]


# ==================== OBB / AABB conversions ====================

def obb_to_aabb_row(box: OBBOX) -> np.ndarray:
    """Return a single AABB detection row ``[x1, y1, x2, y2, conf, cls_id]``."""
    pts = box.poly.reshape(-1, 2)
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    return np.array(
        [x1, y1, x2, y2, box.conf, box.cls_id], dtype=np.float32
    )


def obb_centroid(box: OBBOX) -> Tuple[float, float]:
    pts = box.poly.reshape(-1, 2)
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())


def obb_to_xywhr(poly: np.ndarray) -> Tuple[float, float, float, float, float]:
    """Return ``(cx, cy, w, h, angle_deg)`` from a 4-point OBB polygon."""
    pts = poly.reshape(4, 2).astype(np.float32)
    (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
    return (
        round(cx, 2), round(cy, 2),
        round(w, 2), round(h, 2),
        round(angle, 2),
    )


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two arrays of AABBs ``[x1, y1, x2, y2]``."""
    x1 = np.maximum(a[:, 0:1], b[:, 0:1].T)
    y1 = np.maximum(a[:, 1:2], b[:, 1:2].T)
    x2 = np.minimum(a[:, 2:3], b[:, 2:3].T)
    y2 = np.minimum(a[:, 3:4], b[:, 3:4].T)
    inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-6)


# ==================== STrack patching ====================

def update_strack_bbox(strack, x1: float, y1: float, x2: float, y2: float):
    """Best-effort update of a STrack's internal Kalman mean to match a new AABB.

    BotSort state format is typically
    ``[cx, cy, aspect_ratio, h, vx, vy, va, vh]``. Velocities are zeroed so
    the next prediction does not drift from old momentum.
    """
    mean = getattr(strack, "mean", None)
    if mean is None or len(mean) < 4:
        return
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = max(x2 - x1, 1e-6)
    h = max(y2 - y1, 1e-6)
    mean[0] = cx
    mean[1] = cy
    mean[2] = w / h
    mean[3] = h
    if len(mean) >= 8:
        mean[4] = 0.0
        mean[5] = 0.0
        mean[6] = 0.0
        mean[7] = 0.0


# ==================== Trajectory extraction ====================

def extract_trajectories_from_tracker(
    tracker,
) -> Dict[int, List[Tuple[float, float]]]:
    """Walk the tracker's internal STracks and return per-ID centre histories."""
    trajectories: Dict[int, List[Tuple[float, float]]] = {}
    seen = set()
    all_stracks = []
    for attr in ("active_tracks", "lost_stracks"):
        pool = getattr(tracker, attr, None)
        if not pool:
            continue
        for st in pool:
            obj_id = id(st)
            if obj_id not in seen:
                seen.add(obj_id)
                all_stracks.append(st)

    for strack in all_stracks:
        tid = int(getattr(strack, "id", -1))
        if tid < 0:
            continue
        centers = []
        obs = getattr(strack, "history_observations", None)
        if obs:
            for box in obs:
                box_arr = np.asarray(box, dtype=np.float32).ravel()
                if len(box_arr) >= 4:
                    cx = float((box_arr[0] + box_arr[2]) / 2)
                    cy = float((box_arr[1] + box_arr[3]) / 2)
                    centers.append((cx, cy))
        mean = getattr(strack, "mean", None)
        if mean is not None:
            try:
                xyxy = strack.xyxy
                box_arr = np.asarray(xyxy, dtype=np.float32).ravel()
                if len(box_arr) >= 4:
                    cx = float((box_arr[0] + box_arr[2]) / 2)
                    cy = float((box_arr[1] + box_arr[3]) / 2)
                    centers.append((cx, cy))
            except Exception:
                pass
        if centers:
            trajectories[tid] = centers
    return trajectories


def extract_cmc_matrix(tracker) -> Optional[np.ndarray]:
    """Best-effort extraction of the Camera Motion Compensation warp matrix
    from a BoxMOT tracker. Returns a 2×3 (affine) or 3×3 (homography) numpy
    array, or ``None`` if unavailable.
    """
    mat = getattr(tracker, "warp", None)
    if mat is not None and isinstance(mat, np.ndarray):
        return mat.copy()
    return None


# ==================== Drawing ====================

def draw_tracked_annotations(
    img_bgr: np.ndarray,
    annots: List[OBBOX],
    selected_idx: Optional[int],
    trajectories: Dict[int, List[Tuple[float, float]]],
    trail_length: int = 60,
    show_trails: bool = True,
) -> np.ndarray:
    """Render OBBs + (optionally) per-ID trails onto a copy of ``img_bgr``."""
    out = img_bgr.copy()

    # ── Trails ──
    if show_trails:
        for tid, centers in trajectories.items():
            if tid < 0:
                continue
            color = track_color(tid)
            recent = (centers[-trail_length:]
                      if len(centers) > trail_length else centers)
            if len(recent) < 2:
                continue
            coords = np.array(recent, dtype=np.int32)
            n = len(coords)
            for j in range(1, n):
                alpha = j / n
                thick = max(1, int(1 + 2 * alpha))
                c = tuple(int(v * (0.3 + 0.7 * alpha)) for v in color)
                cv2.line(
                    out, tuple(coords[j - 1]), tuple(coords[j]),
                    c, thick, cv2.LINE_AA,
                )
            cv2.circle(out, tuple(coords[-1]), 4, color, -1, cv2.LINE_AA)

    # ── Boxes + labels ──
    for i, b in enumerate(annots):
        if b.deleted:
            continue
        pts = b.poly.reshape(-1, 2).astype(int)
        tid = b.track_id

        if selected_idx is not None and i == selected_idx:
            color, thick = (255, 0, 255), 5
        else:
            color, thick = track_color(tid), 4

        cv2.polylines(out, [pts], True, color, thick, cv2.LINE_AA)

        label = f"ID:{tid}" if tid >= 0 else "?"
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        )
        x0, y0 = int(pts[0, 0]), int(pts[0, 1]) - 6
        cv2.rectangle(
            out, (x0, y0 - th - 4), (x0 + tw + 6, y0 + 2), color, -1
        )
        cv2.putText(
            out, label, (x0 + 3, y0 - 1),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA,
        )

    return out
