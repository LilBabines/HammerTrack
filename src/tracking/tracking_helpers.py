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

from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..tasks import TASK_OBB, normalize_task
from ..utils import OBBOX

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

# BotSort lives in the `bbox` sub-package and is lazily re-exported there.
# It is NOT reachable from `boxmot` nor from `boxmot.trackers`, which only
# export the BoxMOT facade and OccluBoost respectively.
#     boxmot.trackers.bbox            -> BoostTrack, BotSort, ByteTrack,
#                                        DeepOcSort, HybridSort, OccluBoost,
#                                        OcSort, SFSORT, StrongSort
try:
    from boxmot.trackers.bbox import BotSort
    BOXMOT_IMPORT_ERROR = None
except Exception as _exc:          # ImportError, but also AttributeError on
    BotSort = None                 # a version whose export map differs
    BOXMOT_IMPORT_ERROR = _exc

BOXMOT_AVAILABLE = BotSort is not None


# ---------------------------------------------------------------------------
# BoxMOT detection layouts (boxmot >= 22)
#
#   AABB : (x1, y1, x2, y2, conf, cls)          -> 6 columns
#   OBB  : (cx, cy, w, h, angle, conf, cls)     -> 7 columns
#
# The angle is in RADIANS: BoxMOT's own reference detector feeds ultralytics'
# ``result.obb.xywhr`` straight through, and that is radians regularized to
# [0, pi/2). Passing degrees (what ``cv2.minAreaRect`` returns) silently
# corrupts the motion model instead of raising.
#
# Tracker output is a ``TrackResults`` ndarray subclass:
#   AABB : (x1, y1, x2, y2, id, conf, cls, det_ind)         -> 8 columns
#   OBB  : (cx, cy, w, h, angle, id, conf, cls, det_ind)    -> 9 columns
#
# It exposes ``.id``, ``.conf``, ``.cls``, ``.det_ind``, ``.xyxy``, ``.xywha``
# and ``.is_obb``, which is what ``parse_track_results`` below relies on so the
# column offsets never have to be hard-coded.
# ---------------------------------------------------------------------------

AABB_DET_COLS = 6
OBB_DET_COLS = 7


# ==================== Tracker factory ====================

def build_tracker(cfg: dict, task: str = "detect"):
    """Build a BoxMOT tracker matching the project task.

    With ``task == "obb"`` the tracker runs in oriented mode, so the angle
    becomes part of the Kalman state instead of being discarded by an
    axis-aligned approximation. Any other task uses the AABB layout.

    Note: in boxmot >= 22 ``supports_obb`` is a *class attribute* advertising
    capability (True for BotSort), not a constructor argument. The real switch
    is ``is_obb``.
    """
    if not BOXMOT_AVAILABLE:
        # Never claim "not installed": the usual cause is an installed boxmot
        # whose export layout moved, and swallowing the real error turns a
        # one-line fix into a hunt.
        raise RuntimeError(
            "Could not import BotSort from boxmot.trackers.bbox "
            f"({type(BOXMOT_IMPORT_ERROR).__name__}: {BOXMOT_IMPORT_ERROR}).\n"
            "HammerTrack targets boxmot >= 22; check the installed version "
            "with `pip show boxmot`."
        )

    requested = cfg.get("tracker_type", "botsort")
    if requested != "botsort":
        print(
            f"[build_tracker] tracker_type='{requested}' not implemented — "
            f"falling back to BoTSORT."
        )

    is_obb = normalize_task(task) == TASK_OBB
    if is_obb and not getattr(BotSort, "supports_obb", False):
        raise RuntimeError(
            "This BoxMOT build reports no OBB support in BotSort. Upgrade "
            "boxmot (>= 22) or use a detect/pose project."
        )

    # ReID stays off: boxmot >= 22 wants a built ReID *model object* here, not
    # a weights path + device as older versions did. Turning it back on is a
    # deliberate change, not a config flag away.
    with_reid = bool(cfg.get("with_reid", False))
    if with_reid:
        print(
            "[build_tracker] with_reid=True ignored: boxmot >= 22 needs a "
            "ReID model object, none is wired up yet — running without "
            "appearance features."
        )
        with_reid = False

    return BotSort(
        reid_model=None,
        with_reid=with_reid,
        is_obb=is_obb,
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


def obb_to_xywhr_rad(poly: np.ndarray) -> Tuple[float, float, float, float, float]:
    """``(cx, cy, w, h, angle_rad)`` from a 4-point polygon, for the tracker.

    Kept separate from :func:`obb_to_xywhr`, which returns degrees and feeds
    the export files. ``cv2.minAreaRect`` reports degrees, BoxMOT expects
    radians; conflating the two is a silent 57x error on the angle.
    """
    pts = poly.reshape(4, 2).astype(np.float32)
    (cx, cy), (w, h), angle_deg = cv2.minAreaRect(pts)
    return float(cx), float(cy), float(w), float(h), float(np.deg2rad(angle_deg))


def boxes_to_dets(boxes: List[OBBOX], is_obb: bool) -> np.ndarray:
    """Pack annotations into the BoxMOT detection array for the given layout.

    Returns ``(N, 7)`` oriented rows when ``is_obb`` else ``(N, 6)``
    axis-aligned rows. Row order matches ``boxes``, which is what lets
    ``det_ind`` map track IDs straight back onto the source annotations.
    """
    cols = OBB_DET_COLS if is_obb else AABB_DET_COLS
    if not boxes:
        return np.empty((0, cols), dtype=np.float32)

    if is_obb:
        rows = [
            (*obb_to_xywhr_rad(b.poly), float(b.conf), float(b.cls_id))
            for b in boxes
        ]
    else:
        rows = [tuple(obb_to_aabb_row(b)) for b in boxes]
    return np.asarray(rows, dtype=np.float32)


def parse_track_results(tracks) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize a tracker output into ``(ids, det_inds, aabb_boxes)``.

    Works for both layouts. Uses the ``TrackResults`` accessors when present
    (boxmot >= 22) and falls back to positional columns for older builds, so
    the column offsets are never hard-coded at the call site.

    ``det_inds`` is ``-1`` where the tracker reported no source detection
    (a coasting track), and ``aabb_boxes`` is always axis-aligned so it can
    feed the IoU fallback whatever the layout.
    """
    empty = (np.empty(0, dtype=int), np.empty(0, dtype=int),
             np.empty((0, 4), dtype=np.float32))
    if tracks is None or len(tracks) == 0:
        return empty

    arr = np.asarray(tracks, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return empty

    # Preferred path: named accessors from boxmot >= 22.
    try:
        ids = np.asarray(tracks.id, dtype=int)
        det_inds = np.asarray(tracks.det_ind, dtype=int)
        boxes = np.asarray(tracks.xyxy, dtype=np.float32)
        return ids, det_inds, boxes
    except (AttributeError, IndexError, TypeError):
        pass

    # Fallback: infer the layout from the column count.
    is_obb = arr.shape[1] >= 9
    id_col = 5 if is_obb else 4
    det_col = 8 if is_obb else 7

    ids = arr[:, id_col].astype(int)
    det_inds = (arr[:, det_col].astype(int) if arr.shape[1] > det_col
                else np.full(len(arr), -1, dtype=int))

    if is_obb:
        # Convert (cx, cy, w, h, angle) to an enclosing axis-aligned box.
        cx, cy, w, h, ang = (arr[:, i] for i in range(5))
        cos_a, sin_a = np.abs(np.cos(ang)), np.abs(np.sin(ang))
        half_w = (w * cos_a + h * sin_a) / 2.0
        half_h = (w * sin_a + h * cos_a) / 2.0
        boxes = np.stack(
            [cx - half_w, cy - half_h, cx + half_w, cy + half_h], axis=1
        ).astype(np.float32)
    else:
        boxes = arr[:, :4].astype(np.float32)
    return ids, det_inds, boxes


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
    """Rewrite a STrack's Kalman mean so it matches a user-corrected AABB.

    boxmot >= 22 builds ``KalmanFilterXYWH``, so the measured block of the
    state is ``[cx, cy, w, h]`` -- plus the angle when the tracker runs in OBB
    mode -- and NOT the older ``xyah`` layout. Writing an aspect ratio into
    slot 2 (as the xyah code did) puts a value of ~3 where a width in pixels
    is expected: the predicted box collapses, the next IoU association finds
    nothing, and the track is silently replaced by a fresh ID.

    The angle slot is deliberately left untouched: moving a box does not
    reorient it, and zeroing it would flatten the orientation estimate.
    """
    mean = getattr(strack, "mean", None)
    if mean is None or len(mean) < 4:
        return

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = max(x2 - x1, 1e-6)
    h = max(y2 - y1, 1e-6)

    # Measured components: 4 for xywh, 5 for xywha. The velocity block is the
    # mirror half of the state vector.
    n_measured = len(mean) // 2 if len(mean) % 2 == 0 else 4

    mean[0] = cx
    mean[1] = cy
    mean[2] = w          # width in pixels, NOT w / h
    mean[3] = h
    # mean[4] stays as-is: it is the angle in OBB mode.

    # Drop the accumulated momentum so the next prediction starts from the
    # corrected position instead of drifting away from it.
    for i in range(n_measured, len(mean)):
        mean[i] = 0.0


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
    color_of: Optional[Callable[[int], Optional[Tuple[int, int, int]]]] = None,
    label_of: Optional[Callable[[int], Optional[str]]] = None,
) -> np.ndarray:
    """Render OBBs + (optionally) per-ID trails onto a copy of ``img_bgr``.

    ``color_of`` and ``label_of`` let the identity layer take over: pass
    ``IndividualStore.color_for_track`` and every fragment of one animal is
    drawn in that animal's colour, while tracks the resolver returns None for
    keep their own :func:`track_color` hue — so an unassigned fragment stays
    visually distinct instead of blending into a group.
    """
    out = img_bgr.copy()

    def resolve(tid: int) -> Tuple[int, int, int]:
        if color_of is not None and tid >= 0:
            override = color_of(tid)
            if override is not None:
                return tuple(int(c) for c in override)
        return track_color(tid)

    # ── Trails ──
    if show_trails:
        for tid, centers in trajectories.items():
            if tid < 0:
                continue
            color = resolve(tid)
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
            color, thick = resolve(tid), 4

        cv2.polylines(out, [pts], True, color, thick, cv2.LINE_AA)

        label = f"ID:{tid}" if tid >= 0 else "?"
        if label_of is not None and tid >= 0:
            extra = label_of(tid)
            if extra:
                label = f"{label} · {extra}"
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