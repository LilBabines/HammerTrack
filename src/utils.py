from typing import Optional, List, Dict, Sequence, Tuple
from dataclasses import dataclass

import cv2
import numpy as np
from PySide6 import QtGui

import copy
import warnings
import os
import re
from pathlib import Path

from .tasks import TASK_OBB, TASK_POSE, KPT_DIMS, normalize_task


ORIGIN_MANUAL = "manual"      # drawn by hand in this session
ORIGIN_DATASET = "dataset"    # reloaded from an exported label file
ORIGIN_MODEL = "model"        # produced by inference


def ensure_bgr_u8(img: np.ndarray) -> np.ndarray:
    """Convert an image (8/16-bit, mono/RGBA) to BGR uint8 for display and processing.
       - 16-bit → scaled to 0..255 (min-max normalization)
       - 1 channel → BGR
       - 4 channels (BGRA) → BGR
    """
    if img is None:
        return img

    # 16-bit → 8-bit via min-max scaling
    if img.dtype == np.uint16:
        i_min, i_max = int(img.min()), int(img.max())
        if i_max > i_min:
            img8 = ((img - i_min) * 255.0 / (i_max - i_min)).astype(np.uint8)
        else:
            img8 = (img / 256).astype(np.uint8)
        img = img8
    elif img.dtype != np.uint8:
        img = cv2.convertScaleAbs(img)

    # Convert grayscale or BGRA to BGR
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def cvimg_to_qimage(img_bgr: np.ndarray) -> QtGui.QImage:
    """Convert a BGR numpy array to a QImage (RGB888 format)."""
    if img_bgr is None:
        return QtGui.QImage()
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = img_rgb.shape
    return QtGui.QImage(img_rgb.data, w, h, ch * w, QtGui.QImage.Format.Format_RGB888)


# ---------------------------------------------------------------------------
# Annotation data classes
# ---------------------------------------------------------------------------

@dataclass
class PolyClass:
    """Annotation container for generic polygonal regions.

    ``keypoints`` is only populated for the ``pose`` task, where ``poly`` holds
    the (axis-aligned) bounding box and ``keypoints`` the N body points. It
    stays ``None`` for the ``detect`` and ``obb`` tasks.
    """
    poly: np.ndarray                          # shape (n, 2) float32, image coords
    cls_id: int
    conf: float
    verified: bool = False
    deleted: bool = False
    keypoints: Optional[np.ndarray] = None    # shape (N, 2) float32, image coords
    origin: str = ORIGIN_MODEL

    def has_keypoints(self) -> bool:
        return self.keypoints is not None and len(self.keypoints) > 0
    
    def is_ground_truth(self) -> bool:
        """True for annotations that belong in the dataset unconditionally."""
        return self.origin in (ORIGIN_MANUAL, ORIGIN_DATASET)
    
    def to_json(self) -> dict:
        return {
            "poly": self.poly.tolist(),
            "cls_id": int(self.cls_id),
            "conf": float(self.conf),
            "verified": bool(self.verified),
            "deleted": bool(self.deleted),
            "keypoints": (self.keypoints.tolist()
                          if self.has_keypoints() else None),
            "origin": self.origin,
        }


@dataclass
class OBBOX(PolyClass):
    """Annotation container for oriented bounding boxes (4-point polygons)."""
    poly: np.ndarray           # shape (4, 2) float32, image coordinates
    track_id: int = -1 


# ---------------------------------------------------------------------------
# Mask I/O and conversion
# ---------------------------------------------------------------------------

def load_mask_png(path: str) -> Optional[np.ndarray]:
    """Load a mask PNG (RGBA or grayscale) and return a single-channel uint8 array.
       - RGBA/BGRA → uses the alpha channel
       - Grayscale → thresholds dark pixels (< 50) as foreground (255)
    """
    mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        warnings.warn(f"Could not read mask image at {path}.")
        return None

    if mask.ndim == 3 and mask.shape[2] == 4:
        # RGBA / BGRA → extract alpha channel
        m = mask[..., 3].astype(np.uint8)
    elif mask.ndim == 2:
        # Grayscale: dark pixels are foreground
        m = np.where(mask < 50, 255, 0).astype(np.uint8)
    else:
        warnings.warn(f"Mask at {path} has unsupported shape {mask.shape}.")
        m = None

    return m

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def rect_to_poly_xyxy(x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    """Convert an axis-aligned box (x1, y1, x2, y2) to a 4-point polygon."""
    return np.array([[x1, y1],
                     [x2, y1],
                     [x2, y2],
                     [x1, y2]], dtype=np.float32)


def keypoints_to_bbox_poly(
    keypoints: np.ndarray,
    margin_ratio: float = 0.12,
    min_size: float = 4.0,
    img_w: Optional[int] = None,
    img_h: Optional[int] = None,
) -> np.ndarray:
    """Derive an axis-aligned box (as a 4-point polygon) from keypoints.

    The tight keypoint extent is padded by ``margin_ratio`` of the larger side
    so the box actually contains the animal rather than just its landmarks.
    The result is clamped to the image when ``img_w`` / ``img_h`` are given.
    """
    pts = np.asarray(keypoints, dtype=np.float32).reshape(-1, 2)
    x1, y1 = float(pts[:, 0].min()), float(pts[:, 1].min())
    x2, y2 = float(pts[:, 0].max()), float(pts[:, 1].max())

    margin = max(x2 - x1, y2 - y1) * margin_ratio
    x1, y1, x2, y2 = x1 - margin, y1 - margin, x2 + margin, y2 + margin

    # Guarantee a non-degenerate box (e.g. all keypoints on a straight line).
    if x2 - x1 < min_size:
        cx = (x1 + x2) / 2.0
        x1, x2 = cx - min_size / 2.0, cx + min_size / 2.0
    if y2 - y1 < min_size:
        cy = (y1 + y2) / 2.0
        y1, y2 = cy - min_size / 2.0, cy + min_size / 2.0

    if img_w is not None:
        x1, x2 = max(0.0, x1), min(float(img_w - 1), x2)
    if img_h is not None:
        y1, y2 = max(0.0, y1), min(float(img_h - 1), y2)

    return rect_to_poly_xyxy(x1, y1, x2, y2)


def parse_yolo_label_line(
    line: str,
    task: str,
    img_w: int,
    img_h: int,
    num_keypoints: int = 0,
) -> Optional["OBBOX"]:
    """Rebuild an annotation from one YOLO label line (inverse of the export).

    This is what makes an exported dataset reloadable: the label files are the
    single source of truth, so nothing can drift between what is on disk and
    what the GUI shows. Coordinates are denormalized with ``img_w`` / ``img_h``,
    which must be the dimensions of the image the label was written against.

    * ``detect`` → ``cls cx cy w h``                  → axis-aligned 4-pt poly
    * ``obb``    → ``cls x1 y1 ... x4 y4``            → exact 4-pt poly
    * ``pose``   → ``cls cx cy w h x1 y1 ... xN yN``  → poly + (N, 2) keypoints

    Returns ``None`` on a malformed or truncated line rather than raising, so a
    single bad row cannot stop a whole project from loading. Reloaded
    annotations are marked ``verified`` with ``conf = 1.0``: they only ever
    reach a label file after a human accepted them.
    """
    task = normalize_task(task)
    parts = line.split()
    if len(parts) < 5:
        return None
    try:
        cls_id = int(float(parts[0]))
        vals = [float(v) for v in parts[1:]]
    except ValueError:
        return None

    if task == TASK_OBB:
        if len(vals) < 8:
            return None
        poly = np.asarray(vals[:8], dtype=np.float32).reshape(4, 2)
        poly[:, 0] *= img_w
        poly[:, 1] *= img_h
        return OBBOX(poly=poly, cls_id=cls_id, conf=1.0, verified=True, origin=ORIGIN_DATASET)

    # detect and pose share the leading "cx cy w h"
    cx, cy, bw, bh = vals[:4]
    poly = rect_to_poly_xyxy(
        (cx - bw / 2.0) * img_w, (cy - bh / 2.0) * img_h,
        (cx + bw / 2.0) * img_w, (cy + bh / 2.0) * img_h,
    )

    keypoints = None
    if task == TASK_POSE:
        flat = vals[4:]
        n = num_keypoints or (len(flat) // KPT_DIMS)
        if n <= 0 or len(flat) < n * KPT_DIMS:
            return None
        keypoints = np.asarray(
            flat[:n * KPT_DIMS], dtype=np.float32
        ).reshape(n, KPT_DIMS)
        keypoints[:, 0] *= img_w
        keypoints[:, 1] *= img_h

    return OBBOX(poly=poly, cls_id=cls_id, conf=1.0,
                 verified=True, keypoints=keypoints, origin=ORIGIN_DATASET)


def parse_pose_result(res) -> List["OBBOX"]:
    """Build annotations from an ultralytics pose result: box + (K, 2) points.

    Shared by the annotation worker and the tracking worker on purpose. Both
    have to apply the same "all keypoints at the origin means none" guard, and
    a duplicated copy of that rule would eventually drift — leaving one of the
    two pipelines silently recording (0, 0) landmarks.

    ``res.keypoints`` may be absent when the loaded model is not a pose model,
    in which case the boxes are still returned, without keypoints, rather than
    dropping the detections entirely. Row *i* of ``keypoints`` corresponds to
    row *i* of ``boxes``, which is what makes the zip below valid.
    """
    out: List["OBBOX"] = []
    if res.boxes is None or len(res.boxes) == 0:
        return out

    xyxy = res.boxes.xyxy.cpu().numpy()
    cls_ids = res.boxes.cls.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()

    kpts = None
    kp_obj = getattr(res, "keypoints", None)
    if kp_obj is not None and getattr(kp_obj, "xy", None) is not None:
        arr = kp_obj.xy
        kpts = arr.cpu().numpy() if hasattr(arr, "cpu") else np.asarray(arr)

    for i, ((x1, y1, x2, y2), c, sc) in enumerate(zip(xyxy, cls_ids, confs)):
        kp = None
        if kpts is not None and i < len(kpts):
            candidate = np.asarray(kpts[i], dtype=np.float32).reshape(-1, 2)
            # A model that found no keypoints reports them all at the origin;
            # keeping those would poison annotations and exports alike.
            if len(candidate) and not np.allclose(candidate, 0.0):
                kp = candidate
        out.append(OBBOX(
            poly=rect_to_poly_xyxy(x1, y1, x2, y2),
            cls_id=int(c), conf=float(sc), keypoints=kp,
        ))
    return out


def keypoints_to_named_dict(
    keypoints: Optional[np.ndarray],
    names: Optional[List[str]] = None,
    ndigits: int = 2,
) -> Optional[Dict[str, List[float]]]:
    """``{name: [x, y]}`` for export, or None when there are no keypoints.

    Returning None rather than an empty dict is deliberate: a frame where the
    tracker coasted has no landmarks at all, and ``null`` says that, whereas
    ``{}`` reads like "a pose with zero points".
    """
    if keypoints is None or len(keypoints) == 0:
        return None
    pts = np.asarray(keypoints, dtype=np.float32).reshape(-1, 2)
    labels = list(names or [])
    if len(labels) < len(pts):
        labels += [f"kpt_{i}" for i in range(len(labels), len(pts))]
    return {
        labels[i]: [round(float(x), ndigits), round(float(y), ndigits)]
        for i, (x, y) in enumerate(pts)
    }


def find_orthogonal_projection(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
) -> np.ndarray:
    """Given a line segment p1→p2 and a point p3, find the two points that
    complete an oriented rectangle: project p1 and p2 onto the line parallel
    to p1→p2 passing through p3.

    Returns: np.ndarray shape (2, 2) — the two projected corners [proj_p2, proj_p1].
    """
    d = np.asarray(p2, dtype=np.float64) - np.asarray(p1, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    p3 = np.asarray(p3, dtype=np.float64)

    # Vector from p1 to p3
    v = p3 - p1

    # Component of v orthogonal to d (the shift from the original line)
    d_norm_sq = np.dot(d, d)
    if d_norm_sq < 1e-12:
        # p1 and p2 are the same point; degenerate case
        return np.array([p3, p3], dtype=np.float32)

    # Orthogonal offset = v - proj_d(v)
    ortho = v - (np.dot(v, d) / d_norm_sq) * d

    # The two new corners are p1 and p2 shifted by the orthogonal offset
    proj_p1 = p1 + ortho
    proj_p2 = p1 + d + ortho  # = p2 + ortho

    return np.array([proj_p2, proj_p1], dtype=np.float32)


# ---------------------------------------------------------------------------
# Duplicate suppression
# ---------------------------------------------------------------------------


#: Two pose detections whose skeletons sit closer than this fraction of the
#: animal's diagonal are the same animal seen twice, whatever their boxes say.
#: This is an *extra* suppression trigger, never a veto: it catches the
#: duplicate that IoU misses (two boxes of very different sizes on one shark)
#: without ever endangering a genuine neighbour, whose skeleton is nowhere near.
KPT_SAME_RATIO = 0.10


def _aabb(poly: np.ndarray) -> np.ndarray:
    pts = poly.reshape(-1, 2)
    return np.array([pts[:, 0].min(), pts[:, 1].min(),
                     pts[:, 0].max(), pts[:, 1].max()], dtype=np.float32)


def _pair_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """IoU of two AABBs ``[x1, y1, x2, y2]``."""
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
    if inter <= 0.0:
        return 0.0
    area_a = max(float(box_a[2] - box_a[0]) * float(box_a[3] - box_a[1]), 1e-6)
    area_b = max(float(box_b[2] - box_b[0]) * float(box_b[3] - box_b[1]), 1e-6)
    return float(inter / max(area_a + area_b - inter, 1e-6))


def _skeletons_coincide(a: OBBOX, b: OBBOX) -> bool:
    """Whether two pose detections describe the same animal.

    The mean landmark distance is normalised by the **larger** of the two box
    diagonals, which makes the test symmetric: normalising by the candidate's
    own box would make the answer depend on which detection is compared to
    which, so a part-detection nested in a whole-animal one would be judged
    differently depending on iteration order.
    """
    if not (a.has_keypoints() and b.has_keypoints()):
        return False
    pa = np.asarray(a.keypoints, dtype=np.float32).reshape(-1, 2)
    pb = np.asarray(b.keypoints, dtype=np.float32).reshape(-1, 2)
    if len(pa) != len(pb) or len(pa) == 0:
        return False

    box_a, box_b = _aabb(a.poly), _aabb(b.poly)
    diag = max(float(np.hypot(box_a[2] - box_a[0], box_a[3] - box_a[1])),
               float(np.hypot(box_b[2] - box_b[0], box_b[3] - box_b[1])))
    if diag < 1e-6:
        return False
    mean_dist = float(np.linalg.norm(pa - pb, axis=1).mean())
    return mean_dist / diag <= KPT_SAME_RATIO


def suppress_duplicate_detections(
    boxes: List[OBBOX], threshold: float,
) -> Tuple[List[OBBOX], int]:
    """Greedy NMS over detections, run *before* the tracker sees them.

    Ultralytics already NMS-es internally, but its default ``iou=0.7`` is
    permissive for an elongated animal: two boxes offset along a shark's axis
    can both survive, each then claims its own track ID, and the same animal
    ends up with several IDs, several skeletons and several trails. Nothing
    downstream can undo that, because by then the identities are genuinely
    distinct — hence suppressing here, where the tracker and the frame cache
    are handed the same list, so display, tracking and export are all fixed at
    once.

    A detection is dropped when it either overlaps a kept one above
    ``threshold`` (plain IoU, the usual meaning of an NMS threshold) or shares
    its skeleton with it (see :data:`KPT_SAME_RATIO`).

    Containment — intersection over the *smaller* area — is deliberately NOT
    used. It would catch a head-only detection nested in a whole-animal one,
    but it also scores 1.0 for a small shark passing under a large one, and
    deleting a real animal from a school is a far worse failure than leaving a
    spurious box that a higher ``conf`` would remove anyway.

    ``threshold <= 0`` disables the pass. Returns ``(kept, n_removed)``.
    """
    if threshold <= 0.0 or len(boxes) < 2:
        return boxes, 0

    # Highest confidence first, so a duplicate never displaces the better
    # detection of the same animal.
    order = sorted(range(len(boxes)),
                   key=lambda i: float(boxes[i].conf), reverse=True)
    aabbs = {i: _aabb(boxes[i].poly) for i in order}

    kept: List[int] = []
    for i in order:
        if any(_pair_iou(aabbs[i], aabbs[j]) >= threshold
               or _skeletons_coincide(boxes[i], boxes[j])
               for j in kept):
            continue
        kept.append(i)

    kept.sort()          # restore detection order for stable downstream indexing
    return [boxes[i] for i in kept], len(boxes) - len(kept)


# ---------------------------------------------------------------------------
# Two-stage detection (region proposal, then native-resolution refinement)
# ---------------------------------------------------------------------------

#: Padding added around every proposed box before grouping, in pixels. A box
#: from the low-resolution pass clips thin extremities — a shark's tail is the
#: first thing to fall outside it — and the second pass must see them.
REGION_PADDING = 12

#: A second-pass detection whose box comes closer than this to a region border
#: is dropped, unless that border is the image border. It is an animal cut by
#: the crop: its box is truncated and its keypoints are guesswork, yet its
#: confidence can beat the correct detection from a neighbouring region.
REGION_EDGE_MARGIN = 4

#: IoU above which a first-pass detection counts as "found again" by the
#: second pass, and is therefore dropped in favour of the refined version.
REGION_COVERED_IOU = 0.35


def translate_annotation(box: "PolyClass", dx: float, dy: float) -> "PolyClass":
    """Copy of ``box`` moved by ``(dx, dy)``, polygon **and** keypoints.

    Canonical implementation, so crop space → frame space is written once. Every
    hand-rolled version of this shift so far has moved the polygon and left the
    keypoints behind, which parks every skeleton at the top-left of the frame.
    """
    out = copy.copy(box)
    off = np.array([dx, dy], dtype=np.float32)
    out.poly = box.poly.reshape(-1, 2).astype(np.float32) + off
    if box.has_keypoints():
        out.keypoints = box.keypoints.reshape(-1, 2).astype(np.float32) + off
    return out


def merge_overlapping_boxes(boxes: Sequence[np.ndarray]) -> List[np.ndarray]:
    """Replace every group of touching AABBs by its union, transitively.

    Mirrors TRex's ``merge_boxes(..., iou_threshold=0.0)``: any two boxes that
    overlap at all become one region. Splitting an animal across two crops
    would give two truncated detections instead of one good one, and running
    two nearly identical crops wastes a forward pass for nothing.
    """
    remaining = [np.asarray(b, dtype=np.float32).copy() for b in boxes]
    merged: List[np.ndarray] = []
    while remaining:
        current = remaining.pop(0)
        changed = True
        while changed:
            changed = False
            rest = []
            for other in remaining:
                overlap = (current[0] < other[2] and other[0] < current[2]
                           and current[1] < other[3] and other[1] < current[3])
                if overlap:
                    current = np.array([
                        min(current[0], other[0]), min(current[1], other[1]),
                        max(current[2], other[2]), max(current[3], other[3]),
                    ], dtype=np.float32)
                    changed = True
                else:
                    rest.append(other)
            remaining = rest
        merged.append(current)
    return merged


def region_windows(
    boxes: Sequence[np.ndarray],
    img_w: int,
    img_h: int,
    imgsz: int,
    padding: int = REGION_PADDING,
    max_regions: int = 8,
) -> List[Tuple[int, int, int, int]]:
    """Square crop windows for the second pass, as ``(x0, y0, w, h)``.

    Each window is **at least ``imgsz`` wide**, so the model sees native pixels
    rather than an interpolated blow-up. That is the whole point: a shark 140 px
    long in a 3840-wide frame arrives at the network 37 px long once the frame
    is letterboxed to 1024, and its tail is under 3 px. In a 1024 window the
    same shark arrives 140 px long — and, crucially, at a scale the model has
    actually been trained on, since the multi-scale export never upscales
    either. Cropping tighter and letting the resize enlarge the animal would
    look sharper while putting it outside the training distribution.

    Boxes are grouped so that as many animals as possible share one window: two
    sharks 200 px apart would otherwise get two windows overlapping by 80%, for
    two forward passes and one duplicate detection. A group only grows while it
    still fits in ``imgsz``, so grouping never costs resolution.
    """
    if imgsz <= 0 or not len(boxes):
        return []

    padded = []
    for box in boxes:
        b = np.asarray(box, dtype=np.float32).reshape(4)
        padded.append(np.array([
            max(0.0, b[0] - padding), max(0.0, b[1] - padding),
            min(float(img_w), b[2] + padding),
            min(float(img_h), b[3] + padding),
        ], dtype=np.float32))

    regions = merge_overlapping_boxes(padded)
    # Biggest first: a group that already exceeds imgsz cannot absorb anything
    # without costing scale, so it should not steal small neighbours either.
    regions.sort(key=lambda r: -(max(r[2] - r[0], r[3] - r[1])))

    side_max = float(min(img_w, img_h))
    groups: List[np.ndarray] = []
    for region in regions:
        for i, group in enumerate(groups):
            union = np.array([
                min(group[0], region[0]), min(group[1], region[1]),
                max(group[2], region[2]), max(group[3], region[3]),
            ], dtype=np.float32)
            if max(union[2] - union[0], union[3] - union[1]) <= imgsz:
                groups[i] = union
                break
        else:
            groups.append(region)

    if max_regions > 0 and len(groups) > max_regions:
        # Keep the largest groups: they hold the most animals, and a tiny
        # isolated one loses the least by staying at full-frame resolution.
        groups.sort(key=lambda g: -((g[2] - g[0]) * (g[3] - g[1])))
        groups = groups[:max_regions]

    windows: List[Tuple[int, int, int, int]] = []
    for group in groups:
        needed = max(group[2] - group[0], group[3] - group[1])
        side = int(round(min(max(float(imgsz), needed), side_max)))
        cx = (group[0] + group[2]) / 2.0
        cy = (group[1] + group[3]) / 2.0
        x0 = int(round(max(0.0, min(cx - side / 2.0, float(img_w - side)))))
        y0 = int(round(max(0.0, min(cy - side / 2.0, float(img_h - side)))))
        windows.append((x0, y0, side, side))

    # Two groups can still resolve to the same window once expanded to imgsz.
    return sorted(set(windows))


def _touches_region_edge(
    box: np.ndarray, window: Tuple[int, int, int, int],
    img_w: int, img_h: int, margin: int = REGION_EDGE_MARGIN,
) -> bool:
    """Whether a detection is clipped by a crop border that is not the image's."""
    x0, y0, cw, ch = window
    x1, y1, x2, y2 = (float(v) for v in box)
    if x1 <= x0 + margin and x0 > 0:
        return True
    if y1 <= y0 + margin and y0 > 0:
        return True
    if x2 >= x0 + cw - margin and x0 + cw < img_w:
        return True
    if y2 >= y0 + ch - margin and y0 + ch < img_h:
        return True
    return False


def two_stage_detect(
    frame_bgr: np.ndarray,
    predict_frame,
    predict_region,
    imgsz: int,
    nms_threshold: float = 0.6,
    padding: int = REGION_PADDING,
    max_regions: int = 8,
    keep_conf: float = 0.0,
    translate=None,
) -> Tuple[List["OBBOX"], dict]:
    """Detect twice: propose regions on the frame, then refine at native scale.

    Two separate predictors, each ``(image) -> List[OBBOX]``, are injected: this
    module stays free of any ultralytics import, both workers share the exact
    same procedure, and — the reason they are two rather than one — the passes do
    not run with the same settings. ``predict_frame`` should be deliberately
    permissive (confidence around 0.1, as TRex does): it only has to notice that
    something is there, and its boxes are proposals rather than results.
    ``predict_region`` uses the real confidence threshold.

    ``translate(box, dx, dy)`` moves an annotation from crop space to frame
    space; it must move keypoints as well as the polygon.

    Second-pass detections replace the proposals, except where the second pass
    found nothing: a proposal no refined detection covers is kept rather than
    silently dropped, so the two-stage path can only ever add detections.

    ``keep_conf`` is the confidence those surviving proposals must reach — pass
    the *user's* threshold, not ``region_conf``. Without it, a 0.10-confidence
    proposal would be handed on as a real detection: hidden by the display
    threshold, but still exported and still fed to the tracker, where it spawns
    a phantom track. Filtering here makes the two-stage output a strict superset
    of what a single pass at the same threshold would have produced.

    ``translate`` defaults to :func:`translate_annotation`.

    Returns ``(annotations, stats)``.
    """
    if translate is None:
        translate = translate_annotation
    stats = {"regions": 0, "pass1": 0, "pass2": 0, "kept_pass1": 0,
             "below_conf": 0, "edge_dropped": 0, "suppressed": 0}
    if frame_bgr is None or frame_bgr.size == 0:
        return [], stats

    img_h, img_w = frame_bgr.shape[:2]

    first = predict_frame(frame_bgr) or []
    stats["pass1"] = len(first)
    if not first:
        return [], stats

    windows = region_windows(
        [_aabb(b.poly) for b in first], img_w, img_h, imgsz,
        padding=padding, max_regions=max_regions,
    )
    stats["regions"] = len(windows)
    if not windows:
        survivors = [p for p in first if float(p.conf) >= keep_conf]
        stats["below_conf"] = len(first) - len(survivors)
        stats["kept_pass1"] = len(survivors)
        return survivors, stats

    refined: List["OBBOX"] = []
    for (x0, y0, cw, ch) in windows:
        crop = np.ascontiguousarray(frame_bgr[y0:y0 + ch, x0:x0 + cw])
        if crop.size == 0:
            continue
        for box in predict_region(crop) or []:
            moved = translate(box, x0, y0)
            if _touches_region_edge(_aabb(moved.poly), (x0, y0, cw, ch),
                                    img_w, img_h):
                stats["edge_dropped"] += 1
                continue
            refined.append(moved)
    stats["pass2"] = len(refined)

    # Proposals the refinement missed entirely are kept, so enabling the
    # two-stage path can never lose an animal the single pass would have found.
    survivors = [p for p in first if float(p.conf) >= keep_conf]
    stats["below_conf"] = len(first) - len(survivors)
    leftovers = []
    if refined:
        refined_aabbs = [_aabb(r.poly) for r in refined]
        for proposal in survivors:
            pa = _aabb(proposal.poly)
            if not any(_pair_iou(pa, ra) >= REGION_COVERED_IOU
                       for ra in refined_aabbs):
                leftovers.append(proposal)
    else:
        leftovers = list(survivors)
    stats["kept_pass1"] = len(leftovers)

    combined, removed = suppress_duplicate_detections(
        refined + leftovers, nms_threshold
    )
    stats["suppressed"] = removed
    return combined, stats


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_annotations(
    img_bgr: np.ndarray,
    annots: List[PolyClass],
    conf_threshold: float,
    class_names: Dict[int, str] | List[str] | None,
    selected_idx: Optional[int] = None,
    show_label: bool = False,
    show_conf: bool = False,
    show_kpt_index: bool = True,
) -> np.ndarray:
    """Draw verified / unverified / selected annotations on an image copy.

    Pose instances additionally get their keypoints drawn, joined in index
    order, with keypoint 0 highlighted so the ordering stays readable.
    """
    out = img_bgr.copy()
    for i, b in enumerate(annots):
        if b.deleted or b.conf < conf_threshold:
            continue

        pts = b.poly.reshape(-1, 2).astype(int)

        # Color scheme: green=verified, orange=unverified, magenta=selected
        if selected_idx is not None and i == selected_idx:
            color = (255, 0, 255)          # magenta highlight
            thick = 4
        elif b.verified:
            color = (0, 255, 0)            # green
            thick = 4
        else:
            color = (0, 200, 255)          # orange
            thick = 4

        # Draw polygon outline
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=thick)

        # Keypoints (pose task only)
        if b.has_keypoints():
            draw_keypoints(out, b.keypoints, color, show_index=show_kpt_index)

        # Label only for unverified annotations (when requested)
        if not b.verified and (show_label or show_conf):
            parts = []
            if show_label:
                name = class_names[int(b.cls_id)] if class_names is not None else str(int(b.cls_id))
                parts.append(name)
            if show_conf:
                parts.append(f"{b.conf:.2f}")
            label = " ".join(parts)

            if label:
                (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                x0, y0 = int(pts[0, 0]), int(pts[0, 1])
                cv2.rectangle(out, (x0, y0), (x0 + tw + 6, y0 + th + base + 6), color, -1)
                cv2.putText(out, label, (x0 + 3, y0 + th + 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def draw_keypoints(
    img_bgr: np.ndarray,
    keypoints: np.ndarray,
    color: tuple = (0, 255, 255),
    radius: int = 5,
    show_index: bool = True,
) -> np.ndarray:
    """Draw keypoints in place, connected in index order.

    Keypoint 0 is drawn larger and filled white so the start of the chain (and
    therefore the left/right convention) is unambiguous while annotating.
    """
    pts = np.asarray(keypoints, dtype=np.float32).reshape(-1, 2)
    if len(pts) == 0:
        return img_bgr

    int_pts = pts.astype(int)

    # Skeleton: simple chain following the keypoint order.
    for i in range(1, len(int_pts)):
        cv2.line(img_bgr, tuple(int_pts[i - 1]), tuple(int_pts[i]),
                 color, 2, cv2.LINE_AA)

    for i, (x, y) in enumerate(int_pts):
        if i == 0:
            cv2.circle(img_bgr, (x, y), radius + 2, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img_bgr, (x, y), radius + 2, color, 2, cv2.LINE_AA)
        else:
            cv2.circle(img_bgr, (x, y), radius, color, -1, cv2.LINE_AA)
            cv2.circle(img_bgr, (x, y), radius, (20, 20, 20), 1, cv2.LINE_AA)

        if show_index:
            cv2.putText(img_bgr, str(i), (x + radius + 2, y - radius),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
                        cv2.LINE_AA)
    return img_bgr








# ---------------------------------------------------------------------------
# Frame sources
# ---------------------------------------------------------------------------

class FrameSource:
    def count(self) -> int: ...
    def read(self, idx: int) -> Optional[np.ndarray]: ...
    def fps(self) -> float: return 25.0
    def close(self): pass
    def name(self) -> str: return ""

    def stem(self) -> str:
        """Source identifier used as a PREFIX in export file names."""
        return Path(self.name()).stem

    def frame_key(self, idx: int) -> str:
        """Per-frame identifier used as a SUFFIX in export file names."""
        return f"frame{idx:06d}"

    def index_for_key(self, key: str) -> Optional[int]:
        """Inverse of frame_key(), or None if key is not one of ours."""
        m = re.fullmatch(r"frame(\d{6})", key)
        return int(m.group(1)) if m else None
    

    def frame_size(self, idx: int = 0) -> Optional[tuple]:
        """``(width, height)`` of a frame, without decoding it if possible.

        Needed to denormalize YOLO labels when reloading an exported dataset.
        Subclasses answer from metadata; the fallback decodes one frame.
        """
        img = self.read(idx)
        return (int(img.shape[1]), int(img.shape[0])) if img is not None else None


class VideoSource(FrameSource):
    """Random-access video reader that stays fast when read sequentially.

    Seeking with ``CAP_PROP_POS_FRAMES`` forces the decoder to restart from
    the previous keyframe, so on inter-coded video (H.264/H.265) asking for
    frame *n+1* that way costs a whole GOP instead of a single frame. The
    reader therefore tracks its own position and only seeks when it really
    has to:

    * next frame           → plain ``read()``            (cheapest)
    * small forward jump   → ``grab()`` the gap, decode only the last one
    * backward or far jump → real seek
    """

    #: Above this forward gap, seeking beats decoding frame by frame.
    MAX_GRAB_SKIP = 24

    def __init__(self, path: str):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        self._count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 25.0)
        # Index the capture will return on the next plain read().
        self._pos = 0

    def count(self) -> int: return self._count

    def read(self, idx: int) -> Optional[np.ndarray]:
        idx = max(0, min(idx, self._count - 1))

        if idx != self._pos:
            gap = idx - self._pos
            if 0 < gap <= self.MAX_GRAB_SKIP:
                # grab() decodes without building a numpy array: far cheaper
                # than a seek, and cheaper than a full read per skipped frame.
                for _ in range(gap):
                    if not self.cap.grab():
                        break
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            self._pos = idx

        ok, frame = self.cap.read()
        if not ok:
            # Force a real seek next time: the position is now unknown.
            self._pos = -1
            return None
        self._pos = idx + 1
        return frame

    def fps(self) -> float: return self._fps
    def close(self):
        if self.cap: self.cap.release()
    def name(self) -> str: return os.path.basename(self.path)

    def frame_size(self, idx: int = 0) -> Optional[tuple]:
        """Frame size straight from the container: no decode, no seek.

        Reading it from CAP_PROP also leaves ``self._pos`` untouched, so the
        sequential-read fast path is not invalidated.
        """
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if w > 0 and h > 0:
            return (w, h)
        return super().frame_size(idx)


class ImageFolderSource(FrameSource):
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    _SAFE = re.compile(r"[^A-Za-z0-9._-]+")

    def __init__(self, folder: str):
        self.path = os.path.abspath(folder)          # see C.1
        files = [f for f in os.listdir(self.path)
                 if os.path.splitext(f)[1].lower() in self.IMAGE_EXTS]
        if not files:
            raise RuntimeError("No images found in folder.")

        def _key(s):
            return [int(t) if t.isdigit() else t.lower()
                    for t in re.findall(r'\d+|\D+', s)]
        files.sort(key=_key)
        self.paths = [os.path.join(self.path, f) for f in files]

        # Export key = image FILE NAME, not position. Inserting or deleting
        # one image shifts every index after it; keys built from the position
        # would then re-attach every label to the wrong image, silently.
        keys, seen = [], set()
        for p in self.paths:
            k = self._SAFE.sub("_", Path(p).stem)
            if k in seen:
                # photo.jpg and photo.png share a stem: fold the extension in
                # or the second one overwrites the first one's label.
                k = self._SAFE.sub("_", os.path.basename(p))
            seen.add(k)
            keys.append(k)
        self._keys = keys
        self._key_index = {k: i for i, k in enumerate(keys)}

    def stem(self) -> str:
        # A folder has no extension: Path().stem would cut "prise_2024.01.15"
        # down to "prise_2024.01", and two dated folders would collide.
        return self._SAFE.sub("_", self.name()) or "images"
    
    def count(self) -> int: return len(self.paths)

    def frame_key(self, idx: int) -> str:
        return self._keys[max(0, min(idx, len(self._keys) - 1))]

    def index_for_key(self, key: str) -> Optional[int]:
        idx = self._key_index.get(key)
        if idx is not None:
            return idx
        # Fallback for datasets exported back when folders were named by
        # position, so an existing project is not orphaned.
        return super().index_for_key(key)
    
    def read(self, idx: int) -> Optional[np.ndarray]:
        idx = max(0, min(idx, len(self.paths) - 1))
        img = cv2.imread(self.paths[idx], cv2.IMREAD_UNCHANGED)
        return ensure_bgr_u8(img) if img is not None else None

    def fps(self) -> float: return 10.0
    def name(self) -> str: return os.path.basename(self.path)

    def path_at(self, idx: int) -> str:
        idx = max(0, min(idx, len(self.paths) - 1))
        return self.paths[idx]

    def frame_size(self, idx: int = 0) -> Optional[tuple]:
        """Size from the image header only — no pixel decoding.

        Unlike a video, a folder may mix resolutions, so this is queried per
        index rather than once for the whole source.
        """
        idx = max(0, min(idx, len(self.paths) - 1))
        reader = QtGui.QImageReader(self.paths[idx])
        size = reader.size()
        if size.isValid() and size.width() > 0 and size.height() > 0:
            return (int(size.width()), int(size.height()))
        return super().frame_size(idx)