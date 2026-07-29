from typing import Optional, List, Dict
from dataclasses import dataclass

import cv2
import numpy as np
from PySide6 import QtGui

import warnings
import os

from .tasks import TASK_OBB, TASK_POSE, KPT_DIMS, normalize_task


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

    def has_keypoints(self) -> bool:
        return self.keypoints is not None and len(self.keypoints) > 0

    def to_json(self) -> dict:
        return {
            "poly": self.poly.tolist(),
            "cls_id": int(self.cls_id),
            "conf": float(self.conf),
            "verified": bool(self.verified),
            "deleted": bool(self.deleted),
            "keypoints": (self.keypoints.tolist()
                          if self.has_keypoints() else None),
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
        return OBBOX(poly=poly, cls_id=cls_id, conf=1.0, verified=True)

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
                 verified=True, keypoints=keypoints)


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

    def __init__(self, folder: str):
        self.path = folder
        files = [f for f in os.listdir(self.path)
                 if os.path.splitext(f)[1].lower() in self.IMAGE_EXTS]
        if not files:
            raise RuntimeError("No images found in folder.")
        import re
        def _key(s):
            return [int(t) if t.isdigit() else t.lower()
                    for t in re.findall(r'\d+|\D+', s)]
        files.sort(key=_key)
        self.paths = [os.path.join(self.path, f) for f in files]

    def count(self) -> int: return len(self.paths)

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