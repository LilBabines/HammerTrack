from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

import cv2
import numpy as np
from PySide6 import QtGui

import warnings
import os


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
    """Annotation container for generic polygonal regions."""
    poly: np.ndarray           # shape (n, 2) float32, image coordinates
    cls_id: int
    conf: float
    verified: bool = False
    deleted: bool = False

    def to_json(self) -> dict:
        return {
            "poly": self.poly.tolist(),
            "cls_id": int(self.cls_id),
            "conf": float(self.conf),
            "verified": bool(self.verified),
            "deleted": bool(self.deleted),
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
) -> np.ndarray:
    """Draw verified / unverified / selected annotations on an image copy."""
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








# ---------------------------------------------------------------------------
# Frame sources
# ---------------------------------------------------------------------------

class FrameSource:
    def count(self) -> int: ...
    def read(self, idx: int) -> Optional[np.ndarray]: ...
    def fps(self) -> float: return 25.0
    def close(self): pass
    def name(self) -> str: return ""


class VideoSource(FrameSource):
    def __init__(self, path: str):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        self._count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 25.0)

    def count(self) -> int: return self._count

    def read(self, idx: int) -> Optional[np.ndarray]:
        idx = max(0, min(idx, self._count - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        return frame if ok else None

    def fps(self) -> float: return self._fps
    def close(self):
        if self.cap: self.cap.release()
    def name(self) -> str: return os.path.basename(self.path)


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
