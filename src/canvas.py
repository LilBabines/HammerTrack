"""
AnnotationCanvas — a ``QLabel`` subclass that renders a BGR frame with
zoom & pan, maps display↔image coordinates, and forwards user
interactions as Qt signals.

The canvas is *dumb* about annotation modes: it doesn't know about
OBB / BBox / crop-inference. It just displays whatever pre-rendered
BGR image the parent gives it, and tells the parent where the user
clicked, in image-space coordinates. The parent (AnnotatePage) is
in charge of mode-specific behaviour.

Built-in interactions (handled internally, parent doesn't see them):
* Mouse wheel  → zoom in/out around cursor
* Space + LMB drag → pan
"""

from typing import Optional

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .utils import cvimg_to_qimage


class AnnotationCanvas(QtWidgets.QLabel):
    """Image display widget with zoom, pan and image-space mouse signals.

    Signals
    -------
    mouse_pressed(event, x_img, y_img)
    mouse_moved(event, x_img, y_img)
    mouse_released(event, x_img, y_img)
        Forwarded mouse events with image-space coordinates. The original
        ``QMouseEvent`` is passed through so the parent can inspect button,
        modifiers, etc.
    """

    mouse_pressed = QtCore.Signal(object, float, float)
    mouse_moved = QtCore.Signal(object, float, float)
    mouse_released = QtCore.Signal(object, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        # Display config
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background:#111; border:1px solid #333;")
        self.setMinimumSize(720, 405)

        # Zoom & pan state
        self.zoom = 1.0
        self.min_zoom = 0.25
        self.max_zoom = 8.0
        self.pan_img = np.array([0.0, 0.0], dtype=np.float32)

        # Internal pan-with-space-held state
        self._space_held = False
        self._pan_dragging = False
        self._pan_last_disp = (0, 0)

        # Frame state
        self._frame_bgr: Optional[np.ndarray] = None
        # Last computed display↔image mapping
        self._draw_map: dict = {
            "scale": 1.0, "xoff": 0, "yoff": 0,
            "img_w": 0, "img_h": 0,
            "panx": 0.0, "pany": 0.0,
            "base": 1.0, "lbl_w": 1, "lbl_h": 1,
        }

    # ==================== Frame management ====================

    def set_frame(self, frame_bgr: Optional[np.ndarray]):
        """Set the BGR image to display (already pre-rendered with overlays)."""
        self._frame_bgr = frame_bgr
        self.refresh()

    def has_frame(self) -> bool:
        return self._frame_bgr is not None

    def refresh(self):
        """Redraw the cached frame using the current zoom & pan."""
        if self._frame_bgr is None:
            self.clear()
            return

        qimg = cvimg_to_qimage(self._frame_bgr)
        img_w, img_h = qimg.width(), qimg.height()
        lbl_w, lbl_h = self.width(), self.height()
        base = min(lbl_w / img_w, lbl_h / img_h) if img_w and img_h else 1.0
        scale = base * float(self.zoom)
        disp_w, disp_h = int(img_w * scale), int(img_h * scale)
        xoff = (lbl_w - disp_w) // 2
        yoff = (lbl_h - disp_h) // 2

        canvas = QtGui.QPixmap(lbl_w, lbl_h)
        canvas.fill(QtGui.QColor(17, 17, 17))
        painter = QtGui.QPainter(canvas)
        scaled = QtGui.QPixmap.fromImage(qimg).scaled(
            disp_w, disp_h,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        draw_x = int(xoff - self.pan_img[0] * scale)
        draw_y = int(yoff - self.pan_img[1] * scale)
        painter.drawPixmap(draw_x, draw_y, scaled)
        painter.end()

        self._draw_map = {
            "scale": scale, "xoff": xoff, "yoff": yoff,
            "img_w": img_w, "img_h": img_h,
            "panx": float(self.pan_img[0]), "pany": float(self.pan_img[1]),
            "base": base, "lbl_w": lbl_w, "lbl_h": lbl_h,
        }
        self.setPixmap(canvas)

    # ==================== Coordinate mapping ====================

    def display_to_image(self, x_disp: int, y_disp: int):
        """Convert display coords → image coords.

        Returns ``(None, None)`` if the point is outside the image.
        """
        m = self._draw_map
        s = m["scale"]
        if s <= 0:
            return None, None
        xi = (x_disp - (m["xoff"] - m["panx"] * s)) / s
        yi = (y_disp - (m["yoff"] - m["pany"] * s)) / s
        if xi < 0 or yi < 0 or xi >= m["img_w"] or yi >= m["img_h"]:
            return None, None
        return float(xi), float(yi)

    # ==================== Zoom ====================

    def zoom_fit(self):
        """Reset zoom and pan to the default fit-in-view state."""
        self.zoom = 1.0
        self.pan_img[:] = 0.0
        self.refresh()

    def zoom_step(self, direction: int,
                  anchor_disp: Optional[QtCore.QPointF] = None):
        """Zoom in (+1) or out (-1) around an anchor in display space.

        ``anchor_disp`` defaults to the canvas centre.
        """
        if self._frame_bgr is None:
            return
        if anchor_disp is None:
            anchor_disp = QtCore.QPointF(self.width() / 2.0,
                                         self.height() / 2.0)

        step = 1.25 if direction > 0 else 0.8
        new_zoom = float(np.clip(self.zoom * step, self.min_zoom, self.max_zoom))
        if abs(new_zoom - self.zoom) < 1e-6:
            return

        m = self._draw_map
        xd, yd = float(anchor_disp.x()), float(anchor_disp.y())
        xi, yi = self.display_to_image(int(xd), int(yd))
        if xi is None:
            self.zoom = new_zoom
            self.refresh()
            return

        self.zoom = new_zoom
        base = float(m["base"])
        new_scale = base * self.zoom
        xoff = (m["lbl_w"] - m["img_w"] * new_scale) / 2.0
        yoff = (m["lbl_h"] - m["img_h"] * new_scale) / 2.0
        self.pan_img[0] = (xoff + xi * new_scale - xd) / new_scale
        self.pan_img[1] = (yoff + yi * new_scale - yd) / new_scale
        self._clamp_pan()
        self.refresh()

    def _clamp_pan(self):
        m = self._draw_map
        if "base" not in m:
            return
        s = m["base"] * self.zoom
        if s <= 0:
            return
        margin = 0.1
        max_pan_x = (m["img_w"] * s - (1.0 - margin) * m["lbl_w"]) / s / 2.0
        max_pan_y = (m["img_h"] * s - (1.0 - margin) * m["lbl_h"]) / s / 2.0
        self.pan_img[0] = float(np.clip(self.pan_img[0], -max_pan_x, max_pan_x))
        self.pan_img[1] = float(np.clip(self.pan_img[1], -max_pan_y, max_pan_y))

    # ==================== Space-held pan toggle ====================

    def set_space_held(self, held: bool):
        """Tell the canvas whether the spacebar is currently held."""
        self._space_held = held

    # ==================== Native event handlers ====================

    def wheelEvent(self, event: QtGui.QWheelEvent):
        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom_step(+1, anchor_disp=event.position())
        elif delta < 0:
            self.zoom_step(-1, anchor_disp=event.position())
        event.accept()

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        # Always grab focus so keyboard shortcuts work after a click.
        self.setFocus(QtCore.Qt.FocusReason.MouseFocusReason)

        if not self.has_frame():
            return
        x_disp = int(event.position().x())
        y_disp = int(event.position().y())
        x_img, y_img = self.display_to_image(x_disp, y_disp)
        if x_img is None:
            return

        # Built-in pan with Space + LMB drag
        if (event.button() == QtCore.Qt.MouseButton.LeftButton
                and self._space_held):
            self._pan_dragging = True
            self._pan_last_disp = (x_disp, y_disp)
            event.accept()
            return

        self.mouse_pressed.emit(event, x_img, y_img)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        if not self.has_frame():
            return
        x_disp = int(event.position().x())
        y_disp = int(event.position().y())

        if self._pan_dragging:
            dx = x_disp - self._pan_last_disp[0]
            dy = y_disp - self._pan_last_disp[1]
            s = self._draw_map.get("scale", 1.0)
            if s > 0:
                self.pan_img[0] -= dx / s
                self.pan_img[1] -= dy / s
            self._pan_last_disp = (x_disp, y_disp)
            self.refresh()
            event.accept()
            return

        x_img, y_img = self.display_to_image(x_disp, y_disp)
        if x_img is None:
            return
        self.mouse_moved.emit(event, x_img, y_img)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        if self._pan_dragging:
            self._pan_dragging = False
            event.accept()
            return
        if not self.has_frame():
            return
        x_disp = int(event.position().x())
        y_disp = int(event.position().y())
        x_img, y_img = self.display_to_image(x_disp, y_disp)
        if x_img is None:
            return
        self.mouse_released.emit(event, x_img, y_img)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.refresh()
