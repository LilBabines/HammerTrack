"""
AnnotatedFrameSlider — a ``QSlider`` that highlights frames holding
annotations, so they can be spotted and jumped to while scrubbing.

The owner calls :meth:`set_marked_frames` whenever the annotation set changes;
the marks are painted straight into the groove. :meth:`next_marked` and
:meth:`prev_marked` back the navigation buttons and their shortcuts.
"""

from typing import Iterable, List, Optional

from PySide6 import QtCore, QtGui, QtWidgets


class AnnotatedFrameSlider(QtWidgets.QSlider):
    """Frame slider with highlighted annotated frames."""

    #: Minimum width of a mark, in pixels. Wide videos would otherwise render
    #: sub-pixel marks that are invisible.
    MIN_MARK_WIDTH = 2

    def __init__(self, orientation=QtCore.Qt.Orientation.Horizontal, parent=None):
        super().__init__(orientation, parent)
        self._marked: set[int] = set()
        self._mark_color = QtGui.QColor(46, 204, 113)
        self.setMinimumHeight(24)

    # ==================== Marked frames ====================

    def set_marked_frames(self, frames: Iterable[int]):
        """Replace the set of highlighted frames and repaint if it changed."""
        new = {int(f) for f in frames}
        if new != self._marked:
            self._marked = new
            self.update()

    def marked_frames(self) -> List[int]:
        return sorted(self._marked)

    def has_marks(self) -> bool:
        return bool(self._marked)

    def next_marked(self, current: int) -> Optional[int]:
        """First marked frame strictly after ``current``, or None."""
        later = [f for f in self._marked if f > current]
        return min(later) if later else None

    def prev_marked(self, current: int) -> Optional[int]:
        """Last marked frame strictly before ``current``, or None."""
        earlier = [f for f in self._marked if f < current]
        return max(earlier) if earlier else None

    # ==================== Painting ====================

    def _groove_geometry(self):
        """Return ``(x_start, span, groove_rect)`` used to place the marks.

        ``x_start`` is the pixel of the minimum value and ``span`` the usable
        width, both accounting for the handle so the marks line up exactly
        with where the handle sits for a given value.
        """
        opt = QtWidgets.QStyleOptionSlider()
        self.initStyleOption(opt)
        style = self.style()
        groove = style.subControlRect(
            QtWidgets.QStyle.ComplexControl.CC_Slider, opt,
            QtWidgets.QStyle.SubControl.SC_SliderGroove, self,
        )
        handle = style.subControlRect(
            QtWidgets.QStyle.ComplexControl.CC_Slider, opt,
            QtWidgets.QStyle.SubControl.SC_SliderHandle, self,
        )
        span = groove.width() - handle.width()
        x_start = groove.x() + handle.width() / 2.0
        return x_start, span, groove

    def paintEvent(self, event: QtGui.QPaintEvent):
        super().paintEvent(event)

        if not self._marked or self.maximum() <= self.minimum():
            return

        x_start, span, groove = self._groove_geometry()
        if span <= 0:
            return

        lo, hi = self.minimum(), self.maximum()
        # One frame is this many pixels; widen thin marks so they stay visible.
        px_per_frame = span / float(hi - lo)
        width = max(self.MIN_MARK_WIDTH, px_per_frame)

        height = max(6, groove.height())
        top = groove.center().y() - height / 2.0

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(self._mark_color)

        for frame in self._marked:
            if frame < lo or frame > hi:
                continue
            x = x_start + (frame - lo) * px_per_frame - width / 2.0
            painter.drawRect(QtCore.QRectF(x, top, width, height))

        painter.end()
