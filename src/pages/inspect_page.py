"""
Browse dataset images + labels from the project (train / val).

On the val split, can overlay model predictions with confidence scores.
Supports both OBB (8-coord) and BBOX (cx cy w h) label formats.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ..utils import OBBOX, cvimg_to_qimage, ensure_bgr_u8
from ..workers import DetectionWorker, YOLO_MODEL_PATH, resolve_model_path

if TYPE_CHECKING:
    from ..windows import LauncherWindow


class InspectDatasetPage(QtWidgets.QWidget):
    """Browse images + labels from the project dataset."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._launcher: Optional["LauncherWindow"] = None
        self._items: List[Dict[str, str]] = []
        self._current: int = 0
        self._current_bgr: Optional[np.ndarray] = None
        self._pred_cache: Dict[int, List[OBBOX]] = {}
        self._show_preds: bool = False

        self._build_ui()

    # ==================== UI construction ====================

    def _build_ui(self):
        # ---- Controls bar ----
        self.split_combo = QtWidgets.QComboBox()
        self.split_combo.addItems(["train", "val"])
        self.split_combo.currentTextChanged.connect(self._load_split)

        self.stats_label = QtWidgets.QLabel("No dataset loaded.")

        self.prev_btn = QtWidgets.QPushButton("⟸ Prev")
        self.next_btn = QtWidgets.QPushButton("Next ⟹")
        self.prev_btn.clicked.connect(self._prev)
        self.next_btn.clicked.connect(self._next)

        self.idx_spin = QtWidgets.QSpinBox()
        self.idx_spin.setPrefix("Image #")
        self.idx_spin.setRange(0, 0)
        self.idx_spin.valueChanged.connect(self._go_to)

        self.run_pred_btn = QtWidgets.QPushButton("Run predictions (val)")
        self.run_pred_btn.setToolTip(
            "Run the detector on the current image and overlay predictions"
        )
        self.run_pred_btn.clicked.connect(self._run_prediction)

        self.show_preds_chk = QtWidgets.QCheckBox("Show predictions")
        self.show_preds_chk.setChecked(False)
        self.show_preds_chk.toggled.connect(self._toggle_preds)

        self.conf_spin = QtWidgets.QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 0.99)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.5)
        self.conf_spin.setPrefix("conf=")
        self.conf_spin.valueChanged.connect(lambda _: self._redraw())

        ctrl = QtWidgets.QHBoxLayout()
        ctrl.addWidget(QtWidgets.QLabel("Split:"))
        ctrl.addWidget(self.split_combo)
        ctrl.addSpacing(15)
        ctrl.addWidget(self.stats_label, stretch=1)
        ctrl.addSpacing(15)
        ctrl.addWidget(self.prev_btn)
        ctrl.addWidget(self.idx_spin)
        ctrl.addWidget(self.next_btn)

        pred_bar = QtWidgets.QHBoxLayout()
        pred_bar.addWidget(self.run_pred_btn)
        pred_bar.addWidget(self.show_preds_chk)
        pred_bar.addWidget(self.conf_spin)
        pred_bar.addStretch(1)

        # ---- Canvas ----
        self.canvas = QtWidgets.QLabel()
        self.canvas.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.canvas.setStyleSheet("background:#111; border:1px solid #333;")
        self.canvas.setMinimumSize(640, 400)

        self.file_label = QtWidgets.QLabel("")
        self.file_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        # ---- Layout ----
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(8)
        v.addLayout(ctrl)
        v.addLayout(pred_bar)
        v.addWidget(self.canvas, stretch=1)
        v.addWidget(self.file_label)

    def set_launcher(self, launcher: "LauncherWindow"):
        self._launcher = launcher

    # ==================== Data loading ====================

    def refresh(self):
        self._load_split(self.split_combo.currentText())

    def _dataset_dir(self) -> str:
        if self._launcher:
            cfg = self._launcher.project_config()
            return cfg.get("dataset_dir", "")
        return ""

    def _load_split(self, split: str):
        ds = Path(self._dataset_dir())
        img_dir = ds / "images" / split
        lbl_dir = ds / "labels" / split

        self._items.clear()
        self._pred_cache.clear()
        self._current = 0
        self._show_preds = False
        self.show_preds_chk.setChecked(False)

        if not img_dir.is_dir():
            self.stats_label.setText(f"Folder not found: {img_dir}")
            self._update_nav()
            self._clear_canvas()
            return

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        imgs = sorted(
            (p for p in img_dir.iterdir() if p.suffix.lower() in exts),
            key=lambda p: p.name,
        )

        total_annots = 0
        for img_path in imgs:
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            lbl_str = str(lbl_path) if lbl_path.is_file() else ""
            self._items.append({"img": str(img_path), "lbl": lbl_str})
            if lbl_str:
                try:
                    lines = lbl_path.read_text().strip().splitlines()
                    total_annots += sum(1 for ln in lines if ln.strip())
                except Exception:
                    pass

        n_imgs = len(self._items)
        self.stats_label.setText(
            f"{split}: {n_imgs} images, {total_annots} annotations"
        )
        self._update_nav()
        if n_imgs > 0:
            self._show_item(0)
        else:
            self._clear_canvas()

    # ==================== Navigation ====================

    def _update_nav(self):
        n = len(self._items)
        self.prev_btn.setEnabled(n > 0)
        self.next_btn.setEnabled(n > 0)
        self.idx_spin.blockSignals(True)
        self.idx_spin.setRange(0, max(0, n - 1))
        self.idx_spin.setValue(self._current)
        self.idx_spin.blockSignals(False)

        is_val = self.split_combo.currentText() == "val"
        self.run_pred_btn.setVisible(is_val)
        self.show_preds_chk.setVisible(is_val)
        self.conf_spin.setVisible(is_val)

    def _prev(self):
        if self._items and self._current > 0:
            self._show_item(self._current - 1)

    def _next(self):
        if self._items and self._current < len(self._items) - 1:
            self._show_item(self._current + 1)

    def _go_to(self, idx: int):
        if 0 <= idx < len(self._items):
            self._show_item(idx)

    # ==================== Display ====================

    def _clear_canvas(self):
        self.canvas.clear()
        self.file_label.setText("")
        self._current_bgr = None

    def _show_item(self, idx: int):
        if idx < 0 or idx >= len(self._items):
            return
        self._current = idx
        self.idx_spin.blockSignals(True)
        self.idx_spin.setValue(idx)
        self.idx_spin.blockSignals(False)

        item = self._items[idx]
        img = cv2.imread(item["img"], cv2.IMREAD_UNCHANGED)
        if img is None:
            self.file_label.setText(f"Failed to read: {item['img']}")
            self.canvas.clear()
            self._current_bgr = None
            return

        img = ensure_bgr_u8(img)
        self._current_bgr = img
        self.file_label.setText(
            f"{Path(item['img']).name}  "
            f"({img.shape[1]}×{img.shape[0]})  "
            f"[{idx + 1}/{len(self._items)}]"
        )
        self._redraw()

    def _redraw(self):
        if self._current_bgr is None:
            return
        item = self._items[self._current]
        img = self._current_bgr.copy()
        h, w = img.shape[:2]
        conf_thresh = self.conf_spin.value()

        # Draw GT boxes (green)
        gt_boxes = self._parse_label(item["lbl"], w, h)
        for box in gt_boxes:
            pts = box.poly.reshape(-1, 2).astype(int)
            cv2.polylines(img, [pts], True, (0, 255, 0), 2, cv2.LINE_AA)

        # Draw predictions (yellow) if enabled
        if self._show_preds and self._current in self._pred_cache:
            preds = self._pred_cache[self._current]
            for box in preds:
                if box.conf < conf_thresh:
                    continue
                pts = box.poly.reshape(-1, 2).astype(int)
                cv2.polylines(img, [pts], True, (255, 255, 0), 2, cv2.LINE_AA)
                x0, y0 = int(pts[0, 0]), int(pts[0, 1])
                label = f"{box.conf:.2f}"
                (tw, th_), base = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    img, (x0, y0 - th_ - base - 4),
                    (x0 + tw + 6, y0), (255, 255, 0), -1,
                )
                cv2.putText(
                    img, label, (x0 + 3, y0 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
                )

        self._draw_legend(img, gt_boxes, conf_thresh)
        self._display(img)

    def _draw_legend(self, img: np.ndarray, gt_boxes, conf_thresh: float):
        lines = [f"GT: {len(gt_boxes)} boxes"]
        if self._show_preds and self._current in self._pred_cache:
            preds = self._pred_cache[self._current]
            n_shown = sum(1 for b in preds if b.conf >= conf_thresh)
            lines.append(f"Pred: {n_shown}/{len(preds)} (conf>={conf_thresh:.2f})")

        y = 20
        for line in lines:
            (tw, th_), _ = cv2.getTextSize(
                line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
            )
            cv2.rectangle(img, (8, y - th_ - 4), (14 + tw, y + 4), (0, 0, 0), -1)
            cv2.putText(
                img, line, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
            )
            y += th_ + 12

    def _display(self, img_bgr: np.ndarray):
        qimg = cvimg_to_qimage(img_bgr)
        lbl_w, lbl_h = self.canvas.width(), self.canvas.height()
        pix = QtGui.QPixmap.fromImage(qimg).scaled(
            lbl_w, lbl_h,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self.canvas.setPixmap(pix)

    # ==================== Label parsing (OBB vs BBOX auto-detect) ====================

    @staticmethod
    def _parse_label(lbl_path: str, img_w: int, img_h: int) -> List[OBBOX]:
        """Parse a YOLO label file. Auto-detects format:

        * 9+ tokens → OBB:  ``cls x1 y1 x2 y2 x3 y3 x4 y4`` (normalized)
        * 5  tokens → BBOX: ``cls cx cy w h`` (normalized)
        """
        if not lbl_path or not os.path.isfile(lbl_path):
            return []

        boxes: List[OBBOX] = []
        try:
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])

                    if len(parts) >= 9:
                        # OBB: cls x1 y1 x2 y2 x3 y3 x4 y4
                        coords = [float(x) for x in parts[1:9]]
                        pts = np.array(coords, dtype=np.float32).reshape(4, 2)
                        pts[:, 0] *= img_w
                        pts[:, 1] *= img_h
                        boxes.append(OBBOX(poly=pts, cls_id=cls_id, conf=1.0))
                    else:
                        # BBOX: cls cx cy w h (normalized)
                        cx = float(parts[1]) * img_w
                        cy = float(parts[2]) * img_h
                        bw = float(parts[3]) * img_w
                        bh = float(parts[4]) * img_h
                        x1, y1 = cx - bw / 2, cy - bh / 2
                        x2, y2 = cx + bw / 2, cy + bh / 2
                        pts = np.array(
                            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                            dtype=np.float32,
                        )
                        boxes.append(OBBOX(poly=pts, cls_id=cls_id, conf=1.0))
        except Exception:
            pass
        return boxes

    # ==================== Prediction overlay (val only) ====================

    def _toggle_preds(self, checked: bool):
        self._show_preds = checked
        self._redraw()

    def _run_prediction(self):
        if not self._items or self._current_bgr is None:
            return
        if not self._launcher:
            return

        cfg = self._launcher.project_config()
        model_path = cfg.get("model_path", YOLO_MODEL_PATH)
        task = cfg.get("task_type", "auto")
        if task == "auto":
            task = "detect"
        conf = self.conf_spin.value()

        self.run_pred_btn.setEnabled(False)
        self.run_pred_btn.setText("Running...")

        item = self._items[self._current]

        self._pred_thread = QtCore.QThread(self)
        self._pred_worker = DetectionWorker(
            frame_idx=self._current,
            frame_bgr=self._current_bgr,
            conf=conf,
            imgsz=cfg.get("imgsz", 1024),
            model_path=resolve_model_path(model_path, task),
            source_path=item["img"],
        )
        self._pred_worker.moveToThread(self._pred_thread)
        self._pred_thread.started.connect(self._pred_worker.run)
        self._pred_worker.finished.connect(self._on_pred_done)
        self._pred_worker.error.connect(self._on_pred_error)
        self._pred_worker.finished.connect(self._pred_thread.quit)
        self._pred_worker.error.connect(self._pred_thread.quit)
        self._pred_thread.finished.connect(self._pred_worker.deleteLater)
        self._pred_thread.finished.connect(self._pred_thread.deleteLater)
        self._pred_thread.start()

    def _on_pred_done(self, idx: int, class_names, annots: List[OBBOX]):
        self._pred_cache[idx] = annots
        self.show_preds_chk.setChecked(True)
        self._show_preds = True
        self._redraw()
        self.run_pred_btn.setEnabled(True)
        self.run_pred_btn.setText("Run predictions (val)")

    def _on_pred_error(self, msg: str):
        self.run_pred_btn.setEnabled(True)
        self.run_pred_btn.setText("Run predictions (val)")
        if self._launcher:
            self._launcher.statusBar().showMessage(
                f"Prediction error: {msg}", 5000
            )

    # ==================== Resize ====================

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._current_bgr is not None:
            self._redraw()