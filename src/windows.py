import os
import json
import shutil
import random
import threading
from typing import Optional, List, Dict, Any
from pathlib import Path

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

# ------ Local imports ------
from .utils import (
    OBBOX, PolyClass, cvimg_to_qimage, draw_annotations,
    find_orthogonal_projection, ensure_bgr_u8,
    FrameSource,VideoSource, ImageFolderSource
)
from .workers import (
    DetectionWorker, DetectFinetuneWorker, YOLO_MODEL_PATH
)

from .tracking import TrackingPage

from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Project manager — handles project folders and config persistence
# ---------------------------------------------------------------------------

PROJECTS_ROOT = os.path.join(os.getcwd(), "projects")


class ProjectManager:
    """Manages project directories and their config files."""

    def __init__(self, root: str = PROJECTS_ROOT):
        self.root = root
        os.makedirs(self.root, exist_ok=True)

    def list_projects(self) -> List[str]:
        if not os.path.isdir(self.root):
            return []
        return sorted(
            d for d in os.listdir(self.root)
            if os.path.isdir(os.path.join(self.root, d))
        )

    def create_project(self, name: str) -> str:
        proj_dir = os.path.join(self.root, name)
        os.makedirs(proj_dir, exist_ok=True)
        # Sub-folders
        for sub in ("datasets/images/train", "datasets/images/val",
                     "datasets/labels/train", "datasets/labels/val",
                     "finetune_runs", "exports"):
            os.makedirs(os.path.join(proj_dir, sub), exist_ok=True)
        # Default config
        cfg_path = os.path.join(proj_dir, "config.json")
        if not os.path.exists(cfg_path):
            self.save_config(name, self._default_config(name))
        return proj_dir

    def project_dir(self, name: str) -> str:
        return os.path.join(self.root, name)

    def load_config(self, name: str) -> dict:
        cfg_path = os.path.join(self.root, name, "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return self._default_config(name)

    def save_config(self, name: str, cfg: dict):
        cfg_path = os.path.join(self.root, name, "config.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)

    def _default_config(self, name: str) -> dict:
        proj = self.project_dir(name)
        return {
            "project_name": name,
            "dataset_dir": os.path.join(proj, "datasets"),
            "finetune_dir": os.path.join(proj, "finetune_runs"),
            "model_path": YOLO_MODEL_PATH,
            "class_names": ["object"],
            "epochs": 20,
            "imgsz": 1024,
            "batch": 16,
            "val_split": 0.1,
            "conf_threshold": 0.5,
            "tracker_type": "botsort",
            "reid_weights": "osnet_x0_25_msmt17.pt",
            "with_reid": True,
            "track_high_thresh": 0.6,
            "track_low_thresh": 0.1,
            "new_track_thresh": 0.7,
            "track_buffer": 30,
            "match_thresh": 0.8,
            "proximity_thresh": 0.5,
            "appearance_thresh": 0.25,
        }




# ---------------------------------------------------------------------------
# Signal bridge for finetuning
# ---------------------------------------------------------------------------

class _FinetuneSignalBridge(QtCore.QObject):
    progress = QtCore.Signal(str, float)
    finished = QtCore.Signal(str)
    error = QtCore.Signal(str)


# ---------------------------------------------------------------------------
# Settings page (tab content)
# ---------------------------------------------------------------------------

class SettingsPage(QtWidgets.QWidget):
    """Project settings editor."""

    config_changed = QtCore.Signal()   # emitted when user saves

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cfg: dict = {}

        form = QtWidgets.QFormLayout()
        form.setContentsMargins(20, 20, 20, 20)
        form.setSpacing(12)

        self.model_path_edit = QtWidgets.QLineEdit()
        self.model_browse_btn = QtWidgets.QPushButton("Browse...")
        self.model_browse_btn.clicked.connect(self._browse_model)
        model_row = QtWidgets.QHBoxLayout()
        model_row.addWidget(self.model_path_edit, stretch=1)
        model_row.addWidget(self.model_browse_btn)

        self.dataset_dir_edit = QtWidgets.QLineEdit()
        self.dataset_dir_browse_btn = QtWidgets.QPushButton("Browse...")
        self.dataset_dir_browse_btn.clicked.connect(self._browse_dataset)
        dataset_dir_row = QtWidgets.QHBoxLayout()
        dataset_dir_row.addWidget(self.dataset_dir_edit, stretch=1)
        dataset_dir_row.addWidget(self.dataset_dir_browse_btn)
        

        self.class_names_edit = QtWidgets.QLineEdit()
        self.class_names_edit.setToolTip("Comma-separated class names, e.g.: cat, dog, bird")

        self.epochs_spin = QtWidgets.QSpinBox()
        self.epochs_spin.setRange(1, 500)

        self.imgsz_spin = QtWidgets.QSpinBox()
        self.imgsz_spin.setRange(128, 4096)
        self.imgsz_spin.setSingleStep(64)

        self.batch_spin = QtWidgets.QSpinBox()
        self.batch_spin.setRange(1, 128)

        self.val_split_spin = QtWidgets.QDoubleSpinBox()
        self.val_split_spin.setRange(0.0, 0.5)
        self.val_split_spin.setSingleStep(0.05)

        self.conf_spin = QtWidgets.QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 0.99)
        self.conf_spin.setSingleStep(0.05)


        self.finetune_dir_label = QtWidgets.QLineEdit()
        self.finetune_dir_label.setReadOnly(True)

        form.addRow("Model weights:", model_row)
        form.addRow("Dataset dir:", dataset_dir_row)
        form.addRow("Class names:", self.class_names_edit)
        form.addRow("Epochs:", self.epochs_spin)
        form.addRow("Image size:", self.imgsz_spin)
        form.addRow("Batch size:", self.batch_spin)
        form.addRow("Val split:", self.val_split_spin)
        form.addRow("Conf threshold:", self.conf_spin)
        form.addRow("Finetune dir:", self.finetune_dir_label)

        # ── Tracking settings ──
        self.tracker_type_combo = QtWidgets.QComboBox()
        self.tracker_type_combo.addItems([
            "botsort", "deepocsort", "bytetrack", "ocsort", "strongsort", "hybridsort"
        ])

        self.reid_weights_edit = QtWidgets.QLineEdit("osnet_x0_25_msmt17.pt")
        reid_row = QtWidgets.QHBoxLayout()
        reid_row.addWidget(self.reid_weights_edit, stretch=1)
        self.reid_browse_btn = QtWidgets.QPushButton("Browse...")
        self.reid_browse_btn.clicked.connect(self._browse_reid)
        reid_row.addWidget(self.reid_browse_btn)

        self.with_reid_chk = QtWidgets.QCheckBox("Enable ReID")
        self.with_reid_chk.setChecked(True)

        self.track_high_spin = QtWidgets.QDoubleSpinBox()
        self.track_high_spin.setRange(0.01, 0.99); self.track_high_spin.setSingleStep(0.05)
        self.track_high_spin.setValue(0.6)

        self.track_low_spin = QtWidgets.QDoubleSpinBox()
        self.track_low_spin.setRange(0.01, 0.99); self.track_low_spin.setSingleStep(0.05)
        self.track_low_spin.setValue(0.1)

        self.new_track_spin = QtWidgets.QDoubleSpinBox()
        self.new_track_spin.setRange(0.01, 0.99); self.new_track_spin.setSingleStep(0.05)
        self.new_track_spin.setValue(0.7)

        self.track_buffer_spin = QtWidgets.QSpinBox()
        self.track_buffer_spin.setRange(1, 300); self.track_buffer_spin.setValue(30)

        self.match_thresh_spin = QtWidgets.QDoubleSpinBox()
        self.match_thresh_spin.setRange(0.01, 0.99); self.match_thresh_spin.setSingleStep(0.05)
        self.match_thresh_spin.setValue(0.8)

        self.proximity_spin = QtWidgets.QDoubleSpinBox()
        self.proximity_spin.setRange(0.01, 0.99); self.proximity_spin.setSingleStep(0.05)
        self.proximity_spin.setValue(0.5)

        self.appearance_spin = QtWidgets.QDoubleSpinBox()
        self.appearance_spin.setRange(0.01, 0.99); self.appearance_spin.setSingleStep(0.05)
        self.appearance_spin.setValue(0.25)

        # ── Add rows ──
        form.addRow("── Tracking ──", QtWidgets.QLabel(""))
        form.addRow("Tracker type:", self.tracker_type_combo)
        form.addRow("ReID weights:", reid_row)
        form.addRow("", self.with_reid_chk)
        form.addRow("Track high thresh:", self.track_high_spin)
        form.addRow("Track low thresh:", self.track_low_spin)
        form.addRow("New track thresh:", self.new_track_spin)
        form.addRow("Track buffer:", self.track_buffer_spin)
        form.addRow("Match thresh:", self.match_thresh_spin)
        form.addRow("Proximity thresh:", self.proximity_spin)
        form.addRow("Appearance thresh:", self.appearance_spin)


        self.save_btn = QtWidgets.QPushButton("Save settings")
        self.save_btn.setFixedWidth(160)
        self.save_btn.clicked.connect(self._on_save)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addSpacing(10)
        layout.addWidget(self.save_btn, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addStretch(1)




    def load_config(self, cfg: dict):
        self._cfg = cfg
        self.model_path_edit.setText(cfg.get("model_path", ""))
        self.dataset_dir_edit.setText(cfg.get("dataset_dir", ""))
        names = cfg.get("class_names", ["object"])
        self.class_names_edit.setText(", ".join(names) if isinstance(names, list) else str(names))
        self.epochs_spin.setValue(cfg.get("epochs", 20))
        self.imgsz_spin.setValue(cfg.get("imgsz", 1024))
        self.batch_spin.setValue(cfg.get("batch", 16))
        self.val_split_spin.setValue(cfg.get("val_split", 0.1))
        self.conf_spin.setValue(cfg.get("conf_threshold", 0.5))
        self.finetune_dir_label.setText(cfg.get("finetune_dir", ""))

        # -- tracker --
        self.tracker_type_combo.setCurrentText(cfg.get("tracker_type", "botsort"))
        self.reid_weights_edit.setText(cfg.get("reid_weights", "osnet_x0_25_msmt17.pt"))
        self.with_reid_chk.setChecked(cfg.get("with_reid", True))
        self.track_high_spin.setValue(cfg.get("track_high_thresh", 0.6))
        self.track_low_spin.setValue(cfg.get("track_low_thresh", 0.1))
        self.new_track_spin.setValue(cfg.get("new_track_thresh", 0.7))
        self.track_buffer_spin.setValue(cfg.get("track_buffer", 30))
        self.match_thresh_spin.setValue(cfg.get("match_thresh", 0.8))
        self.proximity_spin.setValue(cfg.get("proximity_thresh", 0.5))
        self.appearance_spin.setValue(cfg.get("appearance_thresh", 0.25))

    def to_config(self) -> dict:
        names_raw = self.class_names_edit.text()
        names = [n.strip() for n in names_raw.split(",") if n.strip()]
        if not names:
            names = ["object"]
        cfg = dict(self._cfg)
        cfg.update({
            "model_path": self.model_path_edit.text(),
            "dataset_dir": self.dataset_dir_edit.text(),
            "class_names": names,
            "epochs": self.epochs_spin.value(),
            "imgsz": self.imgsz_spin.value(),
            "batch": self.batch_spin.value(),
            "val_split": self.val_split_spin.value(),
            "conf_threshold": self.conf_spin.value(),
            "tracker_type":       self.tracker_type_combo.currentText(),
            "reid_weights":       self.reid_weights_edit.text(),
            "with_reid":          self.with_reid_chk.isChecked(),
            "track_high_thresh":  self.track_high_spin.value(),
            "track_low_thresh":   self.track_low_spin.value(),
            "new_track_thresh":   self.new_track_spin.value(),
            "track_buffer":       self.track_buffer_spin.value(),
            "match_thresh":       self.match_thresh_spin.value(),
            "proximity_thresh":   self.proximity_spin.value(),
            "appearance_thresh":  self.appearance_spin.value(),
        })
        return cfg

    def _browse_model(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select model weights", "",
            "Model files (*.pt *.ckpt *.pth);;All files (*)",
        )
        if path:
            self.model_path_edit.setText(path)
    
    def _browse_dataset(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select dataset, WARNING it will be modify", "",
        )
        if path:
            self.dataset_dir_edit.setText(path)
    
    def _browse_reid(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select ReID weights", "",
            "Model files (*.pt *.pth *.onnx);;All files (*)",
        )
        if path:
            self.reid_weights_edit.setText(path)


    def _on_save(self):
        self.config_changed.emit()

# ---------------------------------------------------------------------------
# Inspect Dataset page
# ---------------------------------------------------------------------------

class InspectDatasetPage(QtWidgets.QWidget):
    """Browse images + labels from the project dataset (train / val).
    On the val split, can overlay model predictions with confidence scores.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._launcher: Optional["LauncherWindow"] = None
        self._items: List[Dict[str, str]] = []   # [{img_path, lbl_path}, ...]
        self._current: int = 0
        self._current_bgr: Optional[np.ndarray] = None
        self._pred_cache: Dict[int, List[OBBOX]] = {}   # idx → predictions
        self._show_preds: bool = False

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
        self.run_pred_btn.setToolTip("Run the detector on the current image and overlay predictions")
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

    # ---- Data loading ----

    def refresh(self):
        """Reload the current split from the project dataset dir."""
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
            [p for p in img_dir.iterdir() if p.suffix.lower() in exts],
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
                    total_annots += len([l for l in lines if l.strip()])
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

    # ---- Navigation ----

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

    # ---- Display ----

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

        # Draw ground-truth labels (green)
        gt_boxes = self._parse_label(item["lbl"], w, h)
        for box in gt_boxes:
            pts = box.poly.reshape(-1, 2).astype(int)
            cv2.polylines(img, [pts], True, (0, 255, 0), 2, cv2.LINE_AA)

        # Draw predictions (cyan, with confidence) if toggled
        if self._show_preds and self._current in self._pred_cache:
            preds = self._pred_cache[self._current]
            for box in preds:
                if box.conf < conf_thresh:
                    continue
                pts = box.poly.reshape(-1, 2).astype(int)
                cv2.polylines(img, [pts], True, (255, 255, 0), 2, cv2.LINE_AA)
                # Confidence text
                x0, y0 = int(pts[0, 0]), int(pts[0, 1])
                label = f"{box.conf:.2f}"
                (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (x0, y0 - th - base - 4), (x0 + tw + 6, y0), (255, 255, 0), -1)
                cv2.putText(img, label, (x0 + 3, y0 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Draw legend
        self._draw_legend(img, gt_boxes, conf_thresh)

        # Scale to fit canvas
        self._display(img)

    def _draw_legend(self, img: np.ndarray, gt_boxes, conf_thresh: float):
        """Small legend in the top-left corner."""
        lines = [f"GT: {len(gt_boxes)} boxes"]
        if self._show_preds and self._current in self._pred_cache:
            preds = self._pred_cache[self._current]
            n_shown = sum(1 for b in preds if b.conf >= conf_thresh)
            lines.append(f"Pred: {n_shown}/{len(preds)} (conf>={conf_thresh:.2f})")

        y = 20
        for line in lines:
            (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(img, (8, y - th - 4), (14 + tw, y + 4), (0, 0, 0), -1)
            cv2.putText(img, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            y += th + 12

    def _display(self, img_bgr: np.ndarray):
        """Fit image into the canvas label."""
        qimg = cvimg_to_qimage(img_bgr)
        lbl_w, lbl_h = self.canvas.width(), self.canvas.height()
        pix = QtGui.QPixmap.fromImage(qimg).scaled(
            lbl_w, lbl_h,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self.canvas.setPixmap(pix)

    # ---- Label parsing ----

    def _parse_label(self, lbl_path: str, img_w: int, img_h: int) -> List[OBBOX]:
        """Parse a YOLO-OBB label file and return denormalized OBBOX list."""
        if not lbl_path or not os.path.isfile(lbl_path):
            return []
        boxes = []
        try:
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 9:
                        continue
                    cls_id = int(parts[0])
                    coords = [float(x) for x in parts[1:9]]
                    pts = np.array(coords, dtype=np.float32).reshape(4, 2)
                    pts[:, 0] *= img_w
                    pts[:, 1] *= img_h
                    boxes.append(OBBOX(poly=pts, cls_id=cls_id, conf=1.0))
        except Exception:
            pass
        return boxes

    # ---- Prediction overlay (val only) ----

    def _toggle_preds(self, checked: bool):
        self._show_preds = checked
        self._redraw()

    def _run_prediction(self):
        """Run the detector on the current image and cache predictions."""
        if not self._items or self._current_bgr is None:
            return
        if not self._launcher:
            return

        cfg = self._launcher.project_config()
        model_path = cfg.get("model_path", YOLO_MODEL_PATH)
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
            model_path=model_path,
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
            self._launcher.statusBar().showMessage(f"Prediction error: {msg}", 5000)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._current_bgr is not None:
            self._redraw()


# ---------------------------------------------------------------------------
# Train page — wraps the finetune action with project-aware settings
# ---------------------------------------------------------------------------

class TrainPage(QtWidgets.QWidget):
    """Training page with progress bar, epoch metrics table, and console log."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._launcher: Optional["LauncherWindow"] = None

        # --- Train button ---
        self.train_btn = QtWidgets.QPushButton("Launch Training (Finetune)")
        self.train_btn.setFixedHeight(40)
        self.train_btn.clicked.connect(self._on_train)

        # --- Progress bar ---
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Idle")

        # --- Epoch metrics table ---
        self.metrics_table = QtWidgets.QTableWidget()
        self.metrics_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.metrics_table.setAlternatingRowColors(True)
        self.metrics_table.verticalHeader().setVisible(False)
        self.metrics_table.setMaximumHeight(600)

        # --- Console log ---
        self.log_text = QtWidgets.QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFontFamily("Consolas" if os.name == "nt" else "monospace")

        # --- Layout ---
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(16, 16, 16, 16)
        v.setSpacing(10)
        v.addWidget(self.train_btn)
        v.addWidget(self.progress_bar)
        v.addWidget(QtWidgets.QLabel("Epoch metrics:"))
        v.addWidget(self.metrics_table)
        v.addWidget(QtWidgets.QLabel("Console output:"))
        v.addWidget(self.log_text, stretch=1)

    def set_launcher(self, launcher: "LauncherWindow"):
        self._launcher = launcher

    def log(self, msg: str):
        self.log_text.append(msg)
        # Auto-scroll to bottom
        sb = self.log_text.verticalScrollBar()
        sb.setValue(sb.maximum())

    def set_progress(self, msg: str, frac: float):
        pct = int(frac * 100)
        self.progress_bar.setValue(pct)
        self.progress_bar.setFormat(f"{msg}  —  {pct}%")

    def update_metrics(self, epoch: int, total_epochs: int, metrics: dict):
        """Add or update a row in the metrics table for this epoch."""
        if not metrics:
            return

        # Build column list from all metric keys seen so far
        all_keys = ["Epoch"]
        # Gather existing columns
        for col in range(self.metrics_table.columnCount()):
            header = self.metrics_table.horizontalHeaderItem(col)
            if header and header.text() != "Epoch":
                all_keys.append(header.text())
        # Add new keys
        for k in sorted(metrics.keys()):
            if k not in all_keys:
                all_keys.append(k)

        self.metrics_table.setColumnCount(len(all_keys))
        self.metrics_table.setHorizontalHeaderLabels(all_keys)

        # Find or create the row for this epoch
        row = -1
        for r in range(self.metrics_table.rowCount()):
            item = self.metrics_table.item(r, 0)
            if item and item.text() == str(epoch):
                row = r
                break
        if row < 0:
            row = self.metrics_table.rowCount()
            self.metrics_table.insertRow(row)

        # Fill in data
        self.metrics_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(epoch)))
        for k, v in metrics.items():
            if k in all_keys:
                col = all_keys.index(k)
                self.metrics_table.setItem(
                    row, col, QtWidgets.QTableWidgetItem(f"{v:.4f}")
                )

        self.metrics_table.resizeColumnsToContents()
        self.metrics_table.scrollToBottom()

    def reset_for_new_run(self):
        """Clear everything for a fresh training run."""
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Starting...")
        self.metrics_table.setRowCount(0)
        self.metrics_table.setColumnCount(0)
        self.log_text.clear()

    def _on_train(self):
        if self._launcher:
            self._launcher.annotate_page.finetune_model()


# ---------------------------------------------------------------------------
# Annotation page (the existing Base / OBB_VideoPlayer)
# ---------------------------------------------------------------------------

class AnnotatePage(QtWidgets.QWidget):
    """Wraps the annotation canvas + controls as a page widget (not a QMainWindow)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._launcher: Optional["LauncherWindow"] = None

        # --- Source / playback state ---
        self.source: Optional[FrameSource] = None
        self.total_frames: int = 0
        self.current_idx: int = 0
        self.current_frame_bgr: Optional[np.ndarray] = None
        self.src_path: Optional[str] = None
        self.play_timer = QtCore.QTimer(self)
        self.play_timer.timeout.connect(self._on_play_tick)
        self.playing = False

        # --- Zoom & pan ---
        self.zoom = 1.0
        self.min_zoom = 0.25
        self.max_zoom = 8.0
        self.pan_img = np.array([0.0, 0.0], dtype=np.float32)

        # --- Annotations ---
        self.pred_cache: Dict[int, List[PolyClass]] = {}
        self.class_names = None
        self.selected_idx: Optional[int] = None

        # --- Verified dataset ---
        self.dataset: Dict[int, List[PolyClass]] = {}
        self.dataset_images_names: Dict[int, str] = {}

        # --- Display mapping ---
        self.draw_map: Dict[str, float] = {"scale": 1.0, "xoff": 0, "yoff": 0}

        # --- Interaction state ---
        self.space_held = False
        self.mode = "select"
        self.temp_poly_pts: List[List[float]] = []
        self.dragging = False
        self.drag_start_img: Optional[tuple] = None
        self.orig_poly: Optional[np.ndarray] = None
        self.vertex_drag_idx: Optional[int] = None

        # --- Model ---
        self.model_worker = DetectionWorker
        self.model_path = YOLO_MODEL_PATH
        self.dataset_dir = ""

        # ================ UI ================
        self.video_label = QtWidgets.QLabel()
        self.video_label.setMouseTracking(True)
        self.video_label.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.video_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background:#111; border:1px solid #333;")
        self.video_label.setMinimumSize(720, 405)
        self.video_label.installEventFilter(self)

        # Buttons
        self.add_btn = QtWidgets.QPushButton("Add (N)")
        self.add_btn.clicked.connect(self.start_add_mode)
        self.edit_btn = QtWidgets.QPushButton("Edit (E)")
        self.edit_btn.clicked.connect(self.toggle_edit_mode)
        self.verify_btn = QtWidgets.QPushButton("Verify (V)")
        self.verify_btn.clicked.connect(self.verify_selected_toggle)
        self.delete_btn = QtWidgets.QPushButton("Delete (Del)")
        self.delete_btn.clicked.connect(self.delete_selected)

        self.zoom_in_btn = QtWidgets.QPushButton("Zoom +")
        self.zoom_out_btn = QtWidgets.QPushButton("Zoom −")
        self.zoom_fit_btn = QtWidgets.QPushButton("Fit")
        self.zoom_in_btn.clicked.connect(lambda: self.zoom_step(+1))
        self.zoom_out_btn.clicked.connect(lambda: self.zoom_step(-1))
        self.zoom_fit_btn.clicked.connect(self.zoom_fit)

        self.open_video_btn = QtWidgets.QPushButton("Open video")
        self.open_images_btn = QtWidgets.QPushButton("Open image folder")
        self.prev_btn = QtWidgets.QPushButton("⟸ Prev (←)")
        self.next_btn = QtWidgets.QPushButton("Next (→) ⟹")
        self.run_btn = QtWidgets.QPushButton("Run Model")
        self.export_dataset_btn = QtWidgets.QPushButton("Export to Dataset (D)")
        self.play_btn = QtWidgets.QPushButton("Play ▶")
        self.pause_btn = QtWidgets.QPushButton("Pause ⏸")

        self.inference_conf_tresh = QtWidgets.QDoubleSpinBox()
        self.inference_conf_tresh.setRange(0.01, 0.99)
        self.inference_conf_tresh.setSingleStep(0.05)
        self.inference_conf_tresh.setValue(0.5)
        self.inference_conf_tresh.setPrefix("conf=")

        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.sliderReleased.connect(self._on_slider_released)

        # --- Signals ---
        self.open_video_btn.clicked.connect(self.open_video)
        self.open_images_btn.clicked.connect(self.open_folder)
        self.prev_btn.clicked.connect(self.prev_frame)
        self.next_btn.clicked.connect(self.next_frame)
        self.run_btn.clicked.connect(self.run_model_cached)
        self.export_dataset_btn.clicked.connect(self.export_to_dataset)
        self.play_btn.clicked.connect(self.play)
        self.pause_btn.clicked.connect(self.pause)

        # --- Layout ---
        left_stack = QtWidgets.QWidget()
        left_v = QtWidgets.QVBoxLayout(left_stack)
        left_v.setContentsMargins(0, 0, 0, 0)
        left_v.setSpacing(6)
        left_v.addWidget(self.video_label, stretch=1)
        left_v.addWidget(self.frame_slider)

        content_row = QtWidgets.QHBoxLayout()
        content_row.setContentsMargins(0, 0, 0, 0)
        content_row.setSpacing(10)
        content_row.addWidget(left_stack, stretch=1)
        content_row.addWidget(self._build_side_panel(), stretch=0)

        transport = self._build_transport_bar()

        page = QtWidgets.QVBoxLayout(self)
        page.setContentsMargins(8, 8, 8, 8)
        page.setSpacing(8)
        page.addLayout(content_row, stretch=1)
        page.addWidget(transport)

    def set_launcher(self, launcher: "LauncherWindow"):
        self._launcher = launcher

    def _status(self, msg: str):
        if self._launcher:
            self._launcher.statusBar().showMessage(msg, 5000)

    # ---- Layout helpers ----

    def _build_transport_bar(self) -> QtWidgets.QWidget:
        bar = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(bar)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(10)
        h.addStretch(1)
        h.addWidget(self.prev_btn)
        h.addWidget(self.play_btn)
        h.addWidget(self.pause_btn)
        h.addWidget(self.next_btn)
        h.addSpacing(20)
        h.addWidget(self.zoom_out_btn)
        h.addWidget(self.zoom_in_btn)
        h.addWidget(self.zoom_fit_btn)
        h.addStretch(1)
        return bar

    def _build_side_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(panel)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(8)
        v.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        # Source group
        src_box = QtWidgets.QGroupBox("Source")
        src_l = QtWidgets.QVBoxLayout(src_box)
        src_l.addWidget(self.open_video_btn)
        src_l.addWidget(self.open_images_btn)

        # Inference group
        infer_box = QtWidgets.QGroupBox("Inference")
        infer_l = QtWidgets.QVBoxLayout(infer_box)
        infer_l.addWidget(self.run_btn)
        infer_l.addWidget(self.inference_conf_tresh)
        infer_l.addWidget(self.export_dataset_btn)

        # Annotation group
        anno_box = QtWidgets.QGroupBox("Annotation")
        anno_l = QtWidgets.QVBoxLayout(anno_box)
        anno_l.addWidget(self.add_btn)
        anno_l.addWidget(self.edit_btn)
        anno_l.addWidget(self.verify_btn)
        anno_l.addWidget(self.delete_btn)

        v.addWidget(src_box)
        v.addWidget(infer_box)
        v.addWidget(anno_box)
        v.addStretch(1)

        panel.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        return panel

    # ---- Apply project config ----

    def apply_config(self, cfg: dict):
        self.model_path = cfg.get("model_path", YOLO_MODEL_PATH)
        self.dataset_dir = cfg.get("dataset_dir", "")
        names = cfg.get("class_names", ["object"])
        self.class_names = names if isinstance(names, list) else [names]
        self.inference_conf_tresh.setValue(cfg.get("conf_threshold", 0.5))

    # ==================== Source I/O ====================

    def open_video(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Video", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)",
        )
        if path:
            self.load_video(path)

    def open_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Open Image Folder", "")
        if folder:
            self.load_folder(folder)

    def _set_source(self, src: FrameSource):
        if self.source:
            try: self.source.close()
            except: pass
        self.pred_cache.clear()
        self.dataset.clear()
        self.dataset_images_names.clear()
        self.selected_idx = None
        self.mode = "select"
        self.temp_poly_pts.clear()
        self.source = src
        self.total_frames = src.count()
        self.src_path = getattr(src, "path", None)
        self.current_idx = 0
        self.frame_slider.setRange(0, max(0, self.total_frames - 1))
        self.frame_slider.setValue(0)
        self._status(f"Loaded: {src.name()} | frames={self.total_frames} | fps={src.fps():.2f}")
        self.read_frame(self.current_idx)

    def load_video(self, path: str):
        try: src = VideoSource(path)
        except Exception as e:
            self._status(f"Failed to open video: {e}"); return
        self._set_source(src)

    def load_folder(self, folder: str):
        try: src = ImageFolderSource(folder)
        except Exception as e:
            self._status(f"Failed to open folder: {e}"); return
        self._set_source(src)

    # ==================== Frame reading & display ====================

    def read_frame(self, idx: int) -> bool:
        if not self.source: return False
        idx = max(0, min(idx, self.total_frames - 1))
        frame = self.source.read(idx)
        if frame is None:
            self._status("Failed to read frame."); return False
        self.current_idx = idx
        self.current_frame_bgr = frame
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(idx)
        self.frame_slider.blockSignals(False)
        if self._launcher:
            self._launcher.update_title()
        self.redraw_current()
        return True

    def show_frame(self, frame_bgr: np.ndarray):
        qimg = cvimg_to_qimage(frame_bgr)
        img_w, img_h = qimg.width(), qimg.height()
        lbl_w, lbl_h = self.video_label.width(), self.video_label.height()
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

        self.draw_map = {
            "scale": scale, "xoff": xoff, "yoff": yoff,
            "img_w": img_w, "img_h": img_h,
            "panx": float(self.pan_img[0]), "pany": float(self.pan_img[1]),
            "base": base, "lbl_w": lbl_w, "lbl_h": lbl_h,
        }
        self.video_label.setPixmap(canvas)

    def redraw_current(self):
        if self.current_frame_bgr is None: return
        base = self.current_frame_bgr
        annots = self.pred_cache.get(self.current_idx, [])
        annotated = draw_annotations(
            base, annots, self.inference_conf_tresh.value(),
            self.class_names, self.selected_idx,
            show_conf=False, show_label=False,
        )
        if self.mode == "add" and self.temp_poly_pts:
            ghost = np.array(self.temp_poly_pts, dtype=np.int32)
            cv2.polylines(annotated, [ghost], isClosed=False,
                          color=(200, 200, 200), thickness=1, lineType=cv2.LINE_AA)
            for (gx, gy) in ghost:
                cv2.circle(annotated, (int(gx), int(gy)), 3,
                           (200, 200, 200), -1, lineType=cv2.LINE_AA)
        self.show_frame(annotated)

    # ==================== Playback controls ====================

    def prev_frame(self):
        if not self.source: return
        self.pause(); self.read_frame(self.current_idx - 1)

    def next_frame(self):
        if not self.source: return
        self.pause(); self.read_frame(self.current_idx + 1)

    def _on_slider_released(self):
        if not self.source: return
        self.pause(); self.read_frame(self.frame_slider.value())

    def play(self):
        if not self.source or self.playing: return
        fps = self.source.fps() or 25
        self.play_timer.start(max(15, int(1000 / fps)))
        self.playing = True

    def pause(self):
        if self.playing:
            self.play_timer.stop()
            self.playing = False

    def _on_play_tick(self):
        if self.current_idx + 1 >= self.total_frames:
            self.pause(); return
        self.read_frame(self.current_idx + 1)

    # ==================== Model inference ====================

    def run_model_cached(self):
        idx = self.current_idx
        if self.current_frame_bgr is None: return

        self.run_btn.setEnabled(False)
        self.run_btn.setText("Inference running...")
        conf = float(self.inference_conf_tresh.value())

        # Prefer passing the file path directly to YOLO when possible
        source_path = None
        if isinstance(self.source, ImageFolderSource):
            source_path = self.source.path_at(idx)

        self.worker_thread = QtCore.QThread(self)
        self.worker = self.model_worker(
            idx, self.current_frame_bgr,
            conf=conf, imgsz=1024, model_path=self.model_path,
            source_path=source_path,
        )
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._on_inference_done)
        self.worker.error.connect(self._on_inference_error)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.error.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.start()

    def _on_inference_done(self, frame_idx: int, class_names, annots: List[PolyClass]):
        self.class_names = class_names
        self.pred_cache[frame_idx] = annots
        self.selected_idx = None
        if frame_idx == self.current_idx:
            self.redraw_current()
        self.run_btn.setEnabled(True)
        self.run_btn.setText("Run Model")
        self._status(f"Predictions cached for frame {frame_idx + 1}.")

    def _on_inference_error(self, msg: str):
        self.run_btn.setEnabled(True)
        self.run_btn.setText("Run Model")
        self._status(f"Inference error: {msg}")

    # ==================== Finetuning ====================

    def finetune_model(self):
        if not self.src_path:
            QtWidgets.QMessageBox.warning(self.window(), "Fine-tune", "Load a source first.")
            return
        if not self.dataset:
            QtWidgets.QMessageBox.warning(self.window(), "Fine-tune", "No verified annotations to train on.")
            return

        n_new = self._export_verified_to_dataset(val_split=0.1)
        data_yaml = self._ensure_data_yaml()
        self._status(f"Exported {n_new} new images to {self.dataset_dir}")

        ds = Path(self.dataset_dir)
        n_train = sum(1 for _ in (ds / "images" / "train").glob("*"))
        if n_train == 0:
            QtWidgets.QMessageBox.warning(self.window(), "Fine-tune", "0 training images after export.")
            return

        cfg = self._launcher.project_config() if self._launcher else {}

        worker = DetectFinetuneWorker(
            class_names=self.class_names,
            base_model_path=self.model_path,
            out_root=cfg.get("finetune_dir", os.path.join(os.getcwd(), "finetune_runs")),
            epochs=cfg.get("epochs", 20),
            imgsz=cfg.get("imgsz", 1024),
            batch=cfg.get("batch", 16),
            val_split=cfg.get("val_split", 0.1),
            data_yaml=data_yaml,
        )

        bridge = _FinetuneSignalBridge(self)
        bridge.progress.connect(lambda msg, p: self._status(f"{msg} ({int(p * 100)}%)"))
        bridge.error.connect(self._on_finetune_error)
        bridge.finished.connect(self._on_finetune_done)
        self._finetune_bridge = bridge

        # Connect to train page for live feedback
        train_page = getattr(self._launcher, "train_page", None) if self._launcher else None
        if train_page:
            train_page.reset_for_new_run()
            train_page.train_btn.setEnabled(False)
            bridge.progress.connect(train_page.set_progress)

        def _run():
            worker.progress.connect(bridge.progress)
            worker.error.connect(bridge.error)
            worker.finished.connect(bridge.finished)
            # Per-epoch metrics and console lines → train page
            if train_page:
                worker.epoch_metrics.connect(
                    lambda ep, tot, m: train_page.update_metrics(ep, tot, m)
                )
                worker.log_line.connect(train_page.log)
            worker.run()

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        self._finetune_thread = t

    def _on_finetune_error(self, msg: str):
        QtWidgets.QMessageBox.critical(self.window(), "Fine-tune Error", msg)
        train_page = getattr(self._launcher, "train_page", None) if self._launcher else None
        if train_page:
            train_page.train_btn.setEnabled(True)
            train_page.log(f"ERROR: {msg}")
            train_page.progress_bar.setFormat("Error")

    def _on_finetune_done(self, best_pt_path: str):
        self._status(f"Fine-tune complete: {best_pt_path}")
        try:
            self.model_worker._model = YOLO(best_pt_path)
            self.model_path = best_pt_path
        except Exception as e:
            self._status(f"Model saved but failed to load: {e}")
        train_page = getattr(self._launcher, "train_page", None) if self._launcher else None
        if train_page:
            train_page.train_btn.setEnabled(True)
            train_page.log(f"Training complete — weights: {best_pt_path}")
            train_page.set_progress("Complete!", 1.0)

    # ==================== Dataset export ====================

    def _get_frame_image(self, frame_idx: int) -> Optional[np.ndarray]:
        if self.source is None: return None
        return self.source.read(frame_idx)

    def _poly_to_yolo_obb_line(self, box: PolyClass, img_w: int, img_h: int) -> str:
        pts = box.poly.reshape(-1, 2)
        parts = [str(int(box.cls_id))]
        for x, y in pts:
            parts.append(f"{x / img_w:.6f}")
            parts.append(f"{y / img_h:.6f}")
        return " ".join(parts)

    def _export_verified_to_dataset(self, val_split: float = 0.1) -> int:
        ds = Path(self.dataset_dir)
        for split in ("train", "val"):
            (ds / "images" / split).mkdir(parents=True, exist_ok=True)
            (ds / "labels" / split).mkdir(parents=True, exist_ok=True)

        existing_stems = set()
        for split in ("train", "val"):
            for p in (ds / "images" / split).iterdir():
                if p.is_file():
                    existing_stems.add(p.stem)

        exported = 0
        for frame_idx, boxes in self.dataset.items():
            if not boxes: continue
            src_name = self.source.name() if self.source else "src"
            stem = f"{Path(src_name).stem}_frame{frame_idx:06d}"
            if stem in existing_stems: continue

            img = self._get_frame_image(frame_idx)
            if img is None: continue
            h, w = img.shape[:2]
            split = "val" if random.random() < val_split else "train"

            cv2.imwrite(str(ds / "images" / split / f"{stem}.jpg"), img)
            lines = [self._poly_to_yolo_obb_line(b, w, h) for b in boxes]
            (ds / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n")
            exported += 1
        return exported

    def _ensure_data_yaml(self) -> str:
        ds = Path(self.dataset_dir)
        yaml_path = ds / "data.yaml"
        names = self.class_names if self.class_names else ["object"]
        nc = len(names)
        content = (
            f"path: {ds.resolve()}\n"
            f"train: images/train\nval: images/val\n\n"
            f"nc: {nc}\nnames: {names}\n"
        )
        yaml_path.write_text(content)
        return str(yaml_path)

    def export_to_dataset(self):
        if not self.src_path:
            QtWidgets.QMessageBox.warning(self.window(), "Export", "Load a source first.")
            return
        if not self.dataset:
            QtWidgets.QMessageBox.warning(self.window(), "Export", "No verified annotations to export.")
            return

        n_new = self._export_verified_to_dataset(val_split=0.1)
        self._ensure_data_yaml()

        ds = Path(self.dataset_dir)
        n_train = sum(1 for _ in (ds / "images" / "train").glob("*"))
        n_val = sum(1 for _ in (ds / "images" / "val").glob("*"))

        self._status(f"Exported {n_new} new images → {self.dataset_dir}  (total: {n_train} train + {n_val} val)")
        QtWidgets.QMessageBox.information(
            self.window(), "Export done",
            f"{n_new} new images exported to:\n{os.path.abspath(self.dataset_dir)}\n\n"
            f"Dataset totals: {n_train} train / {n_val} val",
        )

    # ==================== Mouse / keyboard interaction ====================

    def display_to_image_coords(self, xd: int, yd: int):
        m = self.draw_map
        if not m: return None, None
        s = m["scale"]
        xi = (xd - (m["xoff"] - m["panx"] * s)) / s
        yi = (yd - (m["yoff"] - m["pany"] * s)) / s
        if xi < 0 or yi < 0 or xi >= m["img_w"] or yi >= m["img_h"]:
            return None, None
        return float(xi), float(yi)

    def _check_canvas_mouse_event(self, event):
        if self.current_frame_bgr is None: return None
        if not hasattr(event, "position"): return None
        pos = event.position()
        x_disp, y_disp = int(pos.x()), int(pos.y())
        x_img, y_img = self.display_to_image_coords(x_disp, y_disp)
        if x_img is None: return None
        return x_disp, y_disp, x_img, y_img

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.Type.MouseButtonPress:
            if self.inference_conf_tresh.hasFocus():
                self.inference_conf_tresh.clearFocus()
            self.video_label.setFocus(QtCore.Qt.FocusReason.MouseFocusReason)

        if obj is not self.video_label:
            return super().eventFilter(obj, event)

        if event.type() == QtCore.QEvent.Type.Wheel:
            delta = event.angleDelta().y()
            if delta > 0: self.zoom_step(+1, anchor_disp=event.position())
            elif delta < 0: self.zoom_step(-1, anchor_disp=event.position())
            return True

        if event.type() in (
            QtCore.QEvent.Type.MouseButtonPress,
            QtCore.QEvent.Type.MouseMove,
            QtCore.QEvent.Type.MouseButtonRelease,
        ):
            coords = self._check_canvas_mouse_event(event)
            if coords is None: return False
            x_disp, y_disp, x_img, y_img = coords

        # PAN
        if event.type() == QtCore.QEvent.Type.MouseButtonPress:
            if event.button() == QtCore.Qt.MouseButton.LeftButton and self.space_held:
                self._pan_dragging = True
                self._pan_last_disp = (x_disp, y_disp)
                return True
        elif event.type() == QtCore.QEvent.Type.MouseMove:
            if getattr(self, "_pan_dragging", False):
                dx = x_disp - self._pan_last_disp[0]
                dy = y_disp - self._pan_last_disp[1]
                s = self.draw_map.get("scale", 1.0)
                self.pan_img[0] -= dx / s
                self.pan_img[1] -= dy / s
                self._pan_last_disp = (x_disp, y_disp)
                self.redraw_current(); return True
        elif event.type() == QtCore.QEvent.Type.MouseButtonRelease:
            if getattr(self, "_pan_dragging", False):
                self._pan_dragging = False; return True

        # LEFT CLICK
        if event.type() == QtCore.QEvent.Type.MouseButtonPress:
            if event.button() == QtCore.Qt.MouseButton.LeftButton:
                if self.mode == "add":
                    self.add_click_point(x_img, y_img); return True
                hit_idx = self.pick_annot(x_img, y_img)
                if hit_idx is not None:
                    self.selected_idx = hit_idx
                    self.redraw_current()
                    if self.mode == "edit" and (event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier):
                        v = self.pick_vertex(x_img, y_img)
                        if v is not None:
                            self.vertex_drag_idx = v; self.dragging = True; return True
                    boxes = self.pred_cache.get(self.current_idx, [])
                    if self.selected_idx is not None and self.selected_idx < len(boxes):
                        self.dragging = True
                        self.drag_start_img = (x_img, y_img)
                        self.orig_poly = boxes[self.selected_idx].poly.copy()
                    return True
                else:
                    if self.mode != "add":
                        self.selected_idx = None; self.redraw_current()
                    return True
            elif event.button() == QtCore.Qt.MouseButton.RightButton:
                hit_idx = self.pick_annot(x_img, y_img)
                if hit_idx is not None:
                    self.selected_idx = hit_idx
                    self.verify_selected_toggle()
                    self.redraw_current()
                return True

        # DRAG
        elif event.type() == QtCore.QEvent.Type.MouseMove:
            if self.dragging:
                if self.vertex_drag_idx is not None:
                    self._set_vertex_selected(self.vertex_drag_idx, x_img, y_img)
                elif self.drag_start_img is not None:
                    dx = x_img - self.drag_start_img[0]
                    dy = y_img - self.drag_start_img[1]
                    self._translate_selected(dx, dy)
                self.redraw_current(); return True

        # RELEASE
        elif event.type() == QtCore.QEvent.Type.MouseButtonRelease:
            if self.dragging:
                self.dragging = False
                self.vertex_drag_idx = None
                self.drag_start_img = None
                self.orig_poly = None
                self.update_dataset_for_frame(self.current_idx)
                self.redraw_current(); return True

        return super().eventFilter(obj, event)

    # ==================== Picking ====================

    def pick_annot(self, x: float, y: float) -> Optional[int]:
        annots = self.pred_cache.get(self.current_idx, [])
        if not annots: return None
        best, best_area = None, None
        for i, b in enumerate(annots):
            if b.deleted: continue
            pts = b.poly.reshape(-1, 2).astype(np.float32)
            if cv2.pointPolygonTest(pts, (x, y), measureDist=False) >= 0:
                area = cv2.contourArea(pts.astype(np.int32))
                if best is None or area < best_area:
                    best, best_area = i, area
        return best

    def pick_vertex(self, x: float, y: float, tol_px: int = 10) -> Optional[int]:
        if self.selected_idx is None: return None
        annots = self.pred_cache.get(self.current_idx, [])
        if self.selected_idx >= len(annots): return None
        a = annots[self.selected_idx]
        if a.deleted: return None
        pts = a.poly.reshape(-1, 2)
        for i in range(pts.shape[0]):
            if np.hypot(pts[i, 0] - x, pts[i, 1] - y) <= tol_px: return i
        return None

    # ==================== Annotation actions ====================

    def verify_selected_toggle(self):
        boxes = self.pred_cache.get(self.current_idx, [])
        if self.selected_idx is None or self.selected_idx >= len(boxes): return
        box = boxes[self.selected_idx]
        if box.deleted: return
        box.verified = not box.verified
        self.update_dataset_for_frame(self.current_idx)
        self._status(f"Box #{self.selected_idx} → {'verified' if box.verified else 'unverified'}")
        self.selected_idx = None
        self.redraw_current()

    def delete_selected(self):
        boxes = self.pred_cache.get(self.current_idx, [])
        if self.selected_idx is None or self.selected_idx >= len(boxes): return
        box = boxes[self.selected_idx]
        box.deleted = True; box.verified = False
        self.update_dataset_for_frame(self.current_idx)
        self._status(f"Box #{self.selected_idx} deleted.")
        self.selected_idx = None
        self.redraw_current()

    def update_dataset_for_frame(self, frame_idx: int):
        all_boxes = self.pred_cache.get(frame_idx, [])
        self.dataset[frame_idx] = [b for b in all_boxes if b.verified and not b.deleted]
        if isinstance(self.source, ImageFolderSource):
            self.dataset_images_names[frame_idx] = self.source.path_at(frame_idx)

    # ==================== Polygon editing ====================

    def _translate_selected(self, dx: float, dy: float):
        annots = self.pred_cache.get(self.current_idx, [])
        if self.selected_idx is None or self.selected_idx >= len(annots): return
        b = annots[self.selected_idx]
        b.poly = (self.orig_poly + np.array([dx, dy], dtype=np.float32)).astype(np.float32)
        self.update_dataset_for_frame(self.current_idx)

    def _set_vertex_selected(self, idx: int, x: float, y: float):
        annots = self.pred_cache.get(self.current_idx, [])
        if self.selected_idx is None or self.selected_idx >= len(annots): return
        b = annots[self.selected_idx]
        p = b.poly.copy(); p[idx] = [x, y]
        b.poly = p.astype(np.float32)
        self.update_dataset_for_frame(self.current_idx)

    # ==================== Mode management ====================

    def set_mode(self, mode: str):
        self.mode = mode
        if mode != "add": self.temp_poly_pts.clear()
        self._status(f"Mode: {mode}")

    def start_add_mode(self):
        self.set_mode("add"); self.selected_idx = None; self.redraw_current()

    def cancel_add_mode(self):
        if self.mode == "add":
            self.temp_poly_pts.clear(); self.set_mode("select"); self.redraw_current()

    def toggle_edit_mode(self):
        self.set_mode("edit" if self.mode != "edit" else "select")

    # ==================== Add-polygon (OBB: 3 clicks) ====================

    def add_click_point(self, x: float, y: float):
        if len(self.temp_poly_pts) == 2:
            primes = find_orthogonal_projection(
                self.temp_poly_pts[0], self.temp_poly_pts[1], [x, y],
            )
            pts = np.concatenate((self.temp_poly_pts, primes), axis=0, dtype=np.float32)
            new_box = OBBOX(poly=pts, cls_id=0, conf=1.0, verified=False)
            self.pred_cache.setdefault(self.current_idx, []).append(new_box)
            self.selected_idx = len(self.pred_cache[self.current_idx]) - 1
            self.temp_poly_pts.clear()
            self.set_mode("select")
            self.update_dataset_for_frame(self.current_idx)
        else:
            self.temp_poly_pts.append([x, y])
        self.redraw_current()

    # ==================== Zoom ====================

    def zoom_fit(self):
        self.zoom = 1.0; self.pan_img[:] = 0.0; self.redraw_current()

    def _clamp_pan(self):
        m = self.draw_map
        if not m or "base" not in m: return
        s = m["base"] * self.zoom
        margin = 0.1
        max_pan_x = (m["img_w"] * s - (1.0 - margin) * m["lbl_w"]) / s / 2.0
        max_pan_y = (m["img_h"] * s - (1.0 - margin) * m["lbl_h"]) / s / 2.0
        self.pan_img[0] = float(np.clip(self.pan_img[0], -max_pan_x, max_pan_x))
        self.pan_img[1] = float(np.clip(self.pan_img[1], -max_pan_y, max_pan_y))

    def zoom_step(self, direction: int, anchor_disp=None):
        if anchor_disp is None:
            w, h = self.video_label.width(), self.video_label.height()
            anchor_disp = QtCore.QPointF(w / 2.0, h / 2.0)
        m = self.draw_map
        if not m or self.current_frame_bgr is None: return
        step = 1.25 if direction > 0 else 0.8
        new_zoom = float(np.clip(self.zoom * step, self.min_zoom, self.max_zoom))
        if abs(new_zoom - self.zoom) < 1e-6: return
        xd, yd = float(anchor_disp.x()), float(anchor_disp.y())
        xi, yi = self.display_to_image_coords(int(xd), int(yd))
        if xi is None:
            self.zoom = new_zoom; self.redraw_current(); return
        self.zoom = new_zoom
        base = float(m["base"])
        new_scale = base * self.zoom
        xoff = (m["lbl_w"] - m["img_w"] * new_scale) / 2.0
        yoff = (m["lbl_h"] - m["img_h"] * new_scale) / 2.0
        self.pan_img[0] = (xoff + xi * new_scale - xd) / new_scale
        self.pan_img[1] = (yoff + yi * new_scale - yd) / new_scale
        self._clamp_pan(); self.redraw_current()


# ---------------------------------------------------------------------------
# Launcher window — the main entry point
# ---------------------------------------------------------------------------

class LauncherWindow(QtWidgets.QMainWindow):
    """Main application window with project management and tabbed pages."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Annotation Tool")
        self.resize(1300, 820)

        self.pm = ProjectManager()
        self._current_project: Optional[str] = None

        # =============== Top bar: project selector ===============
        top_bar = QtWidgets.QWidget()
        top_h = QtWidgets.QHBoxLayout(top_bar)
        top_h.setContentsMargins(12, 8, 12, 4)
        top_h.setSpacing(10)

        top_h.addWidget(QtWidgets.QLabel("Project:"))
        self.project_combo = QtWidgets.QComboBox()
        self.project_combo.setMinimumWidth(200)
        self.project_combo.currentTextChanged.connect(self._on_project_changed)
        top_h.addWidget(self.project_combo)

        self.new_project_btn = QtWidgets.QPushButton("New project")
        self.new_project_btn.clicked.connect(self._new_project)
        top_h.addWidget(self.new_project_btn)

        self.refresh_btn = QtWidgets.QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._refresh_projects)
        top_h.addWidget(self.refresh_btn)

        self.project_label = QtWidgets.QLabel("")
        top_h.addWidget(self.project_label)
        top_h.addStretch(1)

        # =============== Tab buttons ===============
        tab_bar = QtWidgets.QWidget()
        tab_h = QtWidgets.QHBoxLayout(tab_bar)
        tab_h.setContentsMargins(12, 0, 12, 0)
        tab_h.setSpacing(4)

        self.tab_group = QtWidgets.QButtonGroup(self)
        self.tab_group.setExclusive(True)
        self.tab_buttons: List[QtWidgets.QPushButton] = []
        tab_names = ["Settings", "Annotate", "Inspect Dataset", "Train Detector", "Tracking"]

        for i, name in enumerate(tab_names):
            btn = QtWidgets.QPushButton(name)
            btn.setCheckable(True)
            self.tab_group.addButton(btn, i)
            self.tab_buttons.append(btn)
            tab_h.addWidget(btn)
        tab_h.addStretch(1)

        self.tab_group.idClicked.connect(self._switch_tab)

        # =============== Stacked pages ===============
        self.stack = QtWidgets.QStackedWidget()

        # Page 0: Settings
        self.settings_page = SettingsPage()
        self.settings_page.config_changed.connect(self._save_current_config)
        self.stack.addWidget(self.settings_page)

        # Page 1: Annotate
        self.annotate_page = AnnotatePage()
        self.annotate_page.set_launcher(self)
        self.stack.addWidget(self.annotate_page)

        # Page 2: Inspect Dataset
        self.inspect_page = InspectDatasetPage()
        self.inspect_page.set_launcher(self)
        self.stack.addWidget(self.inspect_page)

        # Page 3: Train Detector
        self.train_page = TrainPage()
        self.train_page.set_launcher(self)
        self.stack.addWidget(self.train_page)

        # Page 4: Tracking
        self.tracking_page = TrackingPage()
        self.tracking_page.set_launcher(self)
        self.stack.addWidget(self.tracking_page)

        # =============== Central layout ===============
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        vbox = QtWidgets.QVBoxLayout(central)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)
        vbox.addWidget(top_bar)
        vbox.addWidget(tab_bar)
        vbox.addWidget(self.stack, stretch=1)

        # =============== Menu bar ===============
        self._build_menu_bar()

        # =============== Keyboard shortcuts (global) ===============
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Left), self,
                        activated=self.annotate_page.prev_frame)
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Right), self,
                        activated=self.annotate_page.next_frame)
        QtGui.QShortcut(QtGui.QKeySequence("V"), self,
                        activated=self.annotate_page.verify_selected_toggle)
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Delete), self,
                        activated=self.annotate_page.delete_selected)
        QtGui.QShortcut(QtGui.QKeySequence("N"), self,
                        activated=self.annotate_page.start_add_mode)
        QtGui.QShortcut(QtGui.QKeySequence("E"), self,
                        activated=self.annotate_page.toggle_edit_mode)
        QtGui.QShortcut(QtGui.QKeySequence("Esc"), self,
                        activated=self.annotate_page.cancel_add_mode)
        QtGui.QShortcut(QtGui.QKeySequence("D"), self,
                        activated=self.annotate_page.export_to_dataset)
        QtGui.QShortcut(QtGui.QKeySequence("+"), self,
                        activated=lambda: self.annotate_page.zoom_step(+1))
        QtGui.QShortcut(QtGui.QKeySequence("-"), self,
                        activated=lambda: self.annotate_page.zoom_step(-1))
        QtGui.QShortcut(QtGui.QKeySequence("0"), self,
                        activated=self.annotate_page.zoom_fit)

        # =============== Init ===============
        self._refresh_projects()
        self.tab_buttons[0].setChecked(True)
        self._switch_tab(0)

    # ---- Menu bar ----

    def _build_menu_bar(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")

        open_video_act = QtGui.QAction("Open Video...", self)
        open_video_act.setShortcut("Ctrl+O")
        open_video_act.triggered.connect(self.annotate_page.open_video)

        open_images_act = QtGui.QAction("Open Image Folder...", self)
        open_images_act.setShortcut("Ctrl+I")
        open_images_act.triggered.connect(self.annotate_page.open_folder)

        open_menu = QtWidgets.QMenu("Open", self)
        open_menu.addAction(open_video_act)
        open_menu.addAction(open_images_act)
        file_menu.addMenu(open_menu)

        file_menu.addSeparator()
        exit_act = QtGui.QAction("Exit", self)
        exit_act.setShortcut("Ctrl+Q")
        exit_act.triggered.connect(self.close)
        file_menu.addAction(exit_act)

        help_menu = menubar.addMenu("&Help")
        about_act = QtGui.QAction("About", self)
        about_act.triggered.connect(self._show_about)
        help_menu.addAction(about_act)

    def _show_about(self):
        QtWidgets.QMessageBox.information(
            self, "About",
            "Annotation & Active Learning Tool\n"
            "YOLO-OBB detection with human-in-the-loop finetuning\n"
            "Built with PySide6",
        )

    # ---- Project management ----

    def _refresh_projects(self):
        self.project_combo.blockSignals(True)
        cur = self.project_combo.currentText()
        self.project_combo.clear()
        projects = self.pm.list_projects()
        self.project_combo.addItems(projects)
        if cur in projects:
            self.project_combo.setCurrentText(cur)
        elif projects:
            self.project_combo.setCurrentIndex(0)
        self.project_combo.blockSignals(False)
        if projects:
            self._on_project_changed(self.project_combo.currentText())

    def _new_project(self):
        name, ok = QtWidgets.QInputDialog.getText(
            self, "New Project", "Project name:",
        )
        if not ok or not name.strip():
            return
        name = name.strip().replace(" ", "_")
        self.pm.create_project(name)
        self._refresh_projects()
        self.project_combo.setCurrentText(name)

    def _on_project_changed(self, name: str):
        if not name:
            return
        self._current_project = name
        self.project_label.setText(f"Project: {name}")
        # Ensure project dirs exist
        self.pm.create_project(name)
        cfg = self.pm.load_config(name)
        self.settings_page.load_config(cfg)
        self.annotate_page.apply_config(cfg)
        self.update_title()

    def project_config(self) -> dict:
        if self._current_project:
            return self.pm.load_config(self._current_project)
        return {}

    def _save_current_config(self):
        if not self._current_project:
            QtWidgets.QMessageBox.warning(self, "Save", "No project selected.")
            return
        cfg = self.settings_page.to_config()
        self.pm.save_config(self._current_project, cfg)
        self.annotate_page.apply_config(cfg)
        self.statusBar().showMessage(f"Settings saved for project '{self._current_project}'.", 4000)

    # ---- Tab switching ----

    def _switch_tab(self, idx: int):
        self.stack.setCurrentIndex(idx)
        # Auto-refresh the inspect page when entering it
        if idx == 2:
            self.inspect_page.refresh()

    # ---- Title ----

    def update_title(self):
        parts = ["Annotation Tool"]
        if self._current_project:
            parts.append(self._current_project)
        ap = self.annotate_page
        if ap.source:
            parts.append(f"{ap.source.name()}")
            parts.append(f"frame {ap.current_idx + 1}/{ap.total_frames}")
        self.setWindowTitle(" | ".join(parts))

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if event.key() == QtCore.Qt.Key.Key_Space:
            self.annotate_page.space_held = True
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QtGui.QKeyEvent):
        if event.key() == QtCore.Qt.Key.Key_Space:
            self.annotate_page.space_held = False
        else:
            super().keyReleaseEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.annotate_page.redraw_current()