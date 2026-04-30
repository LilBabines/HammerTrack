"""
Annotation page — wraps the annotation canvas + controls.

Layout
------
::

    ┌── AnnotatePage ──────────────────────────────────────────┐
    │  ┌──────────────────────────────┐  ┌──── side panel ───┐ │
    │  │                              │  │  Source           │ │
    │  │   AnnotationCanvas           │  │  Inference        │ │
    │  │   (zoom / pan / mouse)       │  │  Annotation       │ │
    │  │                              │  │                   │ │
    │  └──────────────────────────────┘  └───────────────────┘ │
    │  ── frame slider ──                                      │
    │  ── transport bar (prev / play / pause / next / zoom) ── │
    └──────────────────────────────────────────────────────────┘

Sections of this file
---------------------
1. UI construction
2. Project config
3. Source I/O & playback
4. Frame display (overlay drawing → canvas)
5. Model inference (run / cropped run)
6. Fine-tuning
7. Dataset export (frames + YOLO labels)
8. Mouse handlers (connected to canvas signals)
9. Picking
10. Annotation actions (verify / delete / translate / vertex edit)
11. Mode management
12. Add-polygon (OBB: 3 clicks)
13. Zoom forwarding
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ultralytics import YOLO

from ..canvas import AnnotationCanvas
from ..signals import FinetuneSignals
from ..utils import (
    OBBOX, PolyClass, draw_annotations,
    find_orthogonal_projection,
    FrameSource, VideoSource, ImageFolderSource,
)
from ..workers import (
    DetectionWorker,
    DetectFinetuneWorker,
    YOLO_MODEL_PATH,
    resolve_model_path,
)

if TYPE_CHECKING:
    from ..windows import LauncherWindow


class AnnotatePage(QtWidgets.QWidget):
    """Annotation tab: open a source, run a detector, edit/verify boxes,
    export to a YOLO dataset and trigger fine-tuning."""

    # ============================================================
    # 1. UI construction
    # ============================================================

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

        # --- Annotations ---
        self.pred_cache: Dict[int, List[PolyClass]] = {}
        self.class_names: Optional[List[str]] = None
        self.selected_idx: Optional[int] = None

        # --- Verified dataset ---
        self.dataset: Dict[int, List[PolyClass]] = {}
        self.dataset_images_names: Dict[int, str] = {}

        # --- Interaction state ---
        self.mode = "select"
        self.temp_poly_pts: List[List[float]] = []
        self.dragging = False
        self.drag_start_img: Optional[tuple] = None
        self.orig_poly: Optional[np.ndarray] = None
        self.vertex_drag_idx: Optional[int] = None

        self.crop_start_img: Optional[tuple] = None
        self.crop_end_img: Optional[tuple] = None
        self.crop_selecting = False
        self._crop_offset: tuple = (0, 0)

        self.bbox_start_img: Optional[tuple] = None
        self.bbox_end_img: Optional[tuple] = None
        self.bbox_selecting: bool = False

        # --- Task type: "obb" or "detect" (auto-detected from model) ---
        self._task_type: str = "obb"
        self._task_type_cfg: str = "auto"   # from project config

        # --- Model ---
        self.model_worker = DetectionWorker
        self.model_path = YOLO_MODEL_PATH
        self.dataset_dir = ""

        self._build_ui()

    def _build_ui(self):
        # ---- Canvas ----
        self.canvas = AnnotationCanvas()
        self.canvas.mouse_pressed.connect(self._on_canvas_mouse_press)
        self.canvas.mouse_moved.connect(self._on_canvas_mouse_move)
        self.canvas.mouse_released.connect(self._on_canvas_mouse_release)

        # ---- Buttons ----
        self.add_btn = QtWidgets.QPushButton("Add OBB (N)")
        self.add_btn.setToolTip("Add oriented bounding box: 3 clicks")
        self.add_btn.clicked.connect(self.start_add_mode)

        self.add_bbox_btn = QtWidgets.QPushButton("Add BBox (B)")
        self.add_bbox_btn.setToolTip(
            "Add axis-aligned box: click + drag + release"
        )
        self.add_bbox_btn.clicked.connect(self.start_add_bbox_mode)

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

        self.crop_infer_btn = QtWidgets.QPushButton("⬒")
        self.crop_infer_btn.setToolTip("Select a region and run inference on it")
        self.crop_infer_btn.setFixedSize(32, 32)
        self.crop_infer_btn.setCheckable(True)
        self.crop_infer_btn.clicked.connect(self._toggle_crop_infer_mode)

        self.inference_conf_tresh = QtWidgets.QDoubleSpinBox()
        self.inference_conf_tresh.setRange(0.01, 0.99)
        self.inference_conf_tresh.setSingleStep(0.05)
        self.inference_conf_tresh.setValue(0.5)
        self.inference_conf_tresh.setPrefix("conf=")

        # Task type indicator label
        self.task_label = QtWidgets.QLabel("task: —")
        self.task_label.setStyleSheet("color: #888; font-size: 11px;")

        # Slider
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.sliderReleased.connect(self._on_slider_released)

        # Wire up the rest of the buttons
        self.open_video_btn.clicked.connect(self.open_video)
        self.open_images_btn.clicked.connect(self.open_folder)
        self.prev_btn.clicked.connect(self.prev_frame)
        self.next_btn.clicked.connect(self.next_frame)
        self.run_btn.clicked.connect(self.run_model_cached)
        self.export_dataset_btn.clicked.connect(self.export_to_dataset)
        self.play_btn.clicked.connect(self.play)
        self.pause_btn.clicked.connect(self.pause)

        # ---- Layout ----
        left_stack = QtWidgets.QWidget()
        left_v = QtWidgets.QVBoxLayout(left_stack)
        left_v.setContentsMargins(0, 0, 0, 0)
        left_v.setSpacing(6)
        left_v.addWidget(self.canvas, stretch=1)
        left_v.addWidget(self.frame_slider)

        content_row = QtWidgets.QHBoxLayout()
        content_row.setContentsMargins(0, 0, 0, 0)
        content_row.setSpacing(10)
        content_row.addWidget(left_stack, stretch=1)
        content_row.addWidget(self._build_side_panel(), stretch=0)

        page = QtWidgets.QVBoxLayout(self)
        page.setContentsMargins(8, 8, 8, 8)
        page.setSpacing(8)
        page.addLayout(content_row, stretch=1)
        page.addWidget(self._build_transport_bar())

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
        run_row = QtWidgets.QHBoxLayout()
        run_row.addWidget(self.run_btn)
        run_row.addWidget(self.crop_infer_btn)
        infer_l.addLayout(run_row)
        infer_l.addWidget(self.inference_conf_tresh)
        infer_l.addWidget(self.task_label)
        infer_l.addWidget(self.export_dataset_btn)

        # Annotation group
        anno_box = QtWidgets.QGroupBox("Annotation")
        anno_l = QtWidgets.QVBoxLayout(anno_box)
        anno_l.addWidget(self.add_btn)
        anno_l.addWidget(self.add_bbox_btn)
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

    def set_launcher(self, launcher: "LauncherWindow"):
        self._launcher = launcher

    def _status(self, msg: str):
        if self._launcher:
            self._launcher.statusBar().showMessage(msg, 5000)

    # ============================================================
    # 2. Project config
    # ============================================================

    def apply_config(self, cfg: dict):
        self.model_path = cfg.get("model_path", YOLO_MODEL_PATH)
        self.dataset_dir = cfg.get("dataset_dir", "")
        names = cfg.get("class_names", ["object"])
        self.class_names = names if isinstance(names, list) else [names]
        self.inference_conf_tresh.setValue(cfg.get("conf_threshold", 0.5))
        self._task_type_cfg = cfg.get("task_type", "auto")
        self._update_task_label()

    def _effective_task(self) -> str:
        """Return 'obb' or 'detect' depending on config + auto-detection."""
        if self._task_type_cfg in ("obb", "detect"):
            return self._task_type_cfg
        return self._task_type   # auto-detected

    def _update_task_label(self):
        self.task_label.setText(f"task: {self._effective_task()}")

    def _auto_detect_task(self):
        """Read the model task from the DetectionWorker class cache."""
        model_task = getattr(DetectionWorker, "_model_task", None)
        if model_task:
            self._task_type = model_task   # "obb" or "detect"
            self._update_task_label()

    # ============================================================
    # 3. Source I/O & playback
    # ============================================================

    def open_video(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Video", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)",
        )
        if path:
            self.load_video(path)

    def open_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Open Image Folder", ""
        )
        if folder:
            self.load_folder(folder)

    def load_video(self, path: str):
        try:
            src = VideoSource(path)
        except Exception as e:
            self._status(f"Failed to open video: {e}")
            return
        self._set_source(src)

    def load_folder(self, folder: str):
        try:
            src = ImageFolderSource(folder)
        except Exception as e:
            self._status(f"Failed to open folder: {e}")
            return
        self._set_source(src)

    def _set_source(self, src: FrameSource):
        if self.source:
            try:
                self.source.close()
            except Exception:
                pass

        self.pred_cache.clear()
        self.dataset.clear()
        self.dataset_images_names.clear()
        self.selected_idx = None
        self.mode = "select"
        self.temp_poly_pts.clear()
        self.bbox_start_img = None
        self.bbox_end_img = None
        self.bbox_selecting = False

        self.source = src
        self.total_frames = src.count()
        self.src_path = getattr(src, "path", None)
        self.current_idx = 0
        self.frame_slider.setRange(0, max(0, self.total_frames - 1))
        self.frame_slider.setValue(0)
        self._status(
            f"Loaded: {src.name()} | frames={self.total_frames} | "
            f"fps={src.fps():.2f}"
        )
        self.read_frame(self.current_idx)

    def read_frame(self, idx: int) -> bool:
        if not self.source:
            return False
        idx = max(0, min(idx, self.total_frames - 1))
        frame = self.source.read(idx)
        if frame is None:
            self._status("Failed to read frame.")
            return False
        self.current_idx = idx
        self.current_frame_bgr = frame
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(idx)
        self.frame_slider.blockSignals(False)
        if self._launcher:
            self._launcher.update_title()
        self.redraw_current()
        return True

    def prev_frame(self):
        if not self.source:
            return
        self.pause()
        self.read_frame(self.current_idx - 1)

    def next_frame(self):
        if not self.source:
            return
        self.pause()
        self.read_frame(self.current_idx + 1)

    def _on_slider_released(self):
        if not self.source:
            return
        self.pause()
        self.read_frame(self.frame_slider.value())

    def play(self):
        if not self.source or self.playing:
            return
        fps = self.source.fps() or 25
        self.play_timer.start(max(15, int(1000 / fps)))
        self.playing = True

    def pause(self):
        if self.playing:
            self.play_timer.stop()
            self.playing = False

    def _on_play_tick(self):
        if self.current_idx + 1 >= self.total_frames:
            self.pause()
            return
        self.read_frame(self.current_idx + 1)

    # ============================================================
    # 4. Frame display (overlay drawing → canvas)
    # ============================================================

    def redraw_current(self):
        """Render annotations + ghost shapes onto the BGR and push to canvas."""
        if self.current_frame_bgr is None:
            return

        annots = self.pred_cache.get(self.current_idx, [])
        annotated = draw_annotations(
            self.current_frame_bgr, annots,
            self.inference_conf_tresh.value(),
            self.class_names, self.selected_idx,
            show_conf=False, show_label=False,
        )

        # Ghost polygon for OBB add mode
        if self.mode == "add" and self.temp_poly_pts:
            ghost = np.array(self.temp_poly_pts, dtype=np.int32)
            cv2.polylines(
                annotated, [ghost], isClosed=False,
                color=(200, 200, 200), thickness=1, lineType=cv2.LINE_AA,
            )
            for (gx, gy) in ghost:
                cv2.circle(
                    annotated, (int(gx), int(gy)), 3,
                    (200, 200, 200), -1, lineType=cv2.LINE_AA,
                )

        # Ghost rectangle for BBOX add mode
        if (self.mode == "add_bbox"
                and self.bbox_start_img and self.bbox_end_img):
            sx, sy = self.bbox_start_img
            ex, ey = self.bbox_end_img
            x1, y1 = int(min(sx, ex)), int(min(sy, ey))
            x2, y2 = int(max(sx, ex)), int(max(sy, ey))
            cv2.rectangle(
                annotated, (x1, y1), (x2, y2),
                (200, 200, 200), 2, cv2.LINE_AA,
            )

        # Crop-inference selection rectangle (dim outside, highlight inside)
        if (self.mode == "crop_infer"
                and self.crop_start_img and self.crop_end_img):
            sx, sy = self.crop_start_img
            ex, ey = self.crop_end_img
            x1, y1 = int(min(sx, ex)), int(min(sy, ey))
            x2, y2 = int(max(sx, ex)), int(max(sy, ey))
            overlay = annotated.copy()
            cv2.rectangle(
                overlay, (0, 0),
                (annotated.shape[1], annotated.shape[0]),
                (0, 0, 0), -1,
            )
            mask = np.zeros(annotated.shape[:2], dtype=np.uint8)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            annotated = np.where(
                mask[..., None] == 255, annotated,
                cv2.addWeighted(annotated, 0.3, overlay, 0.7, 0),
            )
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 255), 2)

        self.canvas.set_frame(annotated)

    # ============================================================
    # 5. Model inference
    # ============================================================

    def run_model_cached(self):
        idx = self.current_idx
        if self.current_frame_bgr is None:
            return

        self.run_btn.setEnabled(False)
        self.run_btn.setText("Inference running...")
        conf = float(self.inference_conf_tresh.value())

        source_path = None
        if isinstance(self.source, ImageFolderSource):
            source_path = self.source.path_at(idx)

        self.worker_thread = QtCore.QThread(self)
        self.worker = self.model_worker(
            idx, self.current_frame_bgr,
            conf=conf, imgsz=1024,
            model_path=resolve_model_path(
                self.model_path, self._effective_task()
            ),
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

    def _on_inference_done(self, frame_idx: int, class_names,
                           annots: List[PolyClass]):
        self.class_names = class_names
        self.pred_cache[frame_idx] = annots
        self.selected_idx = None
        if frame_idx == self.current_idx:
            self.redraw_current()
        self.run_btn.setEnabled(True)
        self.run_btn.setText("Run Model")
        self._status(f"Predictions cached for frame {frame_idx + 1}.")
        self._auto_detect_task()

    def _on_inference_error(self, msg: str):
        self.run_btn.setEnabled(True)
        self.run_btn.setText("Run Model")
        self._status(f"Inference error: {msg}")

    # ---- Cropped inference ----

    def _toggle_crop_infer_mode(self, checked: bool):
        if checked:
            self.set_mode("crop_infer")
            self.crop_start_img = None
            self.crop_end_img = None
            self.crop_selecting = False
            self._status(
                "Crop inference: drag a rectangle on the image, "
                "release to run."
            )
        else:
            self._cancel_crop_infer()

    def _cancel_crop_infer(self):
        self.crop_infer_btn.setChecked(False)
        self.crop_selecting = False
        self.crop_start_img = None
        self.crop_end_img = None
        self.set_mode("select")
        self.redraw_current()

    def _run_cropped_inference(self, x1: int, y1: int, x2: int, y2: int):
        if self.current_frame_bgr is None:
            return

        h_img, w_img = self.current_frame_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)
        if x2 - x1 < 10 or y2 - y1 < 10:
            self._status("Selection too small, ignored.")
            self._cancel_crop_infer()
            return

        crop = self.current_frame_bgr[y1:y2, x1:x2].copy()
        offset = (x1, y1)

        self.run_btn.setEnabled(False)
        self.run_btn.setText("Crop inference...")
        conf = float(self.inference_conf_tresh.value())
        cfg = self._launcher.project_config() if self._launcher else {}

        self._crop_offset = (x1, y1)

        self._crop_thread = QtCore.QThread(self)
        self._crop_worker = self.model_worker(
            frame_idx=self.current_idx,
            frame_bgr=crop,
            conf=conf,
            imgsz=cfg.get("imgsz", 1024),
            model_path=resolve_model_path(
                self.model_path, self._effective_task()
            ),
            source_path=None,
        )
        self._crop_worker.moveToThread(self._crop_thread)
        self._crop_thread.started.connect(self._crop_worker.run)
        self._crop_worker.finished.connect(self._on_cropped_done)
        self._crop_worker.error.connect(self._on_cropped_error)
        self._crop_worker.finished.connect(self._crop_thread.quit)
        self._crop_worker.error.connect(self._crop_thread.quit)
        self._crop_thread.finished.connect(self._crop_worker.deleteLater)
        self._crop_thread.finished.connect(self._crop_thread.deleteLater)
        self._crop_thread.start()

    def _on_cropped_done(self, frame_idx: int, class_names, annots):
        ox, oy = self._crop_offset
        for box in annots:
            box.poly[:, 0] += ox
            box.poly[:, 1] += oy

        self.class_names = class_names
        existing = self.pred_cache.get(frame_idx, [])
        existing.extend(annots)
        self.pred_cache[frame_idx] = existing

        self.selected_idx = None
        if frame_idx == self.current_idx:
            self.redraw_current()

        self.run_btn.setEnabled(True)
        self.run_btn.setText("Run Model")
        n = len(annots)
        self._status(
            f"Crop inference: {n} detection{'s' if n != 1 else ''} added."
        )
        self._cancel_crop_infer()
        self._auto_detect_task()

    def _on_cropped_error(self, msg: str):
        self.run_btn.setEnabled(True)
        self.run_btn.setText("Run Model")
        self._status(f"Crop inference error: {msg}")
        self._cancel_crop_infer()

    # ============================================================
    # 6. Fine-tuning
    # ============================================================

    def finetune_model(self):
        if not self.src_path:
            QtWidgets.QMessageBox.warning(
                self.window(), "Fine-tune", "Load a source first."
            )
            return
        if not self.dataset:
            QtWidgets.QMessageBox.warning(
                self.window(), "Fine-tune",
                "No verified annotations to train on.",
            )
            return

        n_new = self._export_verified_to_dataset(val_split=0.1)
        data_yaml = self._ensure_data_yaml()
        self._status(f"Exported {n_new} new images to {self.dataset_dir}")

        ds = Path(self.dataset_dir)
        n_train = sum(1 for _ in (ds / "images" / "train").glob("*"))
        if n_train == 0:
            QtWidgets.QMessageBox.warning(
                self.window(), "Fine-tune",
                "0 training images after export.",
            )
            return

        cfg = self._launcher.project_config() if self._launcher else {}

        worker = DetectFinetuneWorker(
            class_names=self.class_names,
            base_model_path=resolve_model_path(
                self.model_path, self._effective_task()
            ),
            out_root=cfg.get(
                "finetune_dir",
                os.path.join(os.getcwd(), "finetune_runs"),
            ),
            epochs=cfg.get("epochs", 20),
            imgsz=cfg.get("imgsz", 1024),
            batch=cfg.get("batch", 16),
            val_split=cfg.get("val_split", 0.1),
            data_yaml=data_yaml,
        )

        bridge = FinetuneSignals(self)
        bridge.progress.connect(
            lambda msg, p: self._status(f"{msg} ({int(p * 100)}%)")
        )
        bridge.error.connect(self._on_finetune_error)
        bridge.finished.connect(self._on_finetune_done)
        self._finetune_bridge = bridge

        train_page = (getattr(self._launcher, "train_page", None)
                      if self._launcher else None)
        if train_page:
            train_page.reset_for_new_run()
            train_page.train_btn.setEnabled(False)
            bridge.progress.connect(train_page.set_progress)

        # Run the fine-tune worker in a real QThread (not a raw Python
        # thread) so any QObject / QTimer used internally by the worker
        # or by ultralytics belongs to a thread Qt knows about.
        thread = QtCore.QThread(self)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.progress.connect(bridge.progress)
        worker.error.connect(bridge.error)
        worker.finished.connect(bridge.finished)
        if train_page:
            worker.epoch_metrics.connect(
                lambda ep, tot, m: train_page.update_metrics(ep, tot, m)
            )
            worker.log_line.connect(train_page.log)

        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        self._finetune_worker = worker
        self._finetune_thread = thread
        thread.start()

    def _on_finetune_error(self, msg: str):
        QtWidgets.QMessageBox.critical(self.window(), "Fine-tune Error", msg)
        train_page = (getattr(self._launcher, "train_page", None)
                      if self._launcher else None)
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
        train_page = (getattr(self._launcher, "train_page", None)
                      if self._launcher else None)
        if train_page:
            train_page.train_btn.setEnabled(True)
            train_page.log(f"Training complete — weights: {best_pt_path}")
            train_page.set_progress("Complete!", 1.0)

    # ============================================================
    # 7. Dataset export (frames + YOLO labels)
    # ============================================================

    def _get_frame_image(self, frame_idx: int) -> Optional[np.ndarray]:
        if self.source is None:
            return None
        return self.source.read(frame_idx)

    @staticmethod
    def _is_axis_aligned(poly: np.ndarray, tol: float = 2.0) -> bool:
        """Check if a 4-point polygon is (approximately) axis-aligned."""
        pts = poly.reshape(4, 2)
        for i in range(4):
            dx = abs(pts[(i + 1) % 4, 0] - pts[i, 0])
            dy = abs(pts[(i + 1) % 4, 1] - pts[i, 1])
            if not (dx < tol or dy < tol):
                return False
        return True

    def _poly_to_yolo_line(self, box: PolyClass,
                           img_w: int, img_h: int) -> str:
        """Convert a box to a YOLO label line.

        * detect mode → ``cls cx cy w h``  (normalized)
        * obb    mode → ``cls x1 y1 x2 y2 x3 y3 x4 y4`` (normalized)
        """
        task = self._effective_task()
        pts = box.poly.reshape(4, 2)

        if task == "detect" or (task == "obb" and self._is_axis_aligned(pts)):
            if task == "detect":
                xs, ys = pts[:, 0], pts[:, 1]
                x1, x2 = float(xs.min()), float(xs.max())
                y1, y2 = float(ys.min()), float(ys.max())
                cx = (x1 + x2) / 2.0 / img_w
                cy = (y1 + y2) / 2.0 / img_h
                bw = (x2 - x1) / img_w
                bh = (y2 - y1) / img_h
                return (
                    f"{int(box.cls_id)} "
                    f"{cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
                )

        # OBB format
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
            if not boxes:
                continue
            src_name = self.source.name() if self.source else "src"
            stem = f"{Path(src_name).stem}_frame{frame_idx:06d}"
            if stem in existing_stems:
                continue

            img = self._get_frame_image(frame_idx)
            if img is None:
                continue
            h, w = img.shape[:2]
            split = "val" if random.random() < val_split else "train"

            cv2.imwrite(str(ds / "images" / split / f"{stem}.jpg"), img)
            lines = [self._poly_to_yolo_line(b, w, h) for b in boxes]
            (ds / "labels" / split / f"{stem}.txt").write_text(
                "\n".join(lines) + "\n"
            )
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
            QtWidgets.QMessageBox.warning(
                self.window(), "Export", "Load a source first."
            )
            return
        if not self.dataset:
            QtWidgets.QMessageBox.warning(
                self.window(), "Export",
                "No verified annotations to export.",
            )
            return

        n_new = self._export_verified_to_dataset(val_split=0.1)
        self._ensure_data_yaml()

        ds = Path(self.dataset_dir)
        n_train = sum(1 for _ in (ds / "images" / "train").glob("*"))
        n_val = sum(1 for _ in (ds / "images" / "val").glob("*"))

        task = self._effective_task()
        self._status(
            f"Exported {n_new} new images ({task} format) → "
            f"{self.dataset_dir}  "
            f"(total: {n_train} train + {n_val} val)"
        )
        QtWidgets.QMessageBox.information(
            self.window(), "Export done",
            f"{n_new} new images exported ({task} format) to:\n"
            f"{os.path.abspath(self.dataset_dir)}\n\n"
            f"Dataset totals: {n_train} train / {n_val} val",
        )

    # ============================================================
    # 8. Mouse handlers (canvas signals → mode-aware behaviour)
    # ============================================================

    def _on_canvas_mouse_press(self, event: QtGui.QMouseEvent,
                               x_img: float, y_img: float):
        # Make sure spinbox loses focus when interacting with the canvas
        if self.inference_conf_tresh.hasFocus():
            self.inference_conf_tresh.clearFocus()

        # Crop-infer: start selection
        if self.mode == "crop_infer":
            if event.button() == QtCore.Qt.MouseButton.LeftButton:
                self.crop_start_img = (x_img, y_img)
                self.crop_end_img = (x_img, y_img)
                self.crop_selecting = True
            return

        # BBox-add: start rectangle
        if self.mode == "add_bbox":
            if event.button() == QtCore.Qt.MouseButton.LeftButton:
                self.bbox_start_img = (x_img, y_img)
                self.bbox_end_img = (x_img, y_img)
                self.bbox_selecting = True
            return

        # Default: left = select / add-vertex / start-drag, right = verify
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            if self.mode == "add":
                self.add_click_point(x_img, y_img)
                return

            hit_idx = self.pick_annot(x_img, y_img)
            if hit_idx is not None:
                self.selected_idx = hit_idx
                self.redraw_current()

                if (self.mode == "edit"
                        and (event.modifiers()
                             & QtCore.Qt.KeyboardModifier.ControlModifier)):
                    v = self.pick_vertex(x_img, y_img)
                    if v is not None:
                        self.vertex_drag_idx = v
                        self.dragging = True
                        return

                boxes = self.pred_cache.get(self.current_idx, [])
                if (self.selected_idx is not None
                        and self.selected_idx < len(boxes)):
                    self.dragging = True
                    self.drag_start_img = (x_img, y_img)
                    self.orig_poly = boxes[self.selected_idx].poly.copy()
            else:
                if self.mode != "add":
                    self.selected_idx = None
                    self.redraw_current()

        elif event.button() == QtCore.Qt.MouseButton.RightButton:
            hit_idx = self.pick_annot(x_img, y_img)
            if hit_idx is not None:
                self.selected_idx = hit_idx
                self.verify_selected_toggle()
                self.redraw_current()

    def _on_canvas_mouse_move(self, event: QtGui.QMouseEvent,
                              x_img: float, y_img: float):
        # Crop-infer: update preview rectangle
        if self.mode == "crop_infer" and self.crop_selecting:
            self.crop_end_img = (x_img, y_img)
            self.redraw_current()
            return

        # BBox-add: update preview rectangle
        if self.mode == "add_bbox" and self.bbox_selecting:
            self.bbox_end_img = (x_img, y_img)
            self.redraw_current()
            return

        # Drag selected box / vertex
        if self.dragging:
            if self.vertex_drag_idx is not None:
                self._set_vertex_selected(self.vertex_drag_idx, x_img, y_img)
            elif self.drag_start_img is not None:
                dx = x_img - self.drag_start_img[0]
                dy = y_img - self.drag_start_img[1]
                self._translate_selected(dx, dy)
            self.redraw_current()

    def _on_canvas_mouse_release(self, event: QtGui.QMouseEvent,
                                 x_img: float, y_img: float):
        # Crop-infer: finalize → run cropped inference
        if (self.mode == "crop_infer" and self.crop_selecting
                and event.button() == QtCore.Qt.MouseButton.LeftButton):
            self.crop_end_img = (x_img, y_img)
            self.crop_selecting = False
            sx, sy = self.crop_start_img
            ex, ey = self.crop_end_img
            x1, x2 = int(min(sx, ex)), int(max(sx, ex))
            y1, y2 = int(min(sy, ey)), int(max(sy, ey))
            self.crop_start_img = None
            self.crop_end_img = None
            self._run_cropped_inference(x1, y1, x2, y2)
            return

        # BBox-add: finalize → create new box
        if (self.mode == "add_bbox" and self.bbox_selecting
                and event.button() == QtCore.Qt.MouseButton.LeftButton):
            self.bbox_end_img = (x_img, y_img)
            self.bbox_selecting = False
            sx, sy = self.bbox_start_img
            ex, ey = self.bbox_end_img
            x1, y1 = min(sx, ex), min(sy, ey)
            x2, y2 = max(sx, ex), max(sy, ey)

            if abs(x2 - x1) > 3 and abs(y2 - y1) > 3:
                pts = np.array(
                    [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                    dtype=np.float32,
                )
                new_box = OBBOX(poly=pts, cls_id=0, conf=1.0, verified=False)
                self.pred_cache.setdefault(self.current_idx, []).append(new_box)
                self.selected_idx = len(self.pred_cache[self.current_idx]) - 1
                self.update_dataset_for_frame(self.current_idx)
                self._status("BBox added.")
            else:
                self._status("Box too small, ignored.")

            self.bbox_start_img = None
            self.bbox_end_img = None
            self.set_mode("select")
            self.redraw_current()
            return

        # End drag
        if self.dragging:
            self.dragging = False
            self.vertex_drag_idx = None
            self.drag_start_img = None
            self.orig_poly = None
            self.update_dataset_for_frame(self.current_idx)
            self.redraw_current()

    # ============================================================
    # 9. Picking
    # ============================================================

    def pick_annot(self, x: float, y: float) -> Optional[int]:
        annots = self.pred_cache.get(self.current_idx, [])
        if not annots:
            return None
        best, best_area = None, None
        for i, b in enumerate(annots):
            if b.deleted:
                continue
            pts = b.poly.reshape(-1, 2).astype(np.float32)
            if cv2.pointPolygonTest(pts, (x, y), measureDist=False) >= 0:
                area = cv2.contourArea(pts.astype(np.int32))
                if best is None or area < best_area:
                    best, best_area = i, area
        return best

    def pick_vertex(self, x: float, y: float,
                    tol_px: int = 10) -> Optional[int]:
        if self.selected_idx is None:
            return None
        annots = self.pred_cache.get(self.current_idx, [])
        if self.selected_idx >= len(annots):
            return None
        a = annots[self.selected_idx]
        if a.deleted:
            return None
        pts = a.poly.reshape(-1, 2)
        for i in range(pts.shape[0]):
            if np.hypot(pts[i, 0] - x, pts[i, 1] - y) <= tol_px:
                return i
        return None

    # ============================================================
    # 10. Annotation actions
    # ============================================================

    def verify_selected_toggle(self):
        boxes = self.pred_cache.get(self.current_idx, [])
        if self.selected_idx is None or self.selected_idx >= len(boxes):
            return
        box = boxes[self.selected_idx]
        if box.deleted:
            return
        box.verified = not box.verified
        self.update_dataset_for_frame(self.current_idx)
        state = "verified" if box.verified else "unverified"
        self._status(f"Box #{self.selected_idx} → {state}")
        self.selected_idx = None
        self.redraw_current()

    def delete_selected(self):
        boxes = self.pred_cache.get(self.current_idx, [])
        if self.selected_idx is None or self.selected_idx >= len(boxes):
            return
        box = boxes[self.selected_idx]
        box.deleted = True
        box.verified = False
        self.update_dataset_for_frame(self.current_idx)
        self._status(f"Box #{self.selected_idx} deleted.")
        self.selected_idx = None
        self.redraw_current()

    def update_dataset_for_frame(self, frame_idx: int):
        all_boxes = self.pred_cache.get(frame_idx, [])
        self.dataset[frame_idx] = [
            b for b in all_boxes if b.verified and not b.deleted
        ]
        if isinstance(self.source, ImageFolderSource):
            self.dataset_images_names[frame_idx] = self.source.path_at(frame_idx)

    def _translate_selected(self, dx: float, dy: float):
        annots = self.pred_cache.get(self.current_idx, [])
        if self.selected_idx is None or self.selected_idx >= len(annots):
            return
        b = annots[self.selected_idx]
        b.poly = (
            self.orig_poly + np.array([dx, dy], dtype=np.float32)
        ).astype(np.float32)
        self.update_dataset_for_frame(self.current_idx)

    def _set_vertex_selected(self, idx: int, x: float, y: float):
        annots = self.pred_cache.get(self.current_idx, [])
        if self.selected_idx is None or self.selected_idx >= len(annots):
            return
        b = annots[self.selected_idx]
        p = b.poly.copy()
        p[idx] = [x, y]
        b.poly = p.astype(np.float32)
        self.update_dataset_for_frame(self.current_idx)

    # ============================================================
    # 11. Mode management
    # ============================================================

    def set_mode(self, mode: str):
        self.mode = mode
        if mode != "add":
            self.temp_poly_pts.clear()
        if mode != "add_bbox":
            self.bbox_start_img = None
            self.bbox_end_img = None
            self.bbox_selecting = False
        self._status(f"Mode: {mode}")

    def start_add_mode(self):
        self.set_mode("add")
        self.selected_idx = None
        self.redraw_current()

    def start_add_bbox_mode(self):
        self.set_mode("add_bbox")
        self.selected_idx = None
        self._status("BBox mode: click and drag to draw a rectangle.")
        self.redraw_current()

    def cancel_add_mode(self):
        if self.mode == "add":
            self.temp_poly_pts.clear()
            self.set_mode("select")
            self.redraw_current()
        elif self.mode == "add_bbox":
            self.bbox_start_img = None
            self.bbox_end_img = None
            self.bbox_selecting = False
            self.set_mode("select")
            self.redraw_current()
        elif self.mode == "crop_infer":
            self._cancel_crop_infer()

    def toggle_edit_mode(self):
        self.set_mode("edit" if self.mode != "edit" else "select")

    # ============================================================
    # 12. Add-polygon (OBB: 3 clicks)
    # ============================================================

    def add_click_point(self, x: float, y: float):
        if len(self.temp_poly_pts) == 2:
            primes = find_orthogonal_projection(
                self.temp_poly_pts[0], self.temp_poly_pts[1], [x, y],
            )
            pts = np.concatenate(
                (self.temp_poly_pts, primes), axis=0, dtype=np.float32,
            )
            new_box = OBBOX(poly=pts, cls_id=0, conf=1.0, verified=False)
            self.pred_cache.setdefault(self.current_idx, []).append(new_box)
            self.selected_idx = len(self.pred_cache[self.current_idx]) - 1
            self.temp_poly_pts.clear()
            self.set_mode("select")
            self.update_dataset_for_frame(self.current_idx)
        else:
            self.temp_poly_pts.append([x, y])
        self.redraw_current()

    # ============================================================
    # 13. Zoom & space-key forwarding (to canvas)
    # ============================================================

    def zoom_step(self, direction: int, anchor_disp=None):
        self.canvas.zoom_step(direction, anchor_disp)

    def zoom_fit(self):
        self.canvas.zoom_fit()

    def set_space_held(self, held: bool):
        """Forward the spacebar state to the canvas (used for pan-with-space)."""
        self.canvas.set_space_held(held)