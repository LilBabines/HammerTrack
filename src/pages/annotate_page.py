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

import copy
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ultralytics import YOLO

from ..annotated_slider import AnnotatedFrameSlider
from ..canvas import AnnotationCanvas
from ..signals import FinetuneSignals
from ..tasks import (
    TASK_DETECT, TASK_OBB, TASK_POSE,
    KPT_DIMS, POSE_BBOX_AUTO, POSE_BBOX_MANUAL,
    DEFAULT_TRAIN_OVERRIDES,
    default_flip_idx, default_model_for, normalize_task,
    validate_pretrained,
)
from ..utils import (
    OBBOX, PolyClass, draw_annotations, draw_keypoints,
    find_orthogonal_projection, keypoints_to_bbox_poly,
    parse_yolo_label_line, rect_to_poly_xyxy,
    FrameSource, VideoSource, ImageFolderSource,
    ORIGIN_DATASET, ORIGIN_MANUAL, ORIGIN_MODEL
)
from ..workers import (
    DetectionWorker,
    DetectFinetuneWorker,
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
        # Wall-clock pacing so playback keeps real time even when a frame
        # takes longer than its slot; _play_busy drops overlapping ticks.
        self._play_fps = 25.0
        self._play_anchor_time = 0.0
        self._play_anchor_idx = 0
        self._play_busy = False

        # --- Annotations ---
        self.pred_cache: Dict[int, List[PolyClass]] = {}
        self.class_names: Optional[List[str]] = None
        self.selected_idx: Optional[int] = None

        # --- Verified dataset ---
        self.dataset: Dict[int, List[PolyClass]] = {}
        self.dataset_images_names: Dict[int, str] = {}
        # Frames edited since the last export. A frame already on disk is
        # normally skipped on export; without this set, correcting a reloaded
        # annotation would never reach the dataset.
        self._dirty_frames: set = set()

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

        # --- Pose annotation state ---
        # Keypoints clicked so far for the instance being created, and the
        # box that goes with them when pose_bbox_mode == "manual".
        self.temp_kpts: List[List[float]] = []
        self.pending_pose_poly: Optional[np.ndarray] = None
        self.kpt_drag_idx: Optional[int] = None
        self.orig_keypoints: Optional[np.ndarray] = None

        # --- Task (fixed by the project, never auto-detected) ---
        self._task_type: str = TASK_OBB
        self.num_keypoints: int = 5
        self.keypoint_names: List[str] = []
        self.flip_idx: List[int] = default_flip_idx(5)
        self.pose_bbox_mode: str = POSE_BBOX_AUTO

        # --- Model ---
        self.model_worker = DetectionWorker
        self.model_path = ""
        self.dataset_dir = ""
        # Archive of full-size annotated frames: the ground truth. The dataset
        # is a derived, disposable training artifact (crops, split); this is
        # not, so it is what gets reloaded when a project is reopened.
        self.annot_dir = ""
        # Dataset-export knobs, overridden per project by apply_config().
        self.crop_zoom_levels = 2
        self.crop_group_margin = self.CROP_GROUP_MARGIN
        self.crop_min_scale_step = self.CROP_MIN_SCALE_STEP
        self.crop_edge_pad = self.CROP_EDGE_PAD
        self.export_jpeg_quality = 98

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

        self.add_pose_btn = QtWidgets.QPushButton("Add Pose (K)")
        self.add_pose_btn.clicked.connect(self.start_add_pose_mode)

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

        # Slider — highlights frames that carry annotations
        self.frame_slider = AnnotatedFrameSlider(
            QtCore.Qt.Orientation.Horizontal
        )
        self.frame_slider.setRange(0, 0)
        self.frame_slider.sliderReleased.connect(self._on_slider_released)

        # Jump between annotated frames
        self.prev_annot_btn = QtWidgets.QPushButton("\u25c0 Annot (Shift+\u2190)")
        self.prev_annot_btn.setToolTip(
            "Jump to the previous frame holding annotations"
        )
        self.prev_annot_btn.clicked.connect(self.prev_annotated_frame)

        self.next_annot_btn = QtWidgets.QPushButton("Annot (Shift+\u2192) \u25b6")
        self.next_annot_btn.setToolTip(
            "Jump to the next frame holding annotations"
        )
        self.next_annot_btn.clicked.connect(self.next_annotated_frame)

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
        h.addWidget(self.prev_annot_btn)
        h.addWidget(self.prev_btn)
        h.addWidget(self.play_btn)
        h.addWidget(self.pause_btn)
        h.addWidget(self.next_btn)
        h.addWidget(self.next_annot_btn)
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
        anno_l.addWidget(self.add_pose_btn)
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
        self.model_path = cfg.get("model_path", "")
        self.dataset_dir = cfg.get("dataset_dir", "")
        self.annot_dir = cfg.get("annotations_dir", "") or (
            str(Path(self.dataset_dir).parent / "annotations")
            if self.dataset_dir else ""
        )
        names = cfg.get("class_names", ["object"])
        self.class_names = names if isinstance(names, list) else [names]
        self.inference_conf_tresh.setValue(cfg.get("conf_threshold", 0.5))

        # The task is owned by the project and is immutable.
        self._task_type = normalize_task(cfg.get("task_type"))
        self.num_keypoints = int(cfg.get("num_keypoints", 5))
        self.keypoint_names = cfg.get("keypoint_names") or []
        self.flip_idx = (cfg.get("flip_idx")
                         or default_flip_idx(self.num_keypoints))
        self.pose_bbox_mode = cfg.get("pose_bbox_mode", POSE_BBOX_AUTO)

        self.crop_zoom_levels = int(cfg.get("crop_zoom_levels", 2))
        self.crop_group_margin = float(
            cfg.get("crop_group_margin", self.CROP_GROUP_MARGIN))
        self.crop_min_scale_step = float(
            cfg.get("crop_min_scale_step", self.CROP_MIN_SCALE_STEP))
        self.crop_edge_pad = float(
            cfg.get("crop_edge_pad", self.CROP_EDGE_PAD))
        self.export_jpeg_quality = int(cfg.get("export_jpeg_quality", 98))

        # A mode from a previous project may not exist under the new task.
        self.set_mode("select")
        self._update_task_ui()
        self.redraw_current()

    def _effective_task(self) -> str:
        """Return the project task: 'detect', 'obb' or 'pose'."""
        return self._task_type

    def _update_task_ui(self):
        """Show only the annotation tools that make sense for this task."""
        task = self._effective_task()
        self.task_label.setText(f"task: {task}")

        # OBB drawing is pointless in detect mode (the label would be squashed
        # back to an axis-aligned box) and in pose mode.
        self.add_btn.setVisible(task == TASK_OBB)
        # An axis-aligned box is the annotation unit for detect, and a valid
        # degenerate OBB, but for pose the box comes from the pose tool.
        self.add_bbox_btn.setVisible(task in (TASK_DETECT, TASK_OBB))
        self.add_pose_btn.setVisible(task == TASK_POSE)

        if task == TASK_POSE:
            hint = ("Place %d keypoints" % self.num_keypoints
                    if self.pose_bbox_mode == POSE_BBOX_AUTO
                    else "Drag a box, then place %d keypoints"
                         % self.num_keypoints)
            self.add_pose_btn.setToolTip(hint)

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

    def _clear_annotation_state(self):
        """Drop every annotation and interaction state.

        Shared by :meth:`_set_source` and :meth:`unload_project`: annotations
        are tied to both a source and a project, so either changing means the
        whole cache has to go.
        """
        self.pred_cache.clear()
        self.dataset.clear()
        self.dataset_images_names.clear()
        self._dirty_frames.clear()
        self.selected_idx = None
        self.mode = "select"
        self.temp_poly_pts.clear()
        self.bbox_start_img = None
        self.bbox_end_img = None
        self.bbox_selecting = False
        self.temp_kpts.clear()
        self.pending_pose_poly = None
        self.crop_start_img = None
        self.crop_end_img = None
        self.crop_selecting = False
        self.dragging = False
        self.drag_start_img = None
        self.orig_poly = None
        self.orig_keypoints = None
        self.vertex_drag_idx = None
        self.kpt_drag_idx = None
        self.frame_slider.set_marked_frames([])
        if self.crop_infer_btn.isChecked():
            self.crop_infer_btn.setChecked(False)

    def _close_source(self):
        if self.source:
            try:
                self.source.close()
            except Exception:
                pass
        self.source = None

    def unload_project(self):
        """Release everything tied to the current project.

        Called when the project selection changes: the loaded video, the
        cached predictions and the verified dataset all belong to the project
        that was open, and must not bleed into the next one.
        """
        self.pause()
        self._close_source()
        self._clear_annotation_state()

        self.total_frames = 0
        self.current_idx = 0
        self.current_frame_bgr = None
        self.src_path = None

        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.setValue(0)
        self.frame_slider.blockSignals(False)

        self.canvas.set_frame(None)
        self._refresh_slider_marks()

        # The weights cache is a class attribute shared with the other pages:
        # the next project may use a different task and different weights.
        DetectionWorker.clear_model_cache()

    def is_busy(self) -> bool:
        """True while a fine-tune run owned by this page is still going."""
        thread = getattr(self, "_finetune_thread", None)
        return bool(thread is not None and thread.isRunning())

    def _set_source(self, src: FrameSource):
        self._close_source()
        self._clear_annotation_state()

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

        # Bring back whatever was already exported for this source, before the
        # first frame is drawn so annotations appear immediately.
        self.load_annotations_from_dataset()

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

    def prev_annotated_frame(self):
        """Jump to the closest annotated frame before the current one."""
        target = self.frame_slider.prev_marked(self.current_idx)
        if target is None:
            self._status("No annotated frame before this one.")
            return
        self.pause()
        self.read_frame(target)
        self._status(f"Annotated frame {target + 1}/{self.total_frames}")

    def next_annotated_frame(self):
        """Jump to the closest annotated frame after the current one."""
        target = self.frame_slider.next_marked(self.current_idx)
        if target is None:
            self._status("No annotated frame after this one.")
            return
        self.pause()
        self.read_frame(target)
        self._status(f"Annotated frame {target + 1}/{self.total_frames}")

    def _on_slider_released(self):
        if not self.source:
            return
        self.pause()
        self.read_frame(self.frame_slider.value())

    def play(self):
        if not self.source or self.playing:
            return
        self._play_fps = max(1.0, float(self.source.fps() or 25.0))
        self._play_anchor_time = time.monotonic()
        self._play_anchor_idx = self.current_idx
        self._play_busy = False
        self.playing = True
        # Nearest-neighbour resampling while playing; quality is restored on
        # pause, when the frame is actually being looked at.
        self.canvas.set_fast_scaling(True)
        self.play_timer.start(max(5, int(1000 / self._play_fps)))

    def pause(self):
        if self.playing:
            self.play_timer.stop()
            self.playing = False
            self.canvas.set_fast_scaling(False)
            self.redraw_current()

    def _on_play_tick(self):
        """Advance playback, dropping frames rather than falling behind.

        A tick firing while the previous frame is still being decoded would
        otherwise queue up and starve the event loop, freezing the UI.
        """
        if self._play_busy:
            return
        self._play_busy = True
        try:
            elapsed = time.monotonic() - self._play_anchor_time
            target = self._play_anchor_idx + int(elapsed * self._play_fps)
            # Never go backwards, and always make progress.
            target = max(self.current_idx + 1, target)

            if target >= self.total_frames:
                self.pause()
                if self.current_idx != self.total_frames - 1:
                    self.read_frame(self.total_frames - 1)
                return
            self.read_frame(target)
        finally:
            self._play_busy = False

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

        # Edit-mode handles on the selected annotation, so it is visible what
        # can be grabbed: keypoints get a ring, box corners get squares (and
        # only when the box is actually editable).
        if self.mode == "edit" and self.selected_idx is not None:
            sel_list = self.pred_cache.get(self.current_idx, [])
            if self.selected_idx < len(sel_list):
                sel = sel_list[self.selected_idx]
                if not sel.deleted:
                    if sel.has_keypoints():
                        for (kx, ky) in sel.keypoints.reshape(-1, 2):
                            cv2.circle(
                                annotated, (int(kx), int(ky)), 9,
                                (255, 255, 255), 1, cv2.LINE_AA,
                            )
                    if self._box_is_editable()[0]:
                        for (vx, vy) in sel.poly.reshape(-1, 2):
                            cv2.rectangle(
                                annotated,
                                (int(vx) - 4, int(vy) - 4),
                                (int(vx) + 4, int(vy) + 4),
                                (255, 255, 255), 1, cv2.LINE_AA,
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

        # Ghost keypoints for the pose instance being placed
        if self.mode == "add_pose" and self.temp_kpts:
            draw_keypoints(
                annotated, np.array(self.temp_kpts, dtype=np.float32),
                color=(200, 200, 200), show_index=True,
            )
            # In auto-bbox mode, preview the box the keypoints would produce.
            if (self.pending_pose_poly is None
                    and len(self.temp_kpts) >= 2):
                h_img, w_img = annotated.shape[:2]
                preview = keypoints_to_bbox_poly(
                    np.array(self.temp_kpts, dtype=np.float32),
                    img_w=w_img, img_h=h_img,
                ).astype(np.int32)
                cv2.polylines(
                    annotated, [preview], isClosed=True,
                    color=(160, 160, 160), thickness=1, lineType=cv2.LINE_AA,
                )

        # Locked-in manual box, shown while its keypoints are being placed
        if self.mode == "add_pose" and self.pending_pose_poly is not None:
            cv2.polylines(
                annotated, [self.pending_pose_poly.astype(np.int32)],
                isClosed=True, color=(200, 200, 200), thickness=2,
                lineType=cv2.LINE_AA,
            )

        # Ghost rectangle for BBOX / pose-box add modes
        if (self.mode in ("add_bbox", "add_pose_box")
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

        # Use the project's image size, like the cropped path already does.
        cfg = self._launcher.project_config() if self._launcher else {}
        cfg_imgsz = cfg.get("imgsz", 1024)

        self.worker_thread = QtCore.QThread(self)
        self.worker = self.model_worker(
            idx, self.current_frame_bgr,
            conf=conf, imgsz=cfg_imgsz,
            model_path=resolve_model_path(
                self.model_path, self._effective_task()
            ),
            source_path=source_path,
            task=self._effective_task(),
            two_stage=bool(cfg.get("two_stage", False)),
            region_conf=float(cfg.get("region_conf", 0.10)),
            max_regions=int(cfg.get("max_regions", 8)),
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
        kept = [b for b in self.pred_cache.get(frame_idx, [])
                if b.is_ground_truth() or b.verified]
        self.pred_cache[frame_idx] = kept + list(annots)
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
            task=self._effective_task(),
            # Never two-stage here, whatever the project setting says: the user
            # has already framed the region by hand. A proposal pass would spend
            # a forward pass rediscovering a region that was just handed to it,
            # then crop *inside* it — losing part of the area deliberately
            # selected, and re-detecting at a scale the user did not ask for.
            two_stage=False,
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

        # The worker ran on a crop, so every coordinate it returned — box
        # corners AND pose keypoints — is expressed relative to the crop
        # origin. Both have to come back to image space through the same
        # translation, or the boxes land correctly while the keypoints stay
        # bunched near the top-left of the full frame.
        annots = [self._translate_annot(box, ox, oy) for box in annots]

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
        msg = f"Region inference: {n} detection{'s' if n != 1 else ''} added."

        # On a pose project a box without keypoints cannot be exported
        # (_poly_to_yolo_line rejects it), so say so now rather than letting
        # the annotation be verified and then silently dropped.
        if self._effective_task() == TASK_POSE:
            n_bare = sum(1 for b in annots if not b.has_keypoints())
            if n_bare:
                msg += (f" {n_bare} without keypoints — place them by hand or "
                        f"they will not be exported.")
        self._status(msg)
        self._cancel_crop_infer()

    def _on_cropped_error(self, msg: str):
        self.run_btn.setEnabled(True)
        self.run_btn.setText("Run Model")
        self._status(f"Crop inference error: {msg}")
        self._cancel_crop_infer()

    # ============================================================
    # 6. Fine-tuning
    # ============================================================

    def dataset_counts(self) -> Dict[str, int]:
        """Count labelled images already on disk, per split.

        Training depends on what the dataset directory holds, not on what was
        annotated in this session: a dataset exported earlier is perfectly
        trainable after a restart. Images without a matching, non-empty label
        file are not counted, since they would act as false negatives.
        """
        counts = {"train": 0, "val": 0, "train_unlabelled": 0, "val_unlabelled": 0}
        if not self.dataset_dir:
            return counts

        ds = Path(self.dataset_dir)
        for split in ("train", "val"):
            img_dir = ds / "images" / split
            lbl_dir = ds / "labels" / split
            if not img_dir.is_dir():
                continue
            for img in img_dir.iterdir():
                if not img.is_file():
                    continue
                lbl = lbl_dir / f"{img.stem}.txt"
                try:
                    labelled = lbl.is_file() and lbl.stat().st_size > 0
                except OSError:
                    labelled = False
                counts[split if labelled else f"{split}_unlabelled"] += 1
        return counts

    def finetune_model(self):
        if not self.dataset_dir:
            QtWidgets.QMessageBox.warning(
                self.window(), "Fine-tune",
                "No dataset directory configured for this project.",
            )
            return

        # Push any annotation verified during this session, then judge the
        # dataset by what is actually on disk. A source does not have to be
        # loaded: an already-exported dataset is trainable on its own.
        n_new = 0
        if self.src_path and (any(self.dataset.values()) or self._dirty_frames):
            cfg_pre = self._launcher.project_config() if self._launcher else {}
            n_new = self._export_verified_to_dataset(
                val_split=cfg_pre.get("val_split", 0.1),
                imgsz=int(cfg_pre.get("imgsz", 1024)),
                multiscale=bool(cfg_pre.get("multiscale_export", True)),
                val_type=str(cfg_pre.get("val_type", "end")),
            )
            self._status(f"Exported {n_new} new images to {self.dataset_dir}")

        data_yaml = self._ensure_data_yaml()

        counts = self.dataset_counts()
        if counts["train"] == 0:
            skipped = counts["train_unlabelled"] + counts["val_unlabelled"]
            detail = (
                f"\n\n{skipped} image(s) were found without a matching label "
                f"file and were ignored."
                if skipped else
                "\n\nVerify some annotations and export them to the dataset "
                "(button 'Export to Dataset', shortcut D)."
            )
            QtWidgets.QMessageBox.warning(
                self.window(), "Fine-tune",
                f"No labelled training image in:\n{self.dataset_dir}{detail}",
            )
            return

        self._status(
            f"Training on {counts['train']} train / {counts['val']} val "
            f"labelled images"
            + (f" (+{n_new} new)" if n_new else "")
        )

        cfg = self._launcher.project_config() if self._launcher else {}
        task = self._effective_task()

        # Always restart from the official pretrained backbone, never from
        # self.model_path: after a first run that points at custom best.pt
        # weights, and stacking fine-tunes compounds drift across iterations.
        base_model = cfg.get("default_model") or default_model_for(task)
        ok, reason = validate_pretrained(base_model, task)
        if not ok:
            QtWidgets.QMessageBox.critical(
                self.window(), "Fine-tune \u2014 invalid base model", reason
            )
            return

        worker = DetectFinetuneWorker(
            # Same fallback as _ensure_data_yaml, so the yaml and the worker
            # can never disagree on the class list.
            class_names=self.class_names or ["object"],
            base_model=base_model,
            data_yaml=data_yaml,
            task=task,
            out_root=cfg.get(
                "finetune_dir",
                os.path.join(os.getcwd(), "finetune_runs"),
            ),
            epochs=cfg.get("epochs", 20),
            imgsz=cfg.get("imgsz", 1024),
            batch=cfg.get("batch", 16),
            # Augmentation and runtime knobs, forwarded verbatim to
            # model.train(). The split itself is already materialised on disk
            # by the export, so val_split is not passed any more.
            overrides=self._build_train_overrides(cfg),
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
        """Adopt the freshly trained weights as the current inference model."""
        self._status(f"Fine-tune complete: {best_pt_path}")
        try:
            # Load the new weights and keep the cache key in sync, otherwise
            # _get_model() sees a stale path and reloads them from disk.
            self.model_worker._model = YOLO(best_pt_path)
            self.model_worker._model_path = best_pt_path
            self.model_worker._model_task = self._effective_task()
            self.model_path = best_pt_path
            self._persist_model_path(best_pt_path)
            self._status(f"Inference now uses fine-tuned weights: {best_pt_path}")
        except Exception as e:
            self._status(f"Model saved but failed to load: {e}")
        train_page = (getattr(self._launcher, "train_page", None)
                      if self._launcher else None)
        if train_page:
            train_page.train_btn.setEnabled(True)
            train_page.log(f"Training complete — weights: {best_pt_path}")
            train_page.set_progress("Complete!", 1.0)

    def _persist_model_path(self, weights: str):
        """Save the new inference weights into the project config.

        Without this the fine-tuned model is forgotten as soon as the project
        is reopened, silently falling back to the pretrained checkpoint.
        """
        launcher = self._launcher
        if launcher is None or not getattr(launcher, "_current_project", None):
            return
        try:
            cfg = launcher.project_config()
            cfg["model_path"] = weights
            launcher.pm.save_config(launcher._current_project, cfg)
        except Exception as e:
            self._status(f"Could not save model path to config: {e}")

    # ============================================================
    # 7. Dataset export (frames + YOLO labels)
    # ============================================================

    def _get_frame_image(self, frame_idx: int) -> Optional[np.ndarray]:
        if self.source is None:
            return None
        return self.source.read(frame_idx)

    @staticmethod
    def _bbox_cxcywh(pts: np.ndarray, img_w: int, img_h: int) -> str:
        """Normalized ``cx cy w h`` of the axis-aligned hull of ``pts``."""
        xs, ys = pts[:, 0], pts[:, 1]
        x1, x2 = float(xs.min()), float(xs.max())
        y1, y2 = float(ys.min()), float(ys.max())
        cx = (x1 + x2) / 2.0 / img_w
        cy = (y1 + y2) / 2.0 / img_h
        bw = (x2 - x1) / img_w
        bh = (y2 - y1) / img_h
        return f"{cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"

    def _poly_to_yolo_line(self, box: PolyClass,
                           img_w: int, img_h: int) -> Optional[str]:
        """Convert an annotation to a YOLO label line for the project task.

        * detect → ``cls cx cy w h``                       (normalized)
        * obb    → ``cls x1 y1 x2 y2 x3 y3 x4 y4``         (normalized)
        * pose   → ``cls cx cy w h x1 y1 ... xN yN``       (normalized)

        Returns ``None`` when the annotation cannot be written for this task
        (e.g. a pose instance with the wrong number of keypoints), so the
        caller can skip it rather than emit a malformed label.
        """
        task = self._effective_task()
        pts = box.poly.reshape(-1, 2)

        if task == TASK_DETECT:
            return f"{int(box.cls_id)} {self._bbox_cxcywh(pts, img_w, img_h)}"

        if task == TASK_POSE:
            if not box.has_keypoints():
                return None
            kpts = box.keypoints.reshape(-1, KPT_DIMS)
            # Ultralytics requires a fixed keypoint count across the dataset.
            if len(kpts) != self.num_keypoints:
                return None
            parts = [str(int(box.cls_id)),
                     self._bbox_cxcywh(pts, img_w, img_h)]
            for x, y in kpts:
                parts.append(f"{x / img_w:.6f} {y / img_h:.6f}")
            return " ".join(parts)

        # OBB format
        parts = [str(int(box.cls_id))]
        for x, y in pts:
            parts.append(f"{x / img_w:.6f}")
            parts.append(f"{y / img_h:.6f}")
        return " ".join(parts)

    # ------------------------------------------------------------------
    # Reloading a previously exported dataset
    # ------------------------------------------------------------------

    #: Full-frame export stem: "<source stem>_frame<6 digits>". Multi-scale
    #: crops add a "_z<n>" suffix and are deliberately excluded — their
    #: coordinates live in crop space, and they duplicate the full view.
    _EXPORT_STEM_RE = re.compile(r"^(?P<src>.+)_frame(?P<idx>\d{6})$")

    def _dataset_label_files(self) -> Dict[int, Path]:
        found: Dict[int, Path] = {}
        if self.source is None:
            return found
        prefix = f"{self.source.stem()}_"

        def collect(lbl_dir: Path, into: Dict[int, Path]):
            if not lbl_dir.is_dir():
                return
            for path in lbl_dir.glob("*.txt"):
                if not path.stem.startswith(prefix):
                    continue
                # The source resolves its own convention. A multi-scale crop
                # ("..._z2") resolves to nothing and is skipped, which is what
                # we want: its coordinates live in crop space.
                idx = self.source.index_for_key(path.stem[len(prefix):])
                if idx is not None and 0 <= idx < max(1, self.total_frames):
                    into.setdefault(idx, path)

        if self.annot_dir:
            collect(Path(self.annot_dir) / "labels", found)

        if self.dataset_dir:
            legacy: Dict[int, Path] = {}
            for split in ("train", "val"):
                collect(Path(self.dataset_dir) / "labels" / split, legacy)
            for idx, path in legacy.items():
                found.setdefault(idx, path)

        return found

    def load_annotations_from_dataset(self) -> int:
        """Repopulate annotations for the loaded source from the dataset.

        Called right after a source is opened, so reopening the app and
        reloading a video brings back everything already exported. Labels are
        the source of truth: nothing is stored twice, and a frame deleted from
        the dataset by hand simply stops coming back.

        Returns the number of frames restored.
        """
        label_files = self._dataset_label_files()
        if not label_files:
            return 0

        task = self._effective_task()
        restored = 0
        skipped = 0

        for idx in sorted(label_files):
            size = self.source.frame_size(idx)
            if not size:
                skipped += 1
                continue
            img_w, img_h = size

            try:
                text = label_files[idx].read_text(encoding="utf-8")
            except OSError:
                skipped += 1
                continue

            boxes: List[PolyClass] = []
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                box = parse_yolo_label_line(
                    line, task, img_w, img_h, self.num_keypoints
                )
                if box is not None:
                    boxes.append(box)

            if not boxes:
                skipped += 1
                continue

            # pred_cache and dataset must share the SAME objects: the canvas
            # draws from pred_cache, while update_dataset_for_frame() rebuilds
            # dataset from it. Filling only one of the two would either show
            # nothing or silently drop the frame on the first edit.
            self.pred_cache[idx] = boxes
            self.dataset[idx] = boxes
            if isinstance(self.source, ImageFolderSource):
                self.dataset_images_names[idx] = self.source.path_at(idx)
            restored += 1

        self._refresh_slider_marks()
        if restored:
            origin = self.annot_dir or self.dataset_dir
            msg = f"Restored {restored} annotated frame(s) from {origin}"
            if skipped:
                msg += f" ({skipped} label file(s) unreadable, skipped)"
            self._status(msg)
        return restored

    # ------------------------------------------------------------------
    # Multi-scale crop export (SAHI-like)
    # ------------------------------------------------------------------

    #: Breathing room around the instance group in the tightest crop, as a
    #: fraction of the group size on each side.
    CROP_GROUP_MARGIN = 0.15
    #: Minimum scale ratio between two consecutive zoom levels. Below this,
    #: two crops are near-duplicates and only cost disk and epoch time.
    CROP_MIN_SCALE_STEP = 1.3
    #: Keep every annotation at least this many pixels from the crop border,
    #: so no instance is ever flush against the edge.
    CROP_EDGE_PAD = 6.0

    @staticmethod
    def _annot_extent(box: PolyClass) -> tuple:
        """Axis-aligned extent ``(x1, y1, x2, y2)`` of a polygon + keypoints.

        Keypoints are included because in ``auto`` bbox mode the box is
        derived from them and can be clamped to the image, which would let a
        keypoint sit marginally outside the polygon.
        """
        parts = [box.poly.reshape(-1, 2)]
        if box.has_keypoints():
            parts.append(box.keypoints.reshape(-1, 2))
        pts = np.concatenate(parts, axis=0).astype(np.float32)
        return (float(pts[:, 0].min()), float(pts[:, 1].min()),
                float(pts[:, 0].max()), float(pts[:, 1].max()))

    def _group_extent(self, boxes: List[PolyClass]) -> Optional[tuple]:
        """Union extent of every annotation on a frame, or None if empty."""
        ext = [self._annot_extent(b) for b in boxes if not b.deleted]
        if not ext:
            return None
        return (min(e[0] for e in ext), min(e[1] for e in ext),
                max(e[2] for e in ext), max(e[3] for e in ext))

    @staticmethod
    def _window_origin(g1: float, g2: float, side: int,
                       limit: int, pad: float, rng: random.Random) -> int:
        """Random origin on one axis keeping ``[g1, g2]`` inside the window.

        Drawing uniformly from the whole valid interval decentres the group by
        construction: a window always centred on the instances would teach the
        model that the subject lives in the middle of the frame.
        """
        lo = max(0.0, g2 + pad - side)
        hi = min(g1 - pad, float(limit - side))
        if lo > hi:
            # No slack (group nearly as large as the window): centre + clamp.
            centred = (g1 + g2) / 2.0 - side / 2.0
            return int(round(max(0.0, min(centred, float(limit - side)))))
        return int(round(rng.uniform(lo, hi)))

    def _crop_windows(self, group: tuple, img_w: int, img_h: int,
                      imgsz: int, rng: random.Random) -> List[tuple]:
        """Square crop windows around the instance group, coarse then tight.

        Returns a list of ``(x0, y0, w, h)`` in source pixels.

        Every window is **at least ``imgsz`` wide**. A smaller crop would be
        interpolated back up to ``imgsz`` by the training pipeline, which adds
        blur rather than resolution: the whole point is that the letterbox
        step only ever downscales. So a 500x500 group does not give a 500x500
        crop — it gives an ``imgsz``-wide crop centred loosely on the group,
        which is native pixels all the way through.
        """
        gx1, gy1, gx2, gy2 = group
        pad = self.crop_edge_pad
        step = max(1.01, self.crop_min_scale_step)
        levels = max(0, int(self.crop_zoom_levels))
        if levels == 0:
            return []

        group_side = max(gx2 - gx1, gy2 - gy1)
        needed = group_side * (1.0 + 2 * self.crop_group_margin) + 2 * pad

        side_max = float(min(img_w, img_h))       # largest square in the frame
        side_tight = max(float(imgsz), needed)

        # Nothing to gain: the group already fills the frame, or imgsz is as
        # large as the image, so the crop would duplicate the full view.
        if side_tight >= side_max / step:
            return []

        # How many levels actually fit between the tightest crop and the full
        # frame at the requested separation. Clamping here rather than
        # filtering afterwards matters: a greedy filter can discard the
        # tightest level, which is the native-resolution one and the whole
        # point of the exercise.
        span = side_max / side_tight
        max_levels = int(np.floor(np.log(span) / np.log(step)))
        levels = max(1, min(levels, max(1, max_levels)))

        # Geometric ladder: a constant ratio between levels, so every zoom
        # adds the same amount of relative detail. k=0 is side_tight, so the
        # native-resolution crop is always produced. With levels=2 this is
        # exactly the previous "tight + log midpoint" behaviour.
        ratio = span ** (1.0 / levels)
        sides = sorted(
            (side_tight * (ratio ** k) for k in range(levels)), reverse=True
        )

        windows = []
        for side in sides:
            s = int(round(min(side, side_max)))
            x0 = self._window_origin(gx1, gx2, s, img_w, pad, rng)
            y0 = self._window_origin(gy1, gy2, s, img_h, pad, rng)
            windows.append((x0, y0, s, s))
        return windows

    @staticmethod
    def _translate_annot(box: PolyClass, dx: float, dy: float) -> PolyClass:
        """Copy of ``box`` translated by ``(dx, dy)``, keypoints included.

        Used in both directions: image space → crop space when exporting a
        crop (negative offsets), and crop space → image space after a region
        inference (positive offsets). Having a single primitive is deliberate —
        it replaces a hand-written shift that moved ``poly`` and silently left
        ``keypoints`` behind, which is exactly the class of bug that cannot be
        allowed to recur once pose annotations exist.

        ``copy.copy`` keeps the concrete subclass (``OBBOX`` and its
        ``track_id``); only the coordinate arrays are replaced, so the original
        annotation is never mutated.
        """
        out = copy.copy(box)
        off = np.array([dx, dy], dtype=np.float32)
        out.poly = box.poly.reshape(-1, 2).astype(np.float32) + off
        if box.has_keypoints():
            out.keypoints = (box.keypoints.reshape(-1, 2).astype(np.float32)
                             + off)
        return out

    @staticmethod
    def _frame_view_paths(ds: Path, base_stem: str, crops_only: bool = False):
        """Yield ``(split, path)`` for every file belonging to one frame.

        Both splits are swept because the split is drawn at random: a frame
        rewritten into the other one would leave an orphan image behind,
        counted as an unlabelled false negative.

        The patterns are exact rather than ``base_stem*``, which would also
        match ``..._frame1000123`` while working on ``..._frame000123``.
        """
        patterns = ((f"{base_stem}_z*",) if crops_only
                    else (f"{base_stem}.*", f"{base_stem}_z*"))
        for split in ("train", "val"):
            for kind in ("images", "labels"):
                for pattern in patterns:
                    for path in (ds / kind / split).glob(pattern):
                        yield split, path

    @classmethod
    def _purge_frame_views(cls, ds: Path, base_stem: str,
                           crops_only: bool = False) -> Optional[str]:
        """Delete a frame's views and return the split they were in."""
        found_split = None
        # Materialized: the generator walks directories being unlinked.
        for split, path in list(cls._frame_view_paths(ds, base_stem,
                                                     crops_only)):
            found_split = found_split or split
            path.unlink(missing_ok=True)
        return found_split

    # ------------------------------------------------------------------
    # Annotation archive (full-size, unsplit, no augmentation)
    # ------------------------------------------------------------------

    def _export_full_frames_to_annotations(self) -> int:
        """Mirror every annotated frame into ``annotations/images|labels``.

        This folder is the ground truth: one full-size image and one label per
        annotated frame, flat, with no train/val split and no crops. The
        dataset folder next to it is a *derived* training artifact — crops,
        augmentation-friendly duplicates, a split that can be recomputed — so
        it must be safe to delete and rebuild. Keeping the truth somewhere
        else is what makes that safe.

        Incremental: a frame already archived and unedited is skipped without
        being decoded.
        """
        if not self.annot_dir or self.source is None:
            return 0

        root = Path(self.annot_dir)
        (root / "images").mkdir(parents=True, exist_ok=True)
        (root / "labels").mkdir(parents=True, exist_ok=True)

        written = 0
        removed = 0
        for frame_idx, boxes in self.dataset.items():
            stem = self._frame_stem(frame_idx)
            img_path = root / "images" / f"{stem}.jpg"
            lbl_path = root / "labels" / f"{stem}.txt"

            live = [b for b in boxes if not b.deleted]

            if not live:
                # Emptying a frame IS an edit. Skipping it here would leave
                # the previous archive in place, and since the archive is what
                # load_annotations_from_dataset() reads back, the deleted
                # annotations would silently reappear on the next open.
                if img_path.exists() or lbl_path.exists():
                    img_path.unlink(missing_ok=True)
                    lbl_path.unlink(missing_ok=True)
                    removed += 1
                continue

            if (img_path.exists() and lbl_path.exists()
                    and frame_idx not in self._dirty_frames):
                continue

            img = self._get_frame_image(frame_idx)
            if img is None:
                continue
            h, w = img.shape[:2]

            lines = [ln for ln in
                     (self._poly_to_yolo_line(b, w, h) for b in live) if ln]
            if not lines:
                # Every annotation is invalid for this task (e.g. a pose with
                # the wrong keypoint count). Archiving the image with an empty
                # label would record a false negative — and a previously valid
                # archive must not survive either.
                if img_path.exists() or lbl_path.exists():
                    img_path.unlink(missing_ok=True)
                    lbl_path.unlink(missing_ok=True)
                    removed += 1
                continue

            cv2.imwrite(str(img_path), img,
                        [int(cv2.IMWRITE_JPEG_QUALITY),
                         self.export_jpeg_quality])
            lbl_path.write_text("\n".join(lines) + "\n")
            written += 1

        return written, removed

    #: Optional ultralytics ``train()`` knobs stored in the project config.
    #: Anything absent is simply not forwarded, so it keeps the ultralytics
    #: default instead of being pinned to a value duplicated in this file.
    _TRAIN_OVERRIDE_KEYS = (
        "device", "patience", "workers",
        "degrees", "translate", "scale", "fliplr", "flipud",
        "mosaic", "close_mosaic",
        "hsv_h", "hsv_s", "hsv_v",
    )

    def _build_train_overrides(self, cfg: dict) -> dict:
        """Collect the ultralytics ``train()`` knobs from the project config.

        Passing them as one dict rather than as a long kwargs list means a new
        setting only has to be added to the panel and to
        ``_TRAIN_OVERRIDE_KEYS`` — the worker signature stops growing.
        """
        # Start from the shared defaults so a project saved before these
        # settings existed still trains the way the panel says it would,
        # instead of silently falling back to the ultralytics defaults.
        overrides = dict(DEFAULT_TRAIN_OVERRIDES)
        for key in self._TRAIN_OVERRIDE_KEYS:
            if key not in cfg:
                continue
            value = cfg[key]
            if key == "device":
                value = str(value).strip()
                if not value:
                    continue          # empty means "let ultralytics choose"
            elif key == "patience":
                if not int(value):
                    continue          # 0 is "off" in the UI
            overrides[key] = value
        return overrides

    def _frame_stem(self, frame_idx: int) -> str:
        """On-disk base name of a frame's full view.

        The naming convention belongs to the source: an index for a video,
        the image file name for a folder — the only key that survives the
        folder gaining or losing a file.
        """
        if self.source is None:
            return f"src_frame{frame_idx:06d}"
        return f"{self.source.stem()}_{self.source.frame_key(frame_idx)}"

    @staticmethod
    def _temporal_val_frames(frames: List[int], val_split: float,
                             position: str = "end") -> set:
        """Pick ONE contiguous run of frames for validation.

        Consecutive video frames are near-duplicates. A per-frame random draw
        therefore puts frame *n* in train and frame *n+1* in val, and the model
        is validated on images it has effectively already trained on: every
        metric looks excellent and none of them means anything. Holding out a
        single temporal block is the only split that measures generalisation on
        this kind of footage.

        The run is taken over the *annotated* frames sorted by index, not over
        the whole video, so the requested proportion is respected even when
        annotations are sparse or clustered — and since the list is sorted, a
        contiguous slice of it is still contiguous in time.
        """
        n = len(frames)
        if n < 2 or val_split <= 0:
            return set()
        n_val = max(1, int(round(n * val_split)))
        n_val = min(n_val, n - 1)          # never leave training empty

        if position == "random":
            # Opt-in only. Cheap to regenerate is not the same as correct:
            # neighbouring frames stay near-duplicates whatever the dataset
            # folder is worth, so this still reports inflated metrics.
            return set(random.Random(f"split:{n}").sample(frames, n_val))

        if position == "start":
            start = 0
        elif position == "middle":
            start = (n - n_val) // 2
        else:                               # "end": conventional holdout
            start = n - n_val
        return set(frames[start:start + n_val])

    @classmethod
    def _relocate_frame_views(cls, ds: Path, base_stem: str,
                              target_split: str) -> int:
        """Move every view of a frame into ``target_split``.

        The split is a property of the whole dataset, so annotating more frames
        legitimately moves the boundary. Files on the wrong side are moved
        rather than left behind, which also silently converts a dataset written
        with the old random split the first time it is re-exported.
        """
        moved = 0
        for split, path in list(cls._frame_view_paths(ds, base_stem)):
            if split == target_split:
                continue
            kind = path.parent.parent.name          # "images" or "labels"
            dest = ds / kind / target_split / path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            path.replace(dest)
            moved += 1
        return moved

    def _export_verified_to_dataset(self, val_split: float = 0.1,
                                    imgsz: int = 1024,
                                    multiscale: bool = True,
                                    val_type: str = "end") -> int:
        ds = Path(self.dataset_dir)
        for split in ("train", "val"):
            (ds / "images" / split).mkdir(parents=True, exist_ok=True)
            (ds / "labels" / split).mkdir(parents=True, exist_ok=True)

        existing_stems = set()
        for split in ("train", "val"):
            for p in (ds / "images" / split).iterdir():
                if p.is_file():
                    existing_stems.add(p.stem)

        # Ground truth first: whatever happens to the dataset below, the
        # full-size annotated frames are safely archived.
        archived, unarchived = self._export_full_frames_to_annotations()
        if archived or unarchived:
            self._status(f"Archive: {archived} frame(s) written, "
                         f"{unarchived} removed → {self.annot_dir}")

        # --- Split, decided once for the whole source ---
        annotated = sorted(idx for idx, bx in self.dataset.items() if bx)
        val_frames = self._temporal_val_frames(annotated, val_split, val_type)

        if val_frames and val_type == "random":
            self._status(
                f"RANDOM split: {len(val_frames)}/{len(annotated)} frames → "
                f"val, scattered. Neighbouring video frames are "
                f"near-duplicates, so val metrics will be optimistic."
            )
        elif val_frames:
            lo, hi = min(val_frames), max(val_frames)
            self._status(
                f"Temporal split: frames {lo + 1}-{hi + 1} → val "
                f"({len(val_frames)}/{len(annotated)} annotated frames), "
                f"one contiguous block."
            )
        elif annotated:
            self._status("Temporal split: not enough frames for a val block; "
                         "everything goes to train.")

        # Reconcile what is already on disk with the split computed above.
        # Cheap: filesystem only, no frame is decoded.
        bases_on_disk = {re.sub(r"_z\d+$", "", st) for st in existing_stems}
        relocated = 0
        for frame_idx in annotated:
            base_stem = self._frame_stem(frame_idx)
            if base_stem in bases_on_disk:
                relocated += self._relocate_frame_views(
                    ds, base_stem,
                    "val" if frame_idx in val_frames else "train",
                )
        if relocated:
            self._status(f"Moved {relocated} file(s) to match the temporal "
                         f"split.")

        exported = 0
        purged = 0
        for frame_idx, boxes in self.dataset.items():
            if not boxes:
                continue
            base_stem = self._frame_stem(frame_idx)

            if self._purge_frame_views(ds, base_stem) is not None:
                purged += 1
                existing_stems -= {
                    st for st in existing_stems
                    if st == base_stem or st.startswith(f"{base_stem}_z")
                }
            # What is already on disk, view by view. A single "already
            # exported" flag is not enough: a dataset written before
            # multi-scale existed has the full view but no crops, and those
            # crops must still be added.
            has_full = base_stem in existing_stems
            crops_on_disk = sum(1 for st in existing_stems
                                if st.startswith(f"{base_stem}_z"))
            is_dirty = frame_idx in self._dirty_frames

            # How many crops this frame should have. Decided from the frame
            # dimensions, which frame_size() reads from metadata — so a frame
            # that needs nothing is never decoded.
            crops_wanted = 0
            if multiscale and self.source is not None:
                size = self.source.frame_size(frame_idx)
                group = self._group_extent(boxes)
                if size and group is not None:
                    crops_wanted = len(self._crop_windows(
                        group, size[0], size[1], imgsz,
                        random.Random(base_stem),
                    ))

            if is_dirty or not has_full:
                todo = "all"          # never exported, or edited since
            elif multiscale and crops_wanted and crops_wanted != crops_on_disk:
                todo = "crops"        # legacy or partial export: top it up
            else:
                continue              # complete and untouched — no decode

            img = self._get_frame_image(frame_idx)
            if img is None:
                continue
            h, w = img.shape[:2]

            if todo == "all":
                self._purge_frame_views(ds, base_stem)
            else:
                # Keep the existing full view untouched; only its crops are
                # rebuilt.
                self._purge_frame_views(ds, base_stem, crops_only=True)

            # ONE split per frame, shared by all its views, and deterministic:
            # a crop and its parent image show the same pixels, so they must
            # never land on opposite sides. The reconciliation pass above has
            # already moved any existing view here, so this always agrees with
            # what is on disk.
            split = "val" if frame_idx in val_frames else "train"

            # (stem, image, offset_x, offset_y, view_w, view_h)
            views = []
            if todo == "all":
                views.append((base_stem, img, 0.0, 0.0, w, h))

            if multiscale:
                group = self._group_extent(boxes)
                if group is not None:
                    # Seeded on the stem: re-exporting a frame reproduces the
                    # same crops instead of quietly growing the dataset with
                    # near-duplicates.
                    rng = random.Random(base_stem)
                    windows = self._crop_windows(group, w, h, imgsz, rng)
                    for i, (cx0, cy0, cw, ch) in enumerate(windows, start=1):
                        crop = np.ascontiguousarray(
                            img[cy0:cy0 + ch, cx0:cx0 + cw]
                        )
                        views.append((f"{base_stem}_z{i}", crop,
                                      float(cx0), float(cy0), cw, ch))

            for stem, view, off_x, off_y, vw, vh in views:
                if view is None or view.size == 0:
                    continue

                lines = []
                for b in boxes:
                    if b.deleted:
                        continue
                    # Image space → crop space: subtract the crop origin.
                    shifted = (b if (off_x == 0.0 and off_y == 0.0)
                               else self._translate_annot(b, -off_x, -off_y))
                    ln = self._poly_to_yolo_line(shifted, vw, vh)
                    if ln:
                        lines.append(ln)

                if not lines:
                    # Every annotation on this view was invalid for the task
                    # (e.g. incomplete pose): writing the image with an empty
                    # label would teach the model a false negative.
                    continue

                cv2.imwrite(
                    str(ds / "images" / split / f"{stem}.jpg"), view,
                    [int(cv2.IMWRITE_JPEG_QUALITY), self.export_jpeg_quality],
                )
                (ds / "labels" / split / f"{stem}.txt").write_text(
                    "\n".join(lines) + "\n"
                )
                exported += 1

            existing_stems.add(base_stem)

        if purged:
            self._status(f"Removed {purged} frame(s) whose annotations were "
                         f"all deleted.")
            
        # Everything pending has been written; the next export skips these
        # frames again unless they are edited anew.
        self._dirty_frames.clear()
        return exported

    def _ensure_data_yaml(self) -> str:
        """Write ``data.yaml`` for the project task and return its path."""
        ds = Path(self.dataset_dir)
        yaml_path = ds / "data.yaml"
        names = self.class_names if self.class_names else ["object"]

        lines = [
            f"path: {ds.resolve()}",
            "train: images/train",
            "val: images/val",
            "",
        ]

        if self._effective_task() == TASK_POSE:
            # kpt_shape is mandatory for pose; the second value is 2 because
            # every annotated keypoint is considered visible (no v flag).
            flip = self.flip_idx or default_flip_idx(self.num_keypoints)
            lines += [
                f"kpt_shape: [{self.num_keypoints}, {KPT_DIMS}]",
                f"flip_idx: {list(flip)}",
                "",
            ]

        lines += [f"nc: {len(names)}", f"names: {list(names)}", ""]
        yaml_path.write_text("\n".join(lines))
        return str(yaml_path)

    def export_to_dataset(self):
        if not self.src_path:
            QtWidgets.QMessageBox.warning(
                self.window(), "Export", "Load a source first."
            )
            return
        if not any(self.dataset.values()) and not self._dirty_frames:
            QtWidgets.QMessageBox.warning(
                self.window(), "Export",
                "Nothing to export: no annotation, and nothing deleted "
                "since the last export.",
            )
            return

        cfg = self._launcher.project_config() if self._launcher else {}
        n_new = self._export_verified_to_dataset(
            val_split=cfg.get("val_split", 0.1),
            imgsz=int(cfg.get("imgsz", 1024)),
            multiscale=bool(cfg.get("multiscale_export", True)),
            val_type=str(cfg.get("val_type", "end")),
        )
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

        # Dragging redraws on every mouse move; high-quality resampling of a
        # 4K frame costs ~23 ms and would make it feel sluggish. Quality is
        # restored on release.
        self.canvas.set_fast_scaling(True)

        # Crop-infer: start selection
        if self.mode == "crop_infer":
            if event.button() == QtCore.Qt.MouseButton.LeftButton:
                self.crop_start_img = (x_img, y_img)
                self.crop_end_img = (x_img, y_img)
                self.crop_selecting = True
            return

        # BBox-add / pose manual box: start rectangle
        if self.mode in ("add_bbox", "add_pose_box"):
            if event.button() == QtCore.Qt.MouseButton.LeftButton:
                self.bbox_start_img = (x_img, y_img)
                self.bbox_end_img = (x_img, y_img)
                self.bbox_selecting = True
            return

        # Pose: each left click drops one keypoint, right click undoes the last
        if self.mode == "add_pose":
            if event.button() == QtCore.Qt.MouseButton.LeftButton:
                self.add_pose_click(x_img, y_img)
            elif (event.button() == QtCore.Qt.MouseButton.RightButton
                    and self.temp_kpts):
                self.temp_kpts.pop()
                self._status(
                    f"Pose: click keypoint {len(self.temp_kpts) + 1}/"
                    f"{self.num_keypoints}."
                )
                self.redraw_current()
            return

        # Default: left = select / add-vertex / start-drag, right = verify
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            if self.mode == "add":
                self.add_click_point(x_img, y_img)
                return

            # --- Edit mode: handles are grabbed before anything else ---
            # No modifier needed, and the search is not gated on clicking
            # inside a polygon: a keypoint outside its box has to stay
            # reachable, and Ctrl-to-edit was undiscoverable.
            if self.mode == "edit":
                tol = self._edit_tolerance()

                # Keypoints win over box corners: they are smaller targets and
                # they sit inside the box, so a shared position must resolve to
                # the keypoint.
                hit = self.pick_keypoint_any(x_img, y_img, tol)
                if hit is not None:
                    self.selected_idx, self.kpt_drag_idx = hit
                    self.dragging = True
                    self.redraw_current()
                    return

                # Then a corner of the already-selected box: reshaping is a
                # deliberate act on a box the user has chosen.
                if self.selected_idx is not None:
                    v = self.pick_vertex(x_img, y_img, tol)
                    if v is not None:
                        editable, why = self._box_is_editable()
                        if editable:
                            self.vertex_drag_idx = v
                            self.dragging = True
                            return
                        self._status(why)

            hit_idx = self.pick_annot(x_img, y_img)
            if hit_idx is not None:
                self.selected_idx = hit_idx
                self.redraw_current()

                boxes = self.pred_cache.get(self.current_idx, [])
                if (self.selected_idx is not None
                        and self.selected_idx < len(boxes)):
                    sel = boxes[self.selected_idx]
                    self.dragging = True
                    self.drag_start_img = (x_img, y_img)
                    self.orig_poly = sel.poly.copy()
                    self.orig_keypoints = (
                        sel.keypoints.copy() if sel.has_keypoints() else None
                    )
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

        # BBox-add / pose manual box: update preview rectangle
        if self.mode in ("add_bbox", "add_pose_box") and self.bbox_selecting:
            self.bbox_end_img = (x_img, y_img)
            self.redraw_current()
            return

        # Drag selected box / vertex
        if self.dragging:
            if self.kpt_drag_idx is not None:
                self._set_keypoint_selected(self.kpt_drag_idx, x_img, y_img)
            elif self.vertex_drag_idx is not None:
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
            self.redraw_current()
            self._run_cropped_inference(x1, y1, x2, y2)
            return

        # Pose manual box: finalize → switch to keypoint placement
        if (self.mode == "add_pose_box" and self.bbox_selecting
                and event.button() == QtCore.Qt.MouseButton.LeftButton):
            self.bbox_end_img = (x_img, y_img)
            self.bbox_selecting = False
            sx, sy = self.bbox_start_img
            ex, ey = self.bbox_end_img
            x1, y1 = min(sx, ex), min(sy, ey)
            x2, y2 = max(sx, ex), max(sy, ey)
            self.bbox_start_img = None
            self.bbox_end_img = None

            if abs(x2 - x1) > 3 and abs(y2 - y1) > 3:
                self.pending_pose_poly = np.array(
                    [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                    dtype=np.float32,
                )
                self.set_mode("add_pose")
                self._status(
                    f"Pose: click keypoint 1/{self.num_keypoints}."
                )
            else:
                self.set_mode("select")
                self._status("Box too small, pose cancelled.")
            self.redraw_current()
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

        # Back to high-quality resampling now the interaction is over.
        if not self.playing:
            self.canvas.set_fast_scaling(False)

        # End drag
        if self.dragging:
            self.dragging = False
            self.vertex_drag_idx = None
            self.kpt_drag_idx = None
            self.drag_start_img = None
            self.orig_poly = None
            self.orig_keypoints = None
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

    #: Grab radius for keypoints and box corners, in SCREEN pixels.
    EDIT_GRAB_PX = 10

    def _edit_tolerance(self) -> float:
        """Grab radius in image pixels, constant on screen whatever the zoom.

        A fixed image-space tolerance is unusable on this data: at zoom 8 on 4K
        footage, 12 image pixels covers a large part of the widget and every
        keypoint of a small shark falls inside it, while zoomed out the same
        radius is smaller than the cursor. Converting from screen space keeps
        the handles feeling the same size and makes keypoints a few pixels
        apart individually selectable once zoomed in.
        """
        scale = self.canvas.display_scale()
        if scale <= 0:
            return float(self.EDIT_GRAB_PX)
        return float(self.EDIT_GRAB_PX) / scale

    def pick_keypoint_any(self, x: float, y: float,
                          tol_px: float = 12.0) -> Optional[Tuple[int, int]]:
        """``(annot_idx, kpt_idx)`` of the nearest keypoint, or None.

        Searches every visible instance rather than only the selected one: a
        keypoint is a small target, and in manual bbox mode it can legitimately
        sit outside its own box, so requiring a prior selection would make some
        keypoints impossible to reach.

        Instances hidden by the confidence threshold are skipped — grabbing
        something that is not drawn would be baffling.
        """
        annots = self.pred_cache.get(self.current_idx, [])
        threshold = self.inference_conf_tresh.value()
        best: Optional[Tuple[int, int]] = None
        best_dist = float(tol_px)
        for i, b in enumerate(annots):
            if b.deleted or b.conf < threshold or not b.has_keypoints():
                continue
            pts = b.keypoints.reshape(-1, 2)
            dists = np.hypot(pts[:, 0] - x, pts[:, 1] - y)
            j = int(np.argmin(dists))
            if float(dists[j]) <= best_dist:
                best_dist = float(dists[j])
                best = (i, j)
        return best

    def _box_is_editable(self) -> Tuple[bool, str]:
        """Whether the selected box may be reshaped by hand, and why not."""
        if (self._effective_task() == TASK_POSE
                and self.pose_bbox_mode == POSE_BBOX_AUTO):
            return False, (
                "Box is derived from the keypoints (pose_bbox_mode=auto): "
                "move a keypoint to reshape it, or set the project to "
                "'manual' to edit the box directly."
            )
        return True, ""

    def pick_keypoint(self, x: float, y: float,
                      tol_px: float = 12.0) -> Optional[int]:
        """Index of the selected instance's keypoint under (x, y), or None."""
        if self.selected_idx is None:
            return None
        annots = self.pred_cache.get(self.current_idx, [])
        if self.selected_idx >= len(annots):
            return None
        a = annots[self.selected_idx]
        if a.deleted or not a.has_keypoints():
            return None

        pts = a.keypoints.reshape(-1, 2)
        dists = np.hypot(pts[:, 0] - x, pts[:, 1] - y)
        nearest = int(np.argmin(dists))
        return nearest if dists[nearest] <= tol_px else None

    def pick_vertex(self, x: float, y: float,
                    tol_px: float = 10.0) -> Optional[int]:
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
        if box.is_ground_truth():
            self._status("Manual annotation: already in the dataset "
                         "(Del to remove it).")
            self.selected_idx = None
            self.redraw_current()
            return

        # A pose box with no keypoints is rejected by _poly_to_yolo_line.
        # Accepting it would put it in self.dataset, mark the frame annotated
        # in the UI, and then drop it silently at export time.
        if (not box.verified
                and self._effective_task() == TASK_POSE
                and not box.has_keypoints()):
            self._status("Cannot accept: pose instance without keypoints. "
                         "Place them by hand first.")
            return

        box.verified = not box.verified
        self.update_dataset_for_frame(self.current_idx)
        self._status(
            f"Detection #{self.selected_idx} "
            f"{'added to' if box.verified else 'removed from'} the dataset."
        )
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

        sel = self.selected_idx
        if sel is not None and sel < len(all_boxes):
            b = all_boxes[sel]
            if not b.deleted and b.origin == ORIGIN_MODEL:
                b.verified = True
                b.origin = ORIGIN_MANUAL

        all_boxes = self.pred_cache.get(frame_idx, [])
        self.dataset[frame_idx] = [
            b for b in all_boxes if b.verified and not b.deleted
        ]
        # Every edit path goes through here, so this is the one place that
        # knows a frame no longer matches what was exported.
        self._dirty_frames.add(frame_idx)
        if isinstance(self.source, ImageFolderSource):
            self.dataset_images_names[frame_idx] = self.source.path_at(frame_idx)
        self._refresh_slider_marks()

    def _refresh_slider_marks(self):
        """Highlight every frame holding at least one verified annotation."""
        marked = [idx for idx, boxes in self.dataset.items() if boxes]
        self.frame_slider.set_marked_frames(marked)
        n = len(marked)
        has_marks = n > 0
        self.prev_annot_btn.setEnabled(has_marks)
        self.next_annot_btn.setEnabled(has_marks)
        self.prev_annot_btn.setToolTip(
            f"Previous annotated frame ({n} in total)"
        )
        self.next_annot_btn.setToolTip(
            f"Next annotated frame ({n} in total)"
        )

    def _translate_selected(self, dx: float, dy: float):
        annots = self.pred_cache.get(self.current_idx, [])
        if self.selected_idx is None or self.selected_idx >= len(annots):
            return
        b = annots[self.selected_idx]
        delta = np.array([dx, dy], dtype=np.float32)
        b.poly = (self.orig_poly + delta).astype(np.float32)
        # Keypoints are rigidly attached to their box.
        if self.orig_keypoints is not None:
            b.keypoints = (self.orig_keypoints + delta).astype(np.float32)
        self.update_dataset_for_frame(self.current_idx)

    def _set_keypoint_selected(self, idx: int, x: float, y: float):
        """Move one keypoint of the selected instance.

        In auto-bbox mode the box is re-derived so it keeps wrapping the
        keypoints; a manually drawn box is left untouched.
        """
        annots = self.pred_cache.get(self.current_idx, [])
        if self.selected_idx is None or self.selected_idx >= len(annots):
            return
        b = annots[self.selected_idx]
        if not b.has_keypoints() or idx >= len(b.keypoints):
            return

        kpts = b.keypoints.copy()
        kpts[idx] = [x, y]
        b.keypoints = kpts.astype(np.float32)

        if (self.pose_bbox_mode == POSE_BBOX_AUTO
                and self.current_frame_bgr is not None):
            h_img, w_img = self.current_frame_bgr.shape[:2]
            b.poly = keypoints_to_bbox_poly(
                b.keypoints, img_w=w_img, img_h=h_img
            )
        self.update_dataset_for_frame(self.current_idx)

    #: Smallest allowed box side when resizing, in image pixels.
    MIN_BOX_SIDE = 4.0

    def _set_vertex_selected(self, idx: int, x: float, y: float):
        """Drag one corner of the selected annotation.

        ``obb`` polygons are free quadrilaterals, so the corner moves alone.
        For ``detect`` and ``pose`` the polygon has to stay an axis-aligned
        rectangle: moving a single corner would produce a quadrilateral whose
        exported label is its bounding hull — a box the user never drew, and
        silently larger than what is on screen. The corner therefore resizes
        the rectangle with the opposite corner pinned, which is what every
        annotation tool does.
        """
        annots = self.pred_cache.get(self.current_idx, [])
        if self.selected_idx is None or self.selected_idx >= len(annots):
            return
        b = annots[self.selected_idx]
        p = b.poly.reshape(-1, 2).astype(np.float32)
        if idx >= len(p):
            return

        if self._effective_task() == TASK_OBB or len(p) != 4:
            p = p.copy()
            p[idx] = [x, y]
            b.poly = p.astype(np.float32)
        else:
            anchor = p[(idx + 2) % 4]          # opposite corner stays put
            ax, ay = float(anchor[0]), float(anchor[1])
            x1, x2 = min(float(x), ax), max(float(x), ax)
            y1, y2 = min(float(y), ay), max(float(y), ay)

            half = self.MIN_BOX_SIDE / 2.0
            if x2 - x1 < self.MIN_BOX_SIDE:
                cx = (x1 + x2) / 2.0
                x1, x2 = cx - half, cx + half
            if y2 - y1 < self.MIN_BOX_SIDE:
                cy = (y1 + y2) / 2.0
                y1, y2 = cy - half, cy + half

            b.poly = rect_to_poly_xyxy(x1, y1, x2, y2)

            # Dragging past the anchor flips the rectangle, so the canonical
            # corner order no longer matches the grabbed index. Re-anchor on
            # the corner now nearest the cursor, otherwise the drag would jump
            # to a different corner mid-gesture.
            corners = b.poly.reshape(-1, 2)
            self.vertex_drag_idx = int(np.argmin(
                np.hypot(corners[:, 0] - x, corners[:, 1] - y)
            ))

        self.update_dataset_for_frame(self.current_idx)

    # ============================================================
    # 11. Mode management
    # ============================================================

    # Modes that make up the pose workflow: "add_pose_box" is only used when
    # pose_bbox_mode == "manual", and always hands over to "add_pose".
    _POSE_MODES = ("add_pose", "add_pose_box")

    def set_mode(self, mode: str):
        self.mode = mode
        if mode != "add":
            self.temp_poly_pts.clear()
        if mode not in ("add_bbox", "add_pose_box"):
            self.bbox_start_img = None
            self.bbox_end_img = None
            self.bbox_selecting = False
        if mode not in self._POSE_MODES:
            self.temp_kpts.clear()
            self.pending_pose_poly = None
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

    def start_add_pose_mode(self):
        """Begin a pose instance.

        With ``pose_bbox_mode == "manual"`` the user drags the box first and
        the keypoints come after; in ``auto`` mode the keypoints are placed
        straight away and the box is derived from them.
        """
        if self._effective_task() != TASK_POSE:
            self._status("Pose annotation requires a pose project.")
            return

        self.selected_idx = None
        self.temp_kpts.clear()
        self.pending_pose_poly = None

        if self.pose_bbox_mode == POSE_BBOX_MANUAL:
            self.set_mode("add_pose_box")
            self._status("Pose: drag the bounding box first.")
        else:
            self.set_mode("add_pose")
            self._status(
                f"Pose: click keypoint 1/{self.num_keypoints}."
            )
        self.redraw_current()

    def cancel_add_mode(self):
        if self.mode == "add":
            self.temp_poly_pts.clear()
            self.set_mode("select")
            self.redraw_current()
        elif self.mode in ("add_bbox", "add_pose_box"):
            self.bbox_start_img = None
            self.bbox_end_img = None
            self.bbox_selecting = False
            self.set_mode("select")
            self.redraw_current()
        elif self.mode == "add_pose":
            self.temp_kpts.clear()
            self.pending_pose_poly = None
            self.set_mode("select")
            self.redraw_current()
        elif self.mode == "crop_infer":
            self._cancel_crop_infer()

    def toggle_edit_mode(self):
        entering = self.mode != "edit"
        self.set_mode("edit" if entering else "select")
        if not entering:
            self.redraw_current()
            return
        if self._effective_task() == TASK_POSE:
            if self.pose_bbox_mode == POSE_BBOX_AUTO:
                self._status("Edit: drag any keypoint — the box follows it.")
            else:
                self._status(
                    "Edit: drag any keypoint, or a box corner to resize."
                )
        else:
            self._status("Edit: drag a corner to reshape, or the body to move.")
        self.redraw_current()

    # ============================================================
    # 12. Add-polygon (OBB: 3 clicks)
    # ============================================================

    def add_pose_click(self, x: float, y: float):
        """Register one keypoint; finalize the instance once N are placed."""
        self.temp_kpts.append([x, y])
        placed = len(self.temp_kpts)

        if placed < self.num_keypoints:
            nxt = placed + 1
            name = ""
            if nxt <= len(self.keypoint_names):
                name = f" ({self.keypoint_names[nxt - 1]})"
            self._status(
                f"Pose: click keypoint {nxt}/{self.num_keypoints}{name}."
            )
            self.redraw_current()
            return

        self._finalize_pose_instance()

    def _finalize_pose_instance(self):
        """Turn the buffered keypoints (+ box) into an annotation."""
        kpts = np.array(self.temp_kpts, dtype=np.float32)

        if self.pending_pose_poly is not None:
            poly = self.pending_pose_poly
        else:
            h_img, w_img = self.current_frame_bgr.shape[:2]
            poly = keypoints_to_bbox_poly(kpts, img_w=w_img, img_h=h_img)

        new_box = OBBOX(
            poly=poly, cls_id=0, conf=1.0, verified=False, keypoints=kpts,
            origin=ORIGIN_MANUAL
        )
        self.pred_cache.setdefault(self.current_idx, []).append(new_box)
        self.selected_idx = len(self.pred_cache[self.current_idx]) - 1

        self.temp_kpts.clear()
        self.pending_pose_poly = None
        self.set_mode("select")
        self.update_dataset_for_frame(self.current_idx)
        self._status(f"Pose instance added ({self.num_keypoints} keypoints).")
        self.redraw_current()

    def add_click_point(self, x: float, y: float):
        if len(self.temp_poly_pts) == 2:
            primes = find_orthogonal_projection(
                self.temp_poly_pts[0], self.temp_poly_pts[1], [x, y],
            )
            pts = np.concatenate(
                (self.temp_poly_pts, primes), axis=0, dtype=np.float32,
            )
            new_box = OBBOX(poly=pts, cls_id=0, conf=1.0, verified=False, origin=ORIGIN_MANUAL)
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