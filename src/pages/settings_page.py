"""Project settings editor — tabbed orchestrator over 4 sub-panels.

The launcher only sees ``SettingsPage``. Internally the page wraps a
``QTabWidget`` containing one panel per concern:

    Project   — model, dataset, classes, task type
    Training  — epochs, image size, batch, val split
    Tracking  — tracker type, ReID, thresholds

Each sub-panel exposes the same minimal interface:

    panel.load_config(cfg)   # read its keys from the dict
    panel.to_config(cfg)     # write its keys back into the dict (in place)

Adding a new section is therefore: build a panel, add it to the
``QTabWidget`` in :meth:`_build_ui`, register it in :meth:`_panels`.
"""

from PySide6 import QtCore, QtWidgets

from ..tasks import (
    TASK_POSE,
    TASK_LABELS,
    default_model_for,
    pretrained_choices,
    POSE_BBOX_MODES,
    POSE_BBOX_AUTO,
    default_flip_idx,
    default_keypoint_names,
    format_flip_idx,
    normalize_task,
    parse_flip_idx,
    DEFAULT_TRAIN_OVERRIDES
)


class SettingsPage(QtWidgets.QWidget):
    """Tabbed settings editor.

    Emits ``config_changed`` when the user clicks "Save settings". The owner
    is responsible for actually persisting the dict returned by
    :meth:`to_config`.
    """

    config_changed = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cfg: dict = {}
        self._build_ui()

    # ==================== UI construction ====================

    def _build_ui(self):
        self.tabs = QtWidgets.QTabWidget()

        self.project_panel   = ProjectSettingsPanel(self)
        self.training_panel  = TrainingSettingsPanel(self)
        self.tracking_panel  = TrackingSettingsPanel(self)

        self.tabs.addTab(self.project_panel,   "Project")
        self.tabs.addTab(self.training_panel,  "Training")
        self.tabs.addTab(self.tracking_panel,  "Tracking")

        self.save_btn = QtWidgets.QPushButton("Save settings")
        self.save_btn.setFixedWidth(160)
        self.save_btn.clicked.connect(self._on_save)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tabs, stretch=1)
        layout.addSpacing(10)
        layout.addWidget(
            self.save_btn, alignment=QtCore.Qt.AlignmentFlag.AlignCenter
        )
        layout.addSpacing(10)

    @property
    def _panels(self):
        return [
            self.project_panel,
            self.training_panel,
            self.tracking_panel,
        ]

    # ==================== Config marshalling ====================

    def load_config(self, cfg: dict):
        self._cfg = cfg
        for panel in self._panels:
            panel.load_config(cfg)

    def to_config(self) -> dict:
        cfg = dict(self._cfg)
        for panel in self._panels:
            panel.to_config(cfg)
        return cfg

    # ==================== Save ====================

    def _on_save(self):
        self.config_changed.emit()
 
class TrainingSettingsPanel(QtWidgets.QWidget):
    """Panel for training hyperparameters and dataset-export options.
 
    Three groups:
 
    * **Training** — ``epochs``, ``imgsz``, ``batch``, ``device``, ``patience``
    * **Validation split** — ``val_split``, ``val_type``
    * **Dataset export** — how annotated frames are turned into a training
      set: ``multiscale_export`` and the crop-zoom geometry.
    * **Augmentation** — the ultralytics ``train()`` knobs that actually
      matter for small animals filmed straight down from a drone. Deliberately
      not exhaustive: ``mixup``, ``shear``, ``copy_paste`` and friends either
      do not apply to pose or hurt on small objects.
 
    Note: ``imgsz`` is also used at inference time — there is intentionally a
    single value shared by both. It additionally sets the *floor* for crop
    size on export, so that a crop is never interpolated back up.
    """
 
    #: (stored value, label shown to the user) for the validation strategy.
    VAL_TYPES = (
        ("end", "Temporal — last block"),
        ("middle", "Temporal — middle block"),
        ("start", "Temporal — first block"),
        ("random", "Random per frame (leaky)"),
    )
 
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
 
    # ==================== UI ====================
 
    def _build_ui(self):
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(20, 20, 20, 20)
        outer.setSpacing(16)
 
        outer.addWidget(self._build_training_group())
        outer.addWidget(self._build_validation_group())
        outer.addWidget(self._build_export_group())
        outer.addWidget(self._build_augmentation_group())
        outer.addStretch(1)
 
    def _build_training_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Training")
        form = QtWidgets.QFormLayout(box)
        form.setSpacing(10)
 
        self.epochs_spin = QtWidgets.QSpinBox()
        self.epochs_spin.setRange(1, 500)
 
        self.imgsz_spin = QtWidgets.QSpinBox()
        self.imgsz_spin.setRange(128, 4096)
        self.imgsz_spin.setSingleStep(64)
        self.imgsz_spin.setToolTip(
            "Model image size — used at training and inference, and as the\n"
            "minimum crop size on export so crops are never upscaled.\n"
            "Must be a multiple of 32; memory scales with the square."
        )
 
        self.batch_spin = QtWidgets.QSpinBox()
        self.batch_spin.setRange(-1, 128)
        self.batch_spin.setSpecialValueText("auto (-1)")
        self.batch_spin.setToolTip(
            "-1 lets ultralytics pick a batch size targeting ~60% of VRAM."
        )
 
        self.device_edit = QtWidgets.QLineEdit()
        self.device_edit.setPlaceholderText("empty = auto")
        self.device_edit.setToolTip(
            "Passed straight to ultralytics: '0', '0,1', 'cpu', 'mps'.\n"
            "Leave empty to let it choose."
        )
 
        self.patience_spin = QtWidgets.QSpinBox()
        self.patience_spin.setRange(0, 500)
        self.patience_spin.setSpecialValueText("off (0)")
        self.patience_spin.setToolTip(
            "Early stopping: epochs without improvement before aborting.\n"
            "Only meaningful once epochs is large."
        )
 
        self.workers_spin = QtWidgets.QSpinBox()
        self.workers_spin.setRange(0, 32)
        self.workers_spin.setToolTip(
            "Dataloader worker processes. 0 on Windows if you hit spawn "
            "issues."
        )
 
        form.addRow("Epochs:", self.epochs_spin)
        form.addRow("Image size:", self.imgsz_spin)
        form.addRow("Batch size:", self.batch_spin)
        form.addRow("Device:", self.device_edit)
        form.addRow("Patience:", self.patience_spin)
        form.addRow("Dataloader workers:", self.workers_spin)
        return box
 
    def _build_validation_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Validation split")
        form = QtWidgets.QFormLayout(box)
        form.setSpacing(10)
 
        self.val_split_spin = QtWidgets.QDoubleSpinBox()
        self.val_split_spin.setRange(0.0, 0.5)
        self.val_split_spin.setSingleStep(0.05)
        self.val_split_spin.setToolTip(
            "Fraction of annotated frames held out for validation."
        )
 
        self.val_type_combo = QtWidgets.QComboBox()
        for value, label in self.VAL_TYPES:
            self.val_type_combo.addItem(label, value)
        self.val_type_combo.setToolTip(
            "Consecutive video frames are near-duplicates. A random split\n"
            "puts frame n in train and frame n+1 in val, so the model is\n"
            "scored on images it has effectively already seen and every\n"
            "metric looks great for no reason.\n\n"
            "A temporal block holds out one contiguous stretch instead.\n"
            "Use 'middle' if the end of your footage is atypical (return\n"
            "leg, different sun angle)."
        )
 
        self.val_warning = QtWidgets.QLabel()
        self.val_warning.setWordWrap(True)
        self.val_warning.setStyleSheet("color:#c46a00;")
        self.val_type_combo.currentIndexChanged.connect(
            self._update_val_warning
        )
 
        form.addRow("Val split:", self.val_split_spin)
        form.addRow("Val type:", self.val_type_combo)
        form.addRow("", self.val_warning)
        self._update_val_warning()
        return box
 
    def _build_export_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Dataset export")
        form = QtWidgets.QFormLayout(box)
        form.setSpacing(10)
 
        self.multiscale_check = QtWidgets.QCheckBox(
            "Write zoomed crops alongside the full frame"
        )
        self.multiscale_check.setToolTip(
            "Multi-scale export: each annotated frame also produces crops\n"
            "centred loosely on the instances, at native resolution."
        )
        self.multiscale_check.toggled.connect(self._update_export_enabled)
 
        self.zoom_levels_spin = QtWidgets.QSpinBox()
        self.zoom_levels_spin.setRange(0, 5)
        self.zoom_levels_spin.setToolTip(
            "Number of zoom levels per frame, from the tightest crop up\n"
            "towards the full frame on a geometric ladder.\n"
            "The tightest level is always at native resolution.\n"
            "Levels that would not fit at the requested separation are\n"
            "dropped automatically, so asking for 5 may yield 2."
        )
 
        self.group_margin_spin = QtWidgets.QDoubleSpinBox()
        self.group_margin_spin.setRange(0.0, 1.0)
        self.group_margin_spin.setSingleStep(0.05)
        self.group_margin_spin.setDecimals(2)
        self.group_margin_spin.setToolTip(
            "Breathing room around the instance group in the tightest crop,\n"
            "as a fraction of the group size. Larger means more context and\n"
            "fewer usable crops."
        )
 
        self.scale_step_spin = QtWidgets.QDoubleSpinBox()
        self.scale_step_spin.setRange(1.05, 3.0)
        self.scale_step_spin.setSingleStep(0.1)
        self.scale_step_spin.setDecimals(2)
        self.scale_step_spin.setToolTip(
            "Minimum size ratio between two consecutive zoom levels.\n"
            "Below this they are near-duplicates that only cost disk and\n"
            "epoch time."
        )
 
        self.edge_pad_spin = QtWidgets.QSpinBox()
        self.edge_pad_spin.setRange(0, 64)
        self.edge_pad_spin.setSuffix(" px")
        self.edge_pad_spin.setToolTip(
            "Minimum distance kept between any annotation and the crop "
            "border."
        )
 
        self.jpeg_quality_spin = QtWidgets.QSpinBox()
        self.jpeg_quality_spin.setRange(70, 100)
        self.jpeg_quality_spin.setToolTip(
            "JPEG quality for exported images and the annotation archive.\n"
            "Block artefacts are visible on keypoints a few pixels wide;\n"
            "100 is near-lossless at roughly double the size."
        )
 
        form.addRow(self.multiscale_check)
        form.addRow("Zoom levels:", self.zoom_levels_spin)
        form.addRow("Group margin:", self.group_margin_spin)
        form.addRow("Min scale step:", self.scale_step_spin)
        form.addRow("Edge padding:", self.edge_pad_spin)
        form.addRow("JPEG quality:", self.jpeg_quality_spin)
        return box
 
    def _build_augmentation_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Augmentation")
        form = QtWidgets.QFormLayout(box)
        form.setSpacing(10)
 
        self.degrees_spin = QtWidgets.QDoubleSpinBox()
        self.degrees_spin.setRange(0.0, 180.0)
        self.degrees_spin.setSingleStep(15.0)
        self.degrees_spin.setSuffix(" °")
        self.degrees_spin.setToolTip(
            "Random rotation, ± this many degrees.\n"
            "Straight-down drone footage has no privileged heading, so 180 is\n"
            "free signal here. Rotation preserves left/right, so keypoints\n"
            "need no index remapping — unlike flips."
        )
 
        self.scale_spin = QtWidgets.QDoubleSpinBox()
        self.scale_spin.setRange(0.0, 0.9)
        self.scale_spin.setSingleStep(0.05)
        self.scale_spin.setDecimals(2)
        self.scale_spin.setToolTip(
            "Random rescale gain: 0.5 means images vary between 0.5x and\n"
            "1.5x. The ultralytics default of 0.5 halves an already small\n"
            "animal half the time; keep it low when the subject is tiny."
        )
 
        self.translate_spin = QtWidgets.QDoubleSpinBox()
        self.translate_spin.setRange(0.0, 0.9)
        self.translate_spin.setSingleStep(0.05)
        self.translate_spin.setDecimals(2)
        self.translate_spin.setToolTip(
            "Random shift, as a fraction of image size."
        )
 
        self.fliplr_spin = QtWidgets.QDoubleSpinBox()
        self.fliplr_spin.setRange(0.0, 1.0)
        self.fliplr_spin.setSingleStep(0.1)
        self.fliplr_spin.setDecimals(2)
 
        self.flipud_spin = QtWidgets.QDoubleSpinBox()
        self.flipud_spin.setRange(0.0, 1.0)
        self.flipud_spin.setSingleStep(0.1)
        self.flipud_spin.setDecimals(2)
 
        flip_tip = (
            "Probability of a mirror flip. Any mirror swaps left and right,\n"
            "so on a pose task ultralytics remaps the keypoints through\n"
            "data.yaml's flip_idx — for BOTH directions since 8.4.\n\n"
            "If flip_idx is missing from data.yaml, ultralytics silently\n"
            "forces both of these to 0."
        )
        self.fliplr_spin.setToolTip(flip_tip)
        self.flipud_spin.setToolTip(flip_tip)
 
        self.mosaic_spin = QtWidgets.QDoubleSpinBox()
        self.mosaic_spin.setRange(0.0, 1.0)
        self.mosaic_spin.setSingleStep(0.1)
        self.mosaic_spin.setDecimals(2)
        self.mosaic_spin.setToolTip(
            "Probability of tiling 4 images into one. Strong regulariser on\n"
            "small datasets, but it crops aggressively and compounds with\n"
            "'scale' to shrink small subjects further."
        )
 
        self.close_mosaic_spin = QtWidgets.QSpinBox()
        self.close_mosaic_spin.setRange(0, 100)
        self.close_mosaic_spin.setSpecialValueText("never (0)")
        self.close_mosaic_spin.setToolTip(
            "Disable mosaic for the last N epochs so the model finishes on\n"
            "undistorted images. With few epochs this can cover half the run."
        )
 
        self.hsv_v_spin = QtWidgets.QDoubleSpinBox()
        self.hsv_v_spin.setRange(0.0, 1.0)
        self.hsv_v_spin.setSingleStep(0.05)
        self.hsv_v_spin.setDecimals(3)
        self.hsv_v_spin.setToolTip(
            "Brightness jitter. Worth keeping high for aerial water: sun\n"
            "angle, glare and depth change exposure a lot between flights."
        )
 
        self.hsv_s_spin = QtWidgets.QDoubleSpinBox()
        self.hsv_s_spin.setRange(0.0, 1.0)
        self.hsv_s_spin.setSingleStep(0.05)
        self.hsv_s_spin.setDecimals(3)
        self.hsv_s_spin.setToolTip(
            "Saturation jitter — covers turbidity and water colour changes."
        )
 
        self.hsv_h_spin = QtWidgets.QDoubleSpinBox()
        self.hsv_h_spin.setRange(0.0, 0.2)
        self.hsv_h_spin.setSingleStep(0.005)
        self.hsv_h_spin.setDecimals(3)
        self.hsv_h_spin.setToolTip(
            "Hue jitter. Keep small: water hue is fairly consistent and\n"
            "shifting it far makes the images unrepresentative."
        )
 
        form.addRow("Rotation:", self.degrees_spin)
        form.addRow("Scale gain:", self.scale_spin)
        form.addRow("Translate:", self.translate_spin)
        form.addRow("Flip horizontal:", self.fliplr_spin)
        form.addRow("Flip vertical:", self.flipud_spin)
        form.addRow("Mosaic:", self.mosaic_spin)
        form.addRow("Close mosaic:", self.close_mosaic_spin)
        form.addRow("HSV value:", self.hsv_v_spin)
        form.addRow("HSV saturation:", self.hsv_s_spin)
        form.addRow("HSV hue:", self.hsv_h_spin)
        return box
 
    # ==================== Reactions ====================
 
    def _update_val_warning(self):
        leaky = self.val_type_combo.currentData() == "random"
        self.val_warning.setText(
            "Neighbouring frames are near-duplicates: val metrics will be "
            "optimistic and not comparable to a temporal split."
            if leaky else ""
        )
        self.val_warning.setVisible(leaky)
 
    def _update_export_enabled(self, on: bool):
        for widget in (self.zoom_levels_spin, self.group_margin_spin,
                       self.scale_step_spin, self.edge_pad_spin):
            widget.setEnabled(on)
 
    # ==================== Config ====================
 
    def load_config(self, cfg: dict):
        self.epochs_spin.setValue(cfg.get("epochs", 20))
        self.imgsz_spin.setValue(cfg.get("imgsz", 1024))
        self.batch_spin.setValue(cfg.get("batch", 16))
        self.device_edit.setText(str(cfg.get("device", "")))
        self.patience_spin.setValue(cfg.get("patience", 0))
        self.workers_spin.setValue(cfg.get("workers", 8))
 
        self.val_split_spin.setValue(cfg.get("val_split", 0.1))
        wanted = str(cfg.get("val_type", "end"))
        index = self.val_type_combo.findData(wanted)
        self.val_type_combo.setCurrentIndex(index if index >= 0 else 0)
        self._update_val_warning()
 
        multiscale = bool(cfg.get("multiscale_export", True))
        self.multiscale_check.setChecked(multiscale)
        self.zoom_levels_spin.setValue(cfg.get("crop_zoom_levels", 2))
        self.group_margin_spin.setValue(cfg.get("crop_group_margin", 0.15))
        self.scale_step_spin.setValue(cfg.get("crop_min_scale_step", 1.3))
        self.edge_pad_spin.setValue(int(cfg.get("crop_edge_pad", 6)))
        self.jpeg_quality_spin.setValue(cfg.get("export_jpeg_quality", 98))
        self._update_export_enabled(multiscale)
 
        # Defaults come from tasks.DEFAULT_TRAIN_OVERRIDES, the same dict the
        # training page falls back on, so what this panel shows is always what
        # a run would actually use.
        d = DEFAULT_TRAIN_OVERRIDES
        for key, widget in (
            ("degrees", self.degrees_spin), ("scale", self.scale_spin),
            ("translate", self.translate_spin), ("fliplr", self.fliplr_spin),
            ("flipud", self.flipud_spin), ("mosaic", self.mosaic_spin),
            ("close_mosaic", self.close_mosaic_spin),
            ("hsv_v", self.hsv_v_spin), ("hsv_s", self.hsv_s_spin),
            ("hsv_h", self.hsv_h_spin),
        ):
            widget.setValue(cfg.get(key, d[key]))
 
    def to_config(self, cfg: dict):
        cfg["epochs"] = self.epochs_spin.value()
        cfg["imgsz"] = self.imgsz_spin.value()
        cfg["batch"] = self.batch_spin.value()
        # Empty means "let ultralytics decide"; storing "" keeps that explicit.
        cfg["device"] = self.device_edit.text().strip()
        cfg["patience"] = self.patience_spin.value()
        cfg["workers"] = self.workers_spin.value()
 
        cfg["val_split"] = self.val_split_spin.value()
        cfg["val_type"] = self.val_type_combo.currentData()
 
        cfg["multiscale_export"] = self.multiscale_check.isChecked()
        cfg["crop_zoom_levels"] = self.zoom_levels_spin.value()
        cfg["crop_group_margin"] = self.group_margin_spin.value()
        cfg["crop_min_scale_step"] = self.scale_step_spin.value()
        cfg["crop_edge_pad"] = float(self.edge_pad_spin.value())
        cfg["export_jpeg_quality"] = self.jpeg_quality_spin.value()
 
        cfg["degrees"] = self.degrees_spin.value()
        cfg["scale"] = self.scale_spin.value()
        cfg["translate"] = self.translate_spin.value()
        cfg["fliplr"] = self.fliplr_spin.value()
        cfg["flipud"] = self.flipud_spin.value()
        cfg["mosaic"] = self.mosaic_spin.value()
        cfg["close_mosaic"] = self.close_mosaic_spin.value()
        cfg["hsv_v"] = self.hsv_v_spin.value()
        cfg["hsv_s"] = self.hsv_s_spin.value()
        cfg["hsv_h"] = self.hsv_h_spin.value()


class TrackingSettingsPanel(QtWidgets.QWidget):
    """Panel for multi-object tracking settings.

    Owns: ``tracker_type``, ``reid_weights``, ``with_reid``,
    ``track_high_thresh``, ``track_low_thresh``, ``new_track_thresh``,
    ``track_buffer``, ``match_thresh``, ``proximity_thresh``,
    ``appearance_thresh``.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    # ==================== UI ====================

    def _build_ui(self):
        form = QtWidgets.QFormLayout(self)
        form.setContentsMargins(20, 20, 20, 20)
        form.setSpacing(12)

        # Tracker type
        self.tracker_type_combo = QtWidgets.QComboBox()
        self.tracker_type_combo.addItems([
            "botsort"
        ])

        # ReID weights
        # self.reid_weights_edit = QtWidgets.QLineEdit("osnet_x0_25_msmt17.pt")
        # reid_browse = QtWidgets.QPushButton("Browse...")
        # reid_browse.clicked.connect(self._browse_reid)
        # reid_row = QtWidgets.QHBoxLayout()
        # reid_row.addWidget(self.reid_weights_edit, stretch=1)
        # reid_row.addWidget(reid_browse)

        # self.with_reid_chk = QtWidgets.QCheckBox("Enable ReID")
        # self.with_reid_chk.setChecked(True)

        # Threshold spinboxes
        self.track_high_spin   = self._make_unit_spin(0.6)
        self.track_low_spin    = self._make_unit_spin(0.1)
        self.new_track_spin    = self._make_unit_spin(0.7)
        self.match_thresh_spin = self._make_unit_spin(0.8)
        self.proximity_spin    = self._make_unit_spin(0.5)
        self.appearance_spin   = self._make_unit_spin(0.25)

        # Track buffer (frames)
        self.track_buffer_spin = QtWidgets.QSpinBox()
        self.track_buffer_spin.setRange(1, 300)
        self.track_buffer_spin.setValue(30)

        form.addRow("Tracker type:",      self.tracker_type_combo)
        # form.addRow("ReID weights:",      reid_row)
        # form.addRow("",                   self.with_reid_chk)
        form.addRow("Track high thresh:", self.track_high_spin)
        form.addRow("Track low thresh:",  self.track_low_spin)
        form.addRow("New track thresh:",  self.new_track_spin)
        form.addRow("Track buffer:",      self.track_buffer_spin)
        form.addRow("Match thresh:",      self.match_thresh_spin)
        form.addRow("Proximity thresh:",  self.proximity_spin)
        form.addRow("Appearance thresh:", self.appearance_spin)

    @staticmethod
    def _make_unit_spin(default: float) -> QtWidgets.QDoubleSpinBox:
        """Helper: 0.01..0.99 spinbox with 0.05 step and the given default."""
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(0.01, 0.99)
        spin.setSingleStep(0.05)
        spin.setValue(default)
        return spin

    # ==================== Browse ====================

    # def _browse_reid(self):
    #     path, _ = QtWidgets.QFileDialog.getOpenFileName(
    #         self, "Select ReID weights", "",
    #         "Model files (*.pt *.pth *.onnx);;All files (*)",
    #     )
    #     if path:
    #         self.reid_weights_edit.setText(path)

    # ==================== Config ====================

    def load_config(self, cfg: dict):
        self.tracker_type_combo.setCurrentText(
            cfg.get("tracker_type", "botsort")
        )
        # self.reid_weights_edit.setText(
        #     cfg.get("reid_weights", "osnet_x0_25_msmt17.pt")
        # )
        # self.with_reid_chk.setChecked(cfg.get("with_reid", True))
        self.track_high_spin.setValue(cfg.get("track_high_thresh", 0.6))
        self.track_low_spin.setValue(cfg.get("track_low_thresh", 0.1))
        self.new_track_spin.setValue(cfg.get("new_track_thresh", 0.7))
        self.track_buffer_spin.setValue(cfg.get("track_buffer", 30))
        self.match_thresh_spin.setValue(cfg.get("match_thresh", 0.8))
        self.proximity_spin.setValue(cfg.get("proximity_thresh", 0.5))
        self.appearance_spin.setValue(cfg.get("appearance_thresh", 0.25))

    def to_config(self, cfg: dict):
        cfg["tracker_type"]      = self.tracker_type_combo.currentText()
        # cfg["reid_weights"]      = self.reid_weights_edit.text()
        # cfg["with_reid"]         = self.with_reid_chk.isChecked()
        cfg["track_high_thresh"] = self.track_high_spin.value()
        cfg["track_low_thresh"]  = self.track_low_spin.value()
        cfg["new_track_thresh"]  = self.new_track_spin.value()
        cfg["track_buffer"]      = self.track_buffer_spin.value()
        cfg["match_thresh"]      = self.match_thresh_spin.value()
        cfg["proximity_thresh"]  = self.proximity_spin.value()
        cfg["appearance_thresh"] = self.appearance_spin.value()

class ProjectSettingsPanel(QtWidgets.QWidget):
    """Panel for project-level settings.

    Owns: ``model_path``, ``dataset_dir``, ``class_names`` and the pose-only
    keypoint settings (``num_keypoints``, ``keypoint_names``, ``flip_idx``,
    ``pose_bbox_mode``).

    ``task_type`` is displayed read-only: it is fixed when the project is
    created and cannot change afterwards, because the labels already written
    to disk are task-specific.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._task = TASK_POSE
        self._build_ui()

    # ==================== UI ====================

    def _build_ui(self):
        form = QtWidgets.QFormLayout(self)
        form.setContentsMargins(20, 20, 20, 20)
        form.setSpacing(12)

        # Task (read-only — immutable once the project exists)
        self.task_value_label = QtWidgets.QLabel("—")
        self.task_value_label.setStyleSheet("font-weight: bold;")
        self.task_value_label.setToolTip(
            "The task is chosen when the project is created and cannot be "
            "changed: existing labels would no longer match. Create a new "
            "project to work with a different task."
        )

        # Base model used to start every fine-tune. Restricted to official
        # ultralytics checkpoints: training must never stack on top of a
        # previous run's weights.
        self.base_model_combo = QtWidgets.QComboBox()
        self.base_model_combo.setToolTip(
            "Pretrained ultralytics checkpoint every fine-tune starts from.\n"
            "Larger scales (m, l, x) are more accurate but slower to train."
        )

        # Current inference weights (updated automatically after a fine-tune)
        self.model_path_edit = QtWidgets.QLineEdit()
        self.model_path_edit.setPlaceholderText(
            "empty = use the pretrained base model"
        )
        self.model_path_edit.setToolTip(
            "Weights used for inference. Set automatically to the fine-tuned "
            "best.pt at the end of a training run."
        )
        model_browse = QtWidgets.QPushButton("Browse...")
        model_browse.clicked.connect(self._browse_model)
        model_reset = QtWidgets.QPushButton("Reset")
        model_reset.setToolTip("Go back to the pretrained base model")
        model_reset.clicked.connect(lambda: self.model_path_edit.setText(""))
        model_row = QtWidgets.QHBoxLayout()
        model_row.addWidget(self.model_path_edit, stretch=1)
        model_row.addWidget(model_browse)
        model_row.addWidget(model_reset)

        # Dataset dir
        self.dataset_dir_edit = QtWidgets.QLineEdit()
        ds_browse = QtWidgets.QPushButton("Browse...")
        ds_browse.clicked.connect(self._browse_dataset)
        ds_row = QtWidgets.QHBoxLayout()
        ds_row.addWidget(self.dataset_dir_edit, stretch=1)
        ds_row.addWidget(ds_browse)

        # Class names
        self.class_names_edit = QtWidgets.QLineEdit()
        self.class_names_edit.setToolTip(
            "Comma-separated class names, e.g.: cat, dog, bird"
        )

        # Finetune dir (read-only display)
        self.finetune_dir_label = QtWidgets.QLineEdit()
        self.finetune_dir_label.setReadOnly(True)

        form.addRow("Task:",              self.task_value_label)
        form.addRow("Base model:",        self.base_model_combo)
        form.addRow("Inference weights:", model_row)
        form.addRow("Dataset dir:",    ds_row)
        form.addRow("Class names:",    self.class_names_edit)
        form.addRow("Finetune dir:",   self.finetune_dir_label)

        # ---- Pose-only group ----
        self.pose_box = QtWidgets.QGroupBox("Keypoints (pose task)")
        pose_form = QtWidgets.QFormLayout(self.pose_box)
        pose_form.setSpacing(10)

        self.num_kpt_spin = QtWidgets.QSpinBox()
        self.num_kpt_spin.setRange(1, 64)
        self.num_kpt_spin.setToolTip(
            "Number of keypoints annotated per instance."
        )
        self.num_kpt_spin.valueChanged.connect(self._on_num_kpt_changed)

        self.kpt_names_edit = QtWidgets.QLineEdit()
        self.kpt_names_edit.setToolTip(
            "Comma-separated keypoint names, in annotation order.\n"
            "e.g.: head_left, head_right, com, dorsal, tail"
        )

        self.flip_idx_edit = QtWidgets.QLineEdit()
        self.flip_idx_edit.setToolTip(
            "Horizontal-flip mapping, one target index per keypoint.\n"
            "Keypoints that swap under a mirror (e.g. the left and right tips "
            "of the cephalofoil) must point at each other; the others map to "
            "themselves.\n"
            "Must be a permutation of 0..N-1, otherwise the default is used."
        )

        self.pose_bbox_combo = QtWidgets.QComboBox()
        self.pose_bbox_combo.addItems(list(POSE_BBOX_MODES))
        self.pose_bbox_combo.setToolTip(
            "auto   = the box is derived from the keypoint extent\n"
            "manual = draw the box first, then place the keypoints"
        )

        pose_form.addRow("Num keypoints:", self.num_kpt_spin)
        pose_form.addRow("Keypoint names:", self.kpt_names_edit)
        pose_form.addRow("Flip idx:",       self.flip_idx_edit)
        pose_form.addRow("Bbox mode:",      self.pose_bbox_combo)

        form.addRow(self.pose_box)

    # ==================== Browse ====================

    def _browse_model(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select model weights", "",
            "Model files (*.pt *.ckpt *.pth);;All files (*)",
        )
        if path:
            self.model_path_edit.setText(path)

    def _browse_dataset(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select dataset directory, WARNING it will be modified", "",
        )
        if path:
            self.dataset_dir_edit.setText(path)

    # ==================== Reactions ====================

    def _on_num_kpt_changed(self, n: int):
        """Resize the names and flip_idx fields to match the keypoint count.

        Existing names are kept and the list is padded or truncated, so
        renaming work is not lost when the count is nudged.
        """
        names = [s.strip() for s in self.kpt_names_edit.text().split(",") if s.strip()]
        defaults = default_keypoint_names(n)
        names = (names + defaults[len(names):])[:n]
        self.kpt_names_edit.setText(", ".join(names))

        # A flip mapping only makes sense for one fixed N: rebuild it.
        self.flip_idx_edit.setText(format_flip_idx(default_flip_idx(n)))

    # ==================== Config ====================

    def load_config(self, cfg: dict):
        self._task = normalize_task(cfg.get("task_type"))
        self.task_value_label.setText(TASK_LABELS.get(self._task, self._task))

        # Only official pretrained checkpoints for this task are offered.
        choices = pretrained_choices(self._task)
        self.base_model_combo.blockSignals(True)
        self.base_model_combo.clear()
        self.base_model_combo.addItems(choices)
        base = cfg.get("default_model") or default_model_for(self._task)
        if base not in choices:
            # e.g. a config carried over from another task.
            base = default_model_for(self._task)
        self.base_model_combo.setCurrentText(base)
        self.base_model_combo.blockSignals(False)

        self.model_path_edit.setText(cfg.get("model_path", ""))
        self.dataset_dir_edit.setText(cfg.get("dataset_dir", ""))

        names = cfg.get("class_names", ["object"])
        if isinstance(names, list):
            self.class_names_edit.setText(", ".join(names))
        else:
            self.class_names_edit.setText(str(names))

        self.finetune_dir_label.setText(cfg.get("finetune_dir", ""))

        # Pose settings
        n_kpt = int(cfg.get("num_keypoints", 5))
        self.num_kpt_spin.blockSignals(True)
        self.num_kpt_spin.setValue(n_kpt)
        self.num_kpt_spin.blockSignals(False)

        kpt_names = cfg.get("keypoint_names") or default_keypoint_names(n_kpt)
        self.kpt_names_edit.setText(", ".join(kpt_names))

        flip = cfg.get("flip_idx") or default_flip_idx(n_kpt)
        self.flip_idx_edit.setText(format_flip_idx(flip))

        self.pose_bbox_combo.setCurrentText(
            cfg.get("pose_bbox_mode", POSE_BBOX_AUTO)
        )

        # Keypoint settings are meaningless outside the pose task.
        self.pose_box.setVisible(self._task == TASK_POSE)

    def to_config(self, cfg: dict):
        names_raw = self.class_names_edit.text()
        names = [n.strip() for n in names_raw.split(",") if n.strip()]
        if not names:
            names = ["object"]

        cfg["default_model"] = self.base_model_combo.currentText()
        cfg["model_path"]  = self.model_path_edit.text()
        cfg["dataset_dir"] = self.dataset_dir_edit.text()
        cfg["class_names"] = names
        # task_type is intentionally NOT written back: ProjectManager owns it.

        n_kpt = self.num_kpt_spin.value()
        kpt_names = [s.strip() for s in self.kpt_names_edit.text().split(",")
                     if s.strip()]
        defaults = default_keypoint_names(n_kpt)
        kpt_names = (kpt_names + defaults[len(kpt_names):])[:n_kpt]

        cfg["num_keypoints"]  = n_kpt
        cfg["keypoint_names"] = kpt_names
        cfg["flip_idx"]       = parse_flip_idx(self.flip_idx_edit.text(), n_kpt)
        cfg["pose_bbox_mode"] = self.pose_bbox_combo.currentText()