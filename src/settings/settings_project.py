"""Project settings sub-panel — model, dataset, classes, keypoints."""

from PySide6 import QtWidgets

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
)


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