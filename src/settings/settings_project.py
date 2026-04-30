"""Project settings sub-panel — model, dataset, classes, task type."""

from PySide6 import QtWidgets


class ProjectSettingsPanel(QtWidgets.QWidget):
    """Panel for project-level settings.

    Owns: ``model_path``, ``dataset_dir``, ``class_names``, ``task_type``
    (and a read-only display of ``finetune_dir``).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    # ==================== UI ====================

    def _build_ui(self):
        form = QtWidgets.QFormLayout(self)
        form.setContentsMargins(20, 20, 20, 20)
        form.setSpacing(12)

        # Model weights
        self.model_path_edit = QtWidgets.QLineEdit()
        model_browse = QtWidgets.QPushButton("Browse...")
        model_browse.clicked.connect(self._browse_model)
        model_row = QtWidgets.QHBoxLayout()
        model_row.addWidget(self.model_path_edit, stretch=1)
        model_row.addWidget(model_browse)

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

        # Task type
        self.task_type_combo = QtWidgets.QComboBox()
        self.task_type_combo.addItems(["auto", "obb", "detect"])
        self.task_type_combo.setToolTip(
            "auto = detect from model; obb = oriented boxes; "
            "detect = axis-aligned boxes"
        )

        # Finetune dir (read-only display)
        self.finetune_dir_label = QtWidgets.QLineEdit()
        self.finetune_dir_label.setReadOnly(True)

        form.addRow("Model weights:", model_row)
        form.addRow("Dataset dir:",   ds_row)
        form.addRow("Class names:",   self.class_names_edit)
        form.addRow("Task type:",     self.task_type_combo)
        form.addRow("Finetune dir:",  self.finetune_dir_label)

    # ==================== Browse ====================

    def _browse_model(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select model weights", "",
            "Model files (*.pt *.ckpt *.pth);;All files (*)",
        )
        if path:
            self.model_path_edit.setText(path)

    def _browse_dataset(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select dataset, WARNING it will be modified", "",
        )
        if path:
            self.dataset_dir_edit.setText(path)

    # ==================== Config ====================

    def load_config(self, cfg: dict):
        self.model_path_edit.setText(cfg.get("model_path", ""))
        self.dataset_dir_edit.setText(cfg.get("dataset_dir", ""))

        names = cfg.get("class_names", ["object"])
        if isinstance(names, list):
            self.class_names_edit.setText(", ".join(names))
        else:
            self.class_names_edit.setText(str(names))

        self.task_type_combo.setCurrentText(cfg.get("task_type", "auto"))
        self.finetune_dir_label.setText(cfg.get("finetune_dir", ""))

    def to_config(self, cfg: dict):
        names_raw = self.class_names_edit.text()
        names = [n.strip() for n in names_raw.split(",") if n.strip()]
        if not names:
            names = ["object"]

        cfg["model_path"]  = self.model_path_edit.text()
        cfg["dataset_dir"] = self.dataset_dir_edit.text()
        cfg["class_names"] = names
        cfg["task_type"]   = self.task_type_combo.currentText()
