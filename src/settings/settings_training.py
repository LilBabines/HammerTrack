"""Training settings sub-panel — fine-tuning hyperparameters."""

from PySide6 import QtWidgets


class TrainingSettingsPanel(QtWidgets.QWidget):
    """Panel for training (fine-tuning) hyperparameters.

    Owns: ``epochs``, ``imgsz``, ``batch``, ``val_split``.

    Note: ``imgsz`` is also used at inference time — there is intentionally a
    single value shared by both.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    # ==================== UI ====================

    def _build_ui(self):
        form = QtWidgets.QFormLayout(self)
        form.setContentsMargins(20, 20, 20, 20)
        form.setSpacing(12)

        self.epochs_spin = QtWidgets.QSpinBox()
        self.epochs_spin.setRange(1, 500)

        self.imgsz_spin = QtWidgets.QSpinBox()
        self.imgsz_spin.setRange(128, 4096)
        self.imgsz_spin.setSingleStep(64)
        self.imgsz_spin.setToolTip(
            "Model image size — used both at training and inference."
        )

        self.batch_spin = QtWidgets.QSpinBox()
        self.batch_spin.setRange(1, 128)

        self.val_split_spin = QtWidgets.QDoubleSpinBox()
        self.val_split_spin.setRange(0.0, 0.5)
        self.val_split_spin.setSingleStep(0.05)

        form.addRow("Epochs:",     self.epochs_spin)
        form.addRow("Image size:", self.imgsz_spin)
        form.addRow("Batch size:", self.batch_spin)
        form.addRow("Val split:",  self.val_split_spin)

    # ==================== Config ====================

    def load_config(self, cfg: dict):
        self.epochs_spin.setValue(cfg.get("epochs", 20))
        self.imgsz_spin.setValue(cfg.get("imgsz", 1024))
        self.batch_spin.setValue(cfg.get("batch", 16))
        self.val_split_spin.setValue(cfg.get("val_split", 0.1))

    def to_config(self, cfg: dict):
        cfg["epochs"]    = self.epochs_spin.value()
        cfg["imgsz"]     = self.imgsz_spin.value()
        cfg["batch"]     = self.batch_spin.value()
        cfg["val_split"] = self.val_split_spin.value()
