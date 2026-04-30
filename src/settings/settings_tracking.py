"""Tracking settings sub-panel — tracker type and ReID configuration."""

from PySide6 import QtWidgets


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
