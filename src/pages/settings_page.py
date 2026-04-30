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

from ..settings.settings_project import ProjectSettingsPanel
from ..settings.settings_training import TrainingSettingsPanel
from ..settings.settings_tracking import TrackingSettingsPanel


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
