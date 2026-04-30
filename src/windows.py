"""
LauncherWindow — the application's main window.

Responsibilities:
* Project selector / creator on the top bar
* Tab buttons that switch between :class:`SettingsPage`,
  :class:`AnnotatePage`, :class:`InspectDatasetPage`, :class:`TrainPage`
  and :class:`TrackingPage`
* Menu bar (File / Help)
* Global keyboard shortcuts that delegate to the annotate page
* Project config persistence (via :class:`ProjectManager`)
"""

from typing import List, Optional

from PySide6 import QtCore, QtGui, QtWidgets

from .pages.annotate_page import AnnotatePage
from .pages.inspect_page import InspectDatasetPage
from .project_manager import ProjectManager
from .pages.settings_page import SettingsPage
from .tracking.tracking_page import TrackingPage
from .pages.train_page import TrainPage


class LauncherWindow(QtWidgets.QMainWindow):
    """Main application window with project management and tabbed pages."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Annotation Tool")
        self.resize(1300, 820)

        self.pm = ProjectManager()
        self._current_project: Optional[str] = None

        self._build_top_bar()
        self._build_tab_bar()
        self._build_pages()
        self._build_central_layout()
        self._build_menu_bar()
        self._build_shortcuts()

        # Init
        self._refresh_projects()
        self.tab_buttons[0].setChecked(True)
        self._switch_tab(0)

    # ==================== UI construction ====================

    def _build_top_bar(self):
        """Project selector + new/refresh buttons."""
        self._top_bar = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(self._top_bar)
        h.setContentsMargins(12, 8, 12, 4)
        h.setSpacing(10)

        h.addWidget(QtWidgets.QLabel("Project:"))
        self.project_combo = QtWidgets.QComboBox()
        self.project_combo.setMinimumWidth(200)
        self.project_combo.currentTextChanged.connect(self._on_project_changed)
        h.addWidget(self.project_combo)

        self.new_project_btn = QtWidgets.QPushButton("New project")
        self.new_project_btn.clicked.connect(self._new_project)
        h.addWidget(self.new_project_btn)

        self.refresh_btn = QtWidgets.QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._refresh_projects)
        h.addWidget(self.refresh_btn)

        self.project_label = QtWidgets.QLabel("")
        h.addWidget(self.project_label)
        h.addStretch(1)

    def _build_tab_bar(self):
        """Tab buttons that drive ``self.stack``."""
        self._tab_bar = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(self._tab_bar)
        h.setContentsMargins(12, 0, 12, 0)
        h.setSpacing(4)

        self.tab_group = QtWidgets.QButtonGroup(self)
        self.tab_group.setExclusive(True)
        self.tab_buttons: List[QtWidgets.QPushButton] = []
        tab_names = [
            "Settings", "Annotate", "Inspect Dataset",
            "Train Detector", "Tracking",
        ]
        for i, name in enumerate(tab_names):
            btn = QtWidgets.QPushButton(name)
            btn.setCheckable(True)
            self.tab_group.addButton(btn, i)
            self.tab_buttons.append(btn)
            h.addWidget(btn)
        h.addStretch(1)

        self.tab_group.idClicked.connect(self._switch_tab)

    def _build_pages(self):
        """Create all tab pages and stack them."""
        self.stack = QtWidgets.QStackedWidget()

        self.settings_page = SettingsPage()
        self.settings_page.config_changed.connect(self._save_current_config)
        self.stack.addWidget(self.settings_page)

        self.annotate_page = AnnotatePage()
        self.annotate_page.set_launcher(self)
        self.stack.addWidget(self.annotate_page)

        self.inspect_page = InspectDatasetPage()
        self.inspect_page.set_launcher(self)
        self.stack.addWidget(self.inspect_page)

        self.train_page = TrainPage()
        self.train_page.set_launcher(self)
        self.stack.addWidget(self.train_page)

        self.tracking_page = TrackingPage()
        self.tracking_page.set_launcher(self)
        self.stack.addWidget(self.tracking_page)

    def _build_central_layout(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        v = QtWidgets.QVBoxLayout(central)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(self._top_bar)
        v.addWidget(self._tab_bar)
        v.addWidget(self.stack, stretch=1)

    def _build_menu_bar(self):
        menubar = self.menuBar()

        # File
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

        # Help
        help_menu = menubar.addMenu("&Help")
        about_act = QtGui.QAction("About", self)
        about_act.triggered.connect(self._show_about)
        help_menu.addAction(about_act)

    def _build_shortcuts(self):
        """Global keyboard shortcuts that delegate to ``annotate_page``."""
        ap = self.annotate_page
        shortcuts = [
            (QtCore.Qt.Key.Key_Left,   ap.prev_frame),
            (QtCore.Qt.Key.Key_Right,  ap.next_frame),
            ("V",                      ap.verify_selected_toggle),
            (QtCore.Qt.Key.Key_Delete, ap.delete_selected),
            ("N",                      ap.start_add_mode),
            ("B",                      ap.start_add_bbox_mode),
            ("E",                      ap.toggle_edit_mode),
            ("Esc",                    ap.cancel_add_mode),
            ("D",                      ap.export_to_dataset),
            ("+",                      lambda: ap.zoom_step(+1)),
            ("-",                      lambda: ap.zoom_step(-1)),
            ("0",                      ap.zoom_fit),
        ]
        for key, slot in shortcuts:
            QtGui.QShortcut(QtGui.QKeySequence(key), self, activated=slot)

    def _show_about(self):
        QtWidgets.QMessageBox.information(
            self, "About",
            "Annotation & Active Learning Tool\n"
            "YOLO-OBB / YOLO-Detect with human-in-the-loop finetuning\n"
            "Built with PySide6",
        )

    # ==================== Project management ====================

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
        self.statusBar().showMessage(
            f"Settings saved for project '{self._current_project}'.", 4000
        )

    # ==================== Tab switching ====================

    def _switch_tab(self, idx: int):
        self.stack.setCurrentIndex(idx)
        if idx == 2:
            self.inspect_page.refresh()

    # ==================== Title & misc ====================

    def update_title(self):
        parts = ["Annotation Tool"]
        if self._current_project:
            parts.append(self._current_project)
        ap = self.annotate_page
        if ap.source:
            parts.append(ap.source.name())
            parts.append(f"frame {ap.current_idx + 1}/{ap.total_frames}")
        self.setWindowTitle(" | ".join(parts))

    # ==================== Keyboard / resize forwarding ====================

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if event.key() == QtCore.Qt.Key.Key_Space:
            self.annotate_page.set_space_held(True)
            self.tracking_page.set_space_held(True)
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QtGui.QKeyEvent):
        if event.key() == QtCore.Qt.Key.Key_Space:
            self.annotate_page.set_space_held(False)
            self.tracking_page.set_space_held(False)
        else:
            super().keyReleaseEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.annotate_page.redraw_current()
