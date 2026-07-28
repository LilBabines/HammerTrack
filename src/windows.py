"""
LauncherWindow — the application's main window.

Navigation has two states:

* **Home** — no project open. Only :class:`HomePage` is visible: pick a
  project or create one. The top bar and the tab bar are hidden, because
  every other page reads its data from the active project's config and has
  nothing to show without one.
* **In a project** — the top bar (project name + "Projects" button) and the
  tab bar appear, and the five working pages become reachable.

Responsibilities:
* Home / project routing and project config persistence (via ``ProjectManager``)
* Tab buttons switching between :class:`SettingsPage`, :class:`AnnotatePage`,
  :class:`InspectDatasetPage`, :class:`TrainPage` and :class:`TrackingPage`
* Menu bar (File / Help)
* Global keyboard shortcuts that delegate to the annotate page
"""

from typing import List, Optional

from PySide6 import QtCore, QtGui, QtWidgets

from .pages.annotate_page import AnnotatePage
from .pages.home_page import HomePage
from .pages.inspect_page import InspectDatasetPage
from .project_manager import ProjectManager
from .pages.settings_page import SettingsPage
from .tracking.tracking_page import TrackingPage
from .pages.train_page import TrainPage
from .tasks import TASKS, TASK_LABELS, TASK_OBB, project_folder_name


class LauncherWindow(QtWidgets.QMainWindow):
    """Main application window with project management and tabbed pages."""

    #: Stack index of the home page; the tab pages start right after it.
    HOME_INDEX = 0
    #: Tab index shown right after a project is opened.
    DEFAULT_TAB = 0          # Settings

    def __init__(self):
        super().__init__()
        self.setWindowTitle("HammerTrack")
        self.resize(1300, 820)

        self.pm = ProjectManager()
        self._current_project: Optional[str] = None

        self._build_top_bar()
        self._build_tab_bar()
        self._build_pages()
        self._build_central_layout()
        self._build_menu_bar()
        self._build_shortcuts()

        # Start on the home page: no project is open yet.
        self._go_home()

    # ==================== UI construction ====================

    def _build_top_bar(self):
        """Active-project bar. Hidden while the home page is showing."""
        self._top_bar = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(self._top_bar)
        h.setContentsMargins(12, 8, 12, 4)
        h.setSpacing(10)

        self.home_btn = QtWidgets.QPushButton("\u2190 Projects")
        self.home_btn.setToolTip("Close this project and go back to the list")
        self.home_btn.clicked.connect(self._go_home)
        h.addWidget(self.home_btn)

        self.project_label = QtWidgets.QLabel("")
        font = self.project_label.font()
        font.setBold(True)
        self.project_label.setFont(font)
        h.addWidget(self.project_label)
        h.addStretch(1)

    def _build_tab_bar(self):
        """Tab buttons that drive ``self.stack``. Hidden on the home page."""
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
        """Create the home page and all tab pages, then stack them."""
        self.stack = QtWidgets.QStackedWidget()

        # Index 0 — home. Tab N lives at stack index N + 1.
        self.home_page = HomePage()
        self.home_page.project_selected.connect(self._open_project)
        self.home_page.new_project_requested.connect(self._new_project)
        self.home_page.refresh_btn.clicked.connect(lambda: self._refresh_projects())
        self.stack.addWidget(self.home_page)

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

        self.close_project_act = QtGui.QAction("Close project", self)
        self.close_project_act.setShortcut("Ctrl+W")
        self.close_project_act.triggered.connect(self._go_home)
        file_menu.addAction(self.close_project_act)
        file_menu.addSeparator()

        open_video_act = QtGui.QAction("Open Video...", self)
        open_video_act.setShortcut("Ctrl+O")
        open_video_act.triggered.connect(self.annotate_page.open_video)

        open_images_act = QtGui.QAction("Open Image Folder...", self)
        open_images_act.setShortcut("Ctrl+I")
        open_images_act.triggered.connect(self.annotate_page.open_folder)

        self._open_menu = QtWidgets.QMenu("Open", self)
        self._open_menu.addAction(open_video_act)
        self._open_menu.addAction(open_images_act)
        file_menu.addMenu(self._open_menu)

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
        """Global shortcuts delegating to ``annotate_page``.

        They are kept in a list so they can be switched off on the home page,
        where firing an annotation action would make no sense.
        """
        ap = self.annotate_page
        shortcuts = [
            (QtCore.Qt.Key.Key_Left,   ap.prev_frame),
            (QtCore.Qt.Key.Key_Right,  ap.next_frame),
            ("Shift+Left",             ap.prev_annotated_frame),
            ("Shift+Right",            ap.next_annotated_frame),
            ("V",                      ap.verify_selected_toggle),
            (QtCore.Qt.Key.Key_Delete, ap.delete_selected),
            ("N",                      ap.start_add_mode),
            ("B",                      ap.start_add_bbox_mode),
            ("K",                      ap.start_add_pose_mode),
            ("E",                      ap.toggle_edit_mode),
            ("Esc",                    ap.cancel_add_mode),
            ("D",                      ap.export_to_dataset),
            ("+",                      lambda: ap.zoom_step(+1)),
            ("-",                      lambda: ap.zoom_step(-1)),
            ("0",                      ap.zoom_fit),
        ]
        self._shortcuts: List[QtGui.QShortcut] = [
            QtGui.QShortcut(QtGui.QKeySequence(key), self, activated=slot)
            for key, slot in shortcuts
        ]

    def _show_about(self):
        QtWidgets.QMessageBox.information(
            self, "About",
            "HammerTrack — Annotation & Active Learning Tool\n"
            "YOLO detect / OBB / pose with human-in-the-loop finetuning\n"
            "Built with PySide6",
        )

    # ==================== Navigation ====================

    def _set_project_chrome_visible(self, visible: bool):
        """Show or hide everything that only makes sense inside a project."""
        self._top_bar.setVisible(visible)
        self._tab_bar.setVisible(visible)
        self.close_project_act.setEnabled(visible)
        self._open_menu.setEnabled(visible)
        for sc in self._shortcuts:
            sc.setEnabled(visible)

    def _go_home(self):
        """Close the current project and return to the project picker."""
        if self._current_project is not None:
            busy = self._busy_page()
            if busy is not None:
                self._warn_busy(busy)
                return
            self._unload_all_pages()
            self._current_project = None

        self._set_project_chrome_visible(False)
        self.stack.setCurrentIndex(self.HOME_INDEX)
        self._refresh_projects()
        self.project_label.setText("")
        self.update_title()

    def _switch_tab(self, idx: int):
        if self._current_project is None:
            return
        self.stack.setCurrentIndex(idx + 1)   # index 0 is the home page
        if idx == 2:
            self.inspect_page.refresh()

    # ==================== Project management ====================

    def _refresh_projects(self, select: Optional[str] = None):
        """Reload the project list shown on the home page."""
        entries = [(folder, self.pm.project_task(folder))
                   for folder in self.pm.list_projects()]
        self.home_page.set_projects(entries, select=select)

    def _new_project(self):
        """Ask for a name and a task, then create and open ``<name>_<task>``.

        The task is requested here (and nowhere else) because it is immutable:
        the dataset layout, label format and weights all depend on it.
        """
        name, ok = QtWidgets.QInputDialog.getText(
            self, "New Project", "Project name:",
        )
        if not ok or not name.strip():
            return
        name = name.strip().replace(" ", "_")

        labels = [TASK_LABELS[t] for t in TASKS]
        label, ok = QtWidgets.QInputDialog.getItem(
            self, "New Project",
            "Task (cannot be changed later):",
            labels, labels.index(TASK_LABELS[TASK_OBB]), False,
        )
        if not ok:
            return
        task = TASKS[labels.index(label)]

        folder = project_folder_name(name, task)
        if folder in self.pm.list_projects():
            QtWidgets.QMessageBox.warning(
                self, "New Project", f"Project '{folder}' already exists.",
            )
            return

        self.pm.create_project(name, task)
        self._refresh_projects(select=folder)
        self._open_project(folder)

    def _busy_page(self) -> Optional[str]:
        """Name of a page currently running a background job, if any."""
        for label, page in (("Training", self.annotate_page),
                            ("Tracking", self.tracking_page)):
            if hasattr(page, "is_busy") and page.is_busy():
                return label
        return None

    def _warn_busy(self, busy: str):
        QtWidgets.QMessageBox.warning(
            self, "Project",
            f"{busy} is still running. Wait for it to finish before "
            f"leaving this project.",
        )

    def _unload_all_pages(self):
        """Drop every bit of state belonging to the project being left.

        Clips, annotations, tracks and training logs are all project-scoped;
        without this they would bleed into the next project.
        """
        self.annotate_page.unload_project()
        self.tracking_page.unload_project()
        self.train_page.reset_for_new_run()

    def _open_project(self, name: str):
        if not name:
            return
        if name == self._current_project:
            self._set_project_chrome_visible(True)
            self._switch_tab(self.DEFAULT_TAB)
            return

        # Switching mid-job would send the results into the wrong project.
        busy = self._busy_page()
        if busy is not None:
            self._warn_busy(busy)
            return

        # Leaving the previous project must not leak its state. Unloading is
        # idempotent, so this is also a safe no-op on the very first open.
        self._unload_all_pages()

        self._current_project = name
        task = self.pm.project_task(name)
        self.project_label.setText(f"{name}   [{task}]")
        # ensure_project() resolves the stored task instead of defaulting,
        # so opening a pose project can never re-tag it as obb.
        self.pm.ensure_project(name)
        cfg = self.pm.load_config(name)

        self.settings_page.load_config(cfg)
        self.annotate_page.apply_config(cfg)
        # Reloads straight from the new project's dataset_dir.
        self.inspect_page.unload_project()

        # Reveal the working pages and land on Settings.
        self._set_project_chrome_visible(True)
        self.tab_buttons[self.DEFAULT_TAB].setChecked(True)
        self._switch_tab(self.DEFAULT_TAB)

        self.update_title()
        self.statusBar().showMessage(f"Opened project '{name}' [{task}].", 4000)

    def project_config(self) -> dict:
        if self._current_project:
            return self.pm.load_config(self._current_project)
        return {}
    
    def project_dir(self) -> Optional[str]:
        """Directory of the open project, or None when none is open.

        Pages need this to write beside the project rather than wherever a file
        dialog last pointed: exports and the individuals mapping are
        project-scoped, and a mapping saved elsewhere is a mapping lost.
        """
        if self._current_project:
            return self.pm.project_dir(self._current_project)
        return None

    def _save_current_config(self):
        if not self._current_project:
            QtWidgets.QMessageBox.warning(self, "Save", "No project selected.")


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

    # ==================== Title & misc ====================

    def update_title(self):
        parts = ["HammerTrack"]
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