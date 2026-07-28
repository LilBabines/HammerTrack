"""
HomePage — the landing view shown when no project is open.

This is where a project is picked or created; the rest of the app stays
hidden until then. It owns the project selector that used to live in the
window's top bar, because a project has to exist before any other page has
anything meaningful to show.

The page never touches the filesystem itself: the launcher feeds it the
project list via :meth:`set_projects` and reacts to the two signals below.
"""

from typing import Optional, Sequence, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

from ..tasks import TASK_LABELS


class HomePage(QtWidgets.QWidget):
    """Centered project picker.

    Signals
    -------
    project_selected(str)
        The user opened a project; the payload is its folder name.
    new_project_requested()
        The user asked to create a project; the launcher runs the dialogs.
    """

    project_selected = QtCore.Signal(str)
    new_project_requested = QtCore.Signal()

    #: Fixed width of the centered card, in pixels.
    CARD_WIDTH = 460

    def __init__(self, parent=None):
        super().__init__(parent)
        # Tracked explicitly: QWidget.isVisible() is False until the window is
        # shown, so it cannot be used to decide whether a project is selected.
        self._has_projects = False
        self._build_ui()

    # ==================== UI ====================

    def _build_ui(self):
        card = QtWidgets.QWidget()
        card.setFixedWidth(self.CARD_WIDTH)
        v = QtWidgets.QVBoxLayout(card)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(14)

        title = QtWidgets.QLabel("HammerTrack")
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        title_font = title.font()
        title_font.setPointSize(title_font.pointSize() + 10)
        title_font.setBold(True)
        title.setFont(title_font)

        subtitle = QtWidgets.QLabel("Open a project to get started")
        subtitle.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #888;")

        # --- Project list (the selector, relocated from the top bar) ---
        self.project_list = QtWidgets.QListWidget()
        self.project_list.setMinimumHeight(240)
        self.project_list.itemSelectionChanged.connect(self._on_selection_changed)
        self.project_list.itemDoubleClicked.connect(self._on_double_click)

        self.empty_label = QtWidgets.QLabel(
            "No project yet — create one to begin."
        )
        self.empty_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setStyleSheet("color: #888; padding: 8px;")
        self.empty_label.setVisible(False)

        # --- Actions ---
        self.open_btn = QtWidgets.QPushButton("Open project")
        self.open_btn.setFixedHeight(38)
        self.open_btn.setDefault(True)
        self.open_btn.setEnabled(False)
        self.open_btn.clicked.connect(self._emit_selected)

        self.new_btn = QtWidgets.QPushButton("New project...")
        self.new_btn.setFixedHeight(38)
        self.new_btn.clicked.connect(self.new_project_requested.emit)

        self.refresh_btn = QtWidgets.QPushButton("Refresh")
        self.refresh_btn.setFixedHeight(38)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(8)
        btn_row.addWidget(self.new_btn, stretch=1)
        btn_row.addWidget(self.refresh_btn, stretch=0)

        hint = QtWidgets.QLabel(
            "A project is bound to one YOLO task and cannot be changed later."
        )
        hint.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #777; font-size: 11px;")

        v.addWidget(title)
        v.addWidget(subtitle)
        v.addSpacing(6)
        v.addWidget(self.project_list)
        v.addWidget(self.empty_label)
        v.addWidget(self.open_btn)
        v.addLayout(btn_row)
        v.addWidget(hint)

        # Center the card both ways
        outer = QtWidgets.QVBoxLayout(self)
        outer.addStretch(1)
        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        row.addWidget(card)
        row.addStretch(1)
        outer.addLayout(row)
        outer.addStretch(1)

    # ==================== Population ====================

    def set_projects(self, entries: Sequence[Tuple[str, str]],
                     select: Optional[str] = None):
        """Fill the list with ``(folder, task)`` pairs.

        ``select`` pre-selects a folder when present, which is what makes a
        freshly created project land already highlighted.
        """
        self.project_list.blockSignals(True)
        self.project_list.clear()

        for folder, task in entries:
            item = QtWidgets.QListWidgetItem(folder)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, folder)
            item.setToolTip(TASK_LABELS.get(task, task))
            # Keep the task visible at a glance: it drives everything
            # downstream and cannot be changed after creation.
            item.setText(f"{folder}          [{task}]")
            self.project_list.addItem(item)

        self.project_list.blockSignals(False)

        self._has_projects = bool(entries)
        self.project_list.setVisible(self._has_projects)
        self.empty_label.setVisible(not self._has_projects)

        # Nothing is selected by default: the user has to pick a project,
        # which also guarantees the first click is a real selection change.
        if select is not None:
            self.select_project(select)
        else:
            self.project_list.setCurrentRow(-1)
            self._on_selection_changed()

    def select_project(self, folder: str):
        for i in range(self.project_list.count()):
            item = self.project_list.item(i)
            if item.data(QtCore.Qt.ItemDataRole.UserRole) == folder:
                self.project_list.setCurrentRow(i)
                self._on_selection_changed()
                return
        self.project_list.setCurrentRow(-1)
        self._on_selection_changed()

    def selected_project(self) -> Optional[str]:
        """Folder name of the highlighted project, or None when none is."""
        if not self._has_projects:
            return None
        item = self.project_list.currentItem()
        if item is None:
            return None
        return item.data(QtCore.Qt.ItemDataRole.UserRole)

    # ==================== Reactions ====================

    def _on_selection_changed(self):
        self.open_btn.setEnabled(self.selected_project() is not None)

    def _on_double_click(self, _item: QtWidgets.QListWidgetItem):
        self._emit_selected()

    def _emit_selected(self):
        folder = self.selected_project()
        if folder:
            self.project_selected.emit(folder)

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        """Enter opens the highlighted project."""
        if event.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
            self._emit_selected()
            event.accept()
            return
        super().keyPressEvent(event)