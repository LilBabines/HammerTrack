"""Individuals panel — the two strips that turn track fragments into animals.

Layout, top to bottom:

* **Tracks in frame** — one chip per track ID visible on the current frame,
  painted with its individual's colour, hollow when still unassigned. Clicking
  a chip selects that track on the canvas.
* **Assign selected** — ``New individual``, one row per existing individual,
  and ``None``. Enabled only while a track is selected, because every one of
  those actions is "do this to *the selected track*".

The widget owns no state: :meth:`refresh` redraws it from an
:class:`~.individuals.IndividualStore` plus the page's current selection. That
keeps the single source of truth in the store and makes the panel safe to
rebuild on every frame change.

Qt lives here so ``individuals.py`` can stay importable without a display.
"""

from functools import partial
from typing import Dict, Iterable, Optional, Sequence, Tuple

from PySide6 import QtCore, QtWidgets

from .individuals import Individual, IndividualStore

# Chips per row in the track strip. The side panel has a fixed width, so this
# is tuned to it rather than left to the layout.
_TRACK_COLUMNS = 4

_UNASSIGNED_BG = "#3a3a3a"
_UNASSIGNED_FG = "#c8c8c8"


def _bgr_to_css(color: Sequence[int]) -> str:
    """``(b, g, r)`` from the store to a CSS colour for Qt."""
    b, g, r = (int(c) for c in color)
    return f"rgb({r}, {g}, {b})"


def _readable_fg(color: Sequence[int]) -> str:
    """Black or white, whichever stays legible on ``color``."""
    b, g, r = (int(c) for c in color)
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    return "#000000" if luma > 140 else "#ffffff"


def _clear_layout(layout: QtWidgets.QLayout):
    """Remove and destroy every child widget of ``layout``."""
    while layout.count():
        item = layout.takeAt(0)
        child = item.widget()
        if child is not None:
            child.setParent(None)
            child.deleteLater()
        else:
            sub = item.layout()
            if sub is not None:
                _clear_layout(sub)


class IndividualsPanel(QtWidgets.QWidget):
    """Grouping UI. Emits intents; the page performs them on the store."""

    track_selected     = QtCore.Signal(int)   # track_id clicked in the strip
    create_requested   = QtCore.Signal()      # new individual from selection
    assign_requested   = QtCore.Signal(int)   # assign selection to this uid
    unassign_requested = QtCore.Signal()      # detach the selection
    rename_requested   = QtCore.Signal(int)   # uid
    delete_requested   = QtCore.Signal(int)   # uid

    def __init__(self, parent=None):
        super().__init__(parent)
        self._selected_tid: Optional[int] = None

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)

        # ---- Strip 1: tracks present on this frame ----
        self.tracks_box = QtWidgets.QGroupBox("Tracks in frame")
        tb = QtWidgets.QVBoxLayout(self.tracks_box)
        tb.setSpacing(4)
        self.tracks_grid = QtWidgets.QGridLayout()
        self.tracks_grid.setSpacing(3)
        tb.addLayout(self.tracks_grid)
        self.tracks_hint = QtWidgets.QLabel("No tracked detection here.")
        self.tracks_hint.setStyleSheet("color: #888; font-size: 11px;")
        self.tracks_hint.setWordWrap(True)
        tb.addWidget(self.tracks_hint)

        # ---- Strip 2: what to do with the selected track ----
        self.assign_box = QtWidgets.QGroupBox("Assign selected")
        ab = QtWidgets.QVBoxLayout(self.assign_box)
        ab.setSpacing(4)

        self.selection_lbl = QtWidgets.QLabel("Select a track first.")
        self.selection_lbl.setStyleSheet("color: #888; font-size: 11px;")
        self.selection_lbl.setWordWrap(True)
        ab.addWidget(self.selection_lbl)

        self.new_btn = QtWidgets.QPushButton("+ New individual")
        self.new_btn.clicked.connect(self.create_requested.emit)
        ab.addWidget(self.new_btn)

        self.individuals_layout = QtWidgets.QVBoxLayout()
        self.individuals_layout.setSpacing(3)
        ab.addLayout(self.individuals_layout)

        self.none_btn = QtWidgets.QPushButton("None (detach)")
        self.none_btn.setToolTip(
            "Leave this track unassigned: it is skipped on export."
        )
        self.none_btn.clicked.connect(self.unassign_requested.emit)
        ab.addWidget(self.none_btn)

        self.summary_lbl = QtWidgets.QLabel("")
        self.summary_lbl.setStyleSheet("color: #888; font-size: 11px;")
        self.summary_lbl.setWordWrap(True)

        outer.addWidget(self.tracks_box)
        outer.addWidget(self.assign_box)
        outer.addWidget(self.summary_lbl)

    # ---------------- Refresh ----------------

    def refresh(
        self,
        store: IndividualStore,
        frame_track_ids: Iterable[int],
        selected_tid: Optional[int] = None,
        conflicts: Optional[Dict[int, int]] = None,
    ):
        """Rebuild both strips from the store.

        ``frame_track_ids`` are the IDs visible on the current frame;
        ``conflicts`` is the ``{uid: n_frames}`` mapping from
        :meth:`IndividualStore.frame_conflicts`, or None to skip the badges.
        """
        self._selected_tid = (
            int(selected_tid) if selected_tid is not None and selected_tid >= 0
            else None
        )
        conflicts = conflicts or {}

        self._rebuild_tracks(store, sorted({int(t) for t in frame_track_ids}))
        self._rebuild_individuals(store, conflicts)
        self._update_selection_state(store)
        self._update_summary(store, conflicts)

    def _rebuild_tracks(self, store: IndividualStore, tids: Sequence[int]):
        _clear_layout(self.tracks_grid)
        self.tracks_hint.setVisible(not tids)

        for i, tid in enumerate(tids):
            ind = store.individual_of(tid)
            chip = QtWidgets.QPushButton(str(tid))
            chip.setCheckable(True)
            chip.setChecked(tid == self._selected_tid)
            chip.setFixedHeight(26)
            chip.setToolTip(
                f"Track {tid} → {ind.name}" if ind
                else f"Track {tid} — unassigned"
            )
            chip.setStyleSheet(self._chip_style(ind, tid == self._selected_tid))
            chip.clicked.connect(partial(self.track_selected.emit, tid))
            self.tracks_grid.addWidget(
                chip, i // _TRACK_COLUMNS, i % _TRACK_COLUMNS
            )

    @staticmethod
    def _chip_style(ind: Optional[Individual], selected: bool) -> str:
        if ind is not None:
            bg, fg = _bgr_to_css(ind.color), _readable_fg(ind.color)
        else:
            bg, fg = _UNASSIGNED_BG, _UNASSIGNED_FG
        # Magenta border echoes the canvas, where selection is magenta too.
        border = ("2px solid magenta" if selected else "1px solid #555")
        return (
            f"QPushButton {{ background: {bg}; color: {fg};"
            f" border: {border}; border-radius: 4px;"
            f" font-size: 11px; font-weight: bold; padding: 2px; }}"
        )

    def _rebuild_individuals(
        self, store: IndividualStore, conflicts: Dict[int, int],
    ):
        _clear_layout(self.individuals_layout)
        has_selection = self._selected_tid is not None
        owner = (store.individual_of(self._selected_tid)
                 if has_selection else None)

        for ind in store.all():
            row = QtWidgets.QHBoxLayout()
            row.setSpacing(3)

            is_owner = owner is not None and owner.uid == ind.uid
            n_conf = conflicts.get(ind.uid, 0)

            label = f"{ind.name}  ({len(ind.track_ids)})"
            if is_owner:
                label = f"✓ {label}"
            btn = QtWidgets.QPushButton(label)
            btn.setFixedHeight(24)
            btn.setEnabled(has_selection and not is_owner)
            tip = f"Move track {self._selected_tid} to {ind.name}" \
                if has_selection else ind.name
            if ind.notes:
                tip += f"\n{ind.notes}"
            if n_conf:
                tip += (f"\n⚠ {n_conf} frame(s) covered by two of its tracks"
                        " — highest confidence wins on export.")
            btn.setToolTip(tip)
            btn.setStyleSheet(self._row_style(ind, is_owner))
            btn.clicked.connect(partial(self.assign_requested.emit, ind.uid))
            row.addWidget(btn, stretch=1)

            if n_conf:
                dot = QtWidgets.QLabel("●")
                dot.setToolTip(f"{n_conf} conflicting frame(s)")
                dot.setStyleSheet("color: #ff4444; font-size: 15px;")
                dot.setFixedWidth(14)
                row.addWidget(dot)

            ren = QtWidgets.QToolButton()
            ren.setText("✎")
            ren.setToolTip("Rename / edit notes")
            ren.clicked.connect(partial(self.rename_requested.emit, ind.uid))
            row.addWidget(ren)

            dele = QtWidgets.QToolButton()
            dele.setText("✕")
            dele.setToolTip("Delete this individual (its tracks become free)")
            dele.clicked.connect(partial(self.delete_requested.emit, ind.uid))
            row.addWidget(dele)

            self.individuals_layout.addLayout(row)

    @staticmethod
    def _row_style(ind: Individual, is_owner: bool) -> str:
        bg, fg = _bgr_to_css(ind.color), _readable_fg(ind.color)
        border = "2px solid #ffffff" if is_owner else "1px solid #555"
        return (
            f"QPushButton {{ background: {bg}; color: {fg};"
            f" border: {border}; border-radius: 4px;"
            f" font-size: 11px; text-align: left; padding: 2px 6px; }}"
            f"QPushButton:disabled {{ background: {bg}; color: {fg}; }}"
        )

    def _update_selection_state(self, store: IndividualStore):
        has = self._selected_tid is not None
        self.new_btn.setEnabled(has)
        self.none_btn.setEnabled(has)

        if not has:
            self.selection_lbl.setText(
                "Select a track — on the canvas or in the strip above."
            )
            return
        ind = store.individual_of(self._selected_tid)
        self.none_btn.setEnabled(ind is not None)
        if ind is None:
            self.selection_lbl.setText(
                f"Track {self._selected_tid} — unassigned"
            )
        else:
            self.selection_lbl.setText(
                f"Track {self._selected_tid} → {ind.name}"
            )

    def _update_summary(
        self, store: IndividualStore, conflicts: Dict[int, int],
    ):
        n_ind = len(store)
        if not n_ind:
            self.summary_lbl.setText("No individual yet.")
            return
        n_tracks = sum(len(i.track_ids) for i in store.all())
        text = f"{n_ind} individual(s), {n_tracks} track(s) assigned."
        if conflicts:
            text += f"  ⚠ {len(conflicts)} with frame conflicts."
        self.summary_lbl.setText(text)


def ask_rename(
    parent: QtWidgets.QWidget, ind: Individual,
) -> Optional[Tuple[str, str]]:
    """Prompt for a new name and notes. Returns None when cancelled.

    Notes are free text on purpose: they are the ecology field (a scar, an
    estimated size) that no schema here should try to model.
    """
    dlg = QtWidgets.QDialog(parent)
    dlg.setWindowTitle(f"Edit {ind.name}")
    form = QtWidgets.QFormLayout(dlg)

    name_edit = QtWidgets.QLineEdit(ind.name)
    notes_edit = QtWidgets.QPlainTextEdit(ind.notes)
    notes_edit.setFixedHeight(70)
    form.addRow("Name", name_edit)
    form.addRow("Notes", notes_edit)

    buttons = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.StandardButton.Ok
        | QtWidgets.QDialogButtonBox.StandardButton.Cancel
    )
    buttons.accepted.connect(dlg.accept)
    buttons.rejected.connect(dlg.reject)
    form.addRow(buttons)

    if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
        return None
    return name_edit.text(), notes_edit.toPlainText()
