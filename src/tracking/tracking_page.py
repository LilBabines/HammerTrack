"""Tracking page — BoxMOT BoTSORT with AABB detections.

OBB detections are converted to axis-aligned bounding boxes for the tracker,
then track IDs are mapped back onto the original OBBs. Trajectories are read
from the tracker's internal STrack ``history_observations``.

Exports:
* ``per_frame/frame_XXXXXX.txt`` — one line per detection
* ``per_track/track_XXXX.json``  — full history per track ID
* ``cmc_transforms.json``        — per-frame camera-motion-compensation matrix
"""

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PySide6 import QtCore, QtWidgets

from ..canvas import AnnotationCanvas
from .tracking_helpers import (
    build_tracker,
    draw_tracked_annotations,
    obb_to_xywhr,
    update_strack_bbox,
)
from .individuals import (
    IndividualStore, individuals_dir, individuals_path,
)
from .individuals_panel import IndividualsPanel, ask_rename
from .tracking_workers import TrackingStepWorker, VideoExportWorker
from ..tasks import normalize_task
from ..utils import (
    OBBOX, FrameSource, VideoSource, ImageFolderSource,
    ensure_bgr_u8,
)


class TrackingPage(QtWidgets.QWidget):
    """Tracking tab: load a source, step the BoTSORT tracker, edit IDs,
    export tracked video and per-frame/per-track data."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._launcher = None

        # ==================== Source / playback state ====================
        self.source: Optional[FrameSource] = None
        self.total_frames: int = 0
        self.current_idx: int = 0
        self.current_frame_bgr: Optional[np.ndarray] = None
        self.play_timer = QtCore.QTimer(self)
        self.play_timer.timeout.connect(self._on_play_tick)
        self.playing = False
        self._play_fps = 25.0
        self._play_anchor_time = 0.0
        self._play_anchor_idx = 0
        self._play_busy = False

        # ==================== Tracker state ====================
        self.tracker = None
        self.last_tracked_idx: int = -1
        self.track_cache: Dict[int, List[OBBOX]] = {}
        self.traj_snapshots: Dict[int, Dict[int, List[Tuple[float, float]]]] = {}
        self.cmc_cache: Dict[int, Optional[np.ndarray]] = {}

        # ==================== Selection / editing state ====================
        self.selected_idx: Optional[int] = None
        self.mode = "select"          # "select" | "edit"
        self.dragging = False
        self.drag_start_img: Optional[Tuple[float, float]] = None
        self.orig_poly: Optional[np.ndarray] = None
        self.vertex_drag_idx: Optional[int] = None

        # Pending ID changes (old_tid → new_tid) for sync with tracker
        self._pending_id_changes: Dict[int, int] = {}
        # Track IDs whose box the user moved/reshaped since the last step.
        # Only these get their Kalman state rewritten: touching untouched
        # tracks on every step is what used to break ID continuity.
        self._edited_tids: set = set()

        # ==================== Individuals (identity layer) ====================
        # Grouping of tracker fragments into real animals. Per clip, because
        # raw track IDs mean nothing across clips.
        self.individuals = IndividualStore()
        self._export_dir: Optional[str] = None

        # Derived views of the caches above. Rebuilding them costs a pass over
        # every tracked frame, which is far too slow to redo on each playback
        # tick, so they are memoised and invalidated explicitly.
        self._frames_by_track_cache: Optional[Dict[int, set]] = None
        self._conflicts_cache: Optional[Dict[int, int]] = None
        self._panel_state: Optional[tuple] = None

        # Run-state flags (block UI while threads work)
        self._tracking_running = False
        self._exporting = False

        self._build_ui()
        self._wire_signals()

    # ==================== UI construction ====================

    def _build_ui(self):
        # ---- Canvas (reused from annotate_page) ----
        self.canvas = AnnotationCanvas(self)
        self.canvas.mouse_pressed.connect(self._on_canvas_mouse_press)
        self.canvas.mouse_moved.connect(self._on_canvas_mouse_move)
        self.canvas.mouse_released.connect(self._on_canvas_mouse_release)

        # ---- Frame slider ----
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.sliderReleased.connect(self._on_slider)

        # ---- Buttons / spinboxes ----
        self.open_video_btn  = QtWidgets.QPushButton("Open video")
        self.open_images_btn = QtWidgets.QPushButton("Open image folder")
        self.prev_btn        = QtWidgets.QPushButton("⟸ Prev")
        self.next_btn        = QtWidgets.QPushButton("Next ⟹")
        self.play_btn        = QtWidgets.QPushButton("Play ▶")
        self.pause_btn       = QtWidgets.QPushButton("Pause ⏸")
        self.zoom_in_btn     = QtWidgets.QPushButton("Zoom +")
        self.zoom_out_btn    = QtWidgets.QPushButton("Zoom −")
        self.zoom_fit_btn    = QtWidgets.QPushButton("Fit")

        self.step_btn = QtWidgets.QPushButton("Step Tracker ▶▶")
        self.step_btn.setFixedHeight(36)
        self.step_btn.setEnabled(False)

        self.reset_tracker_btn = QtWidgets.QPushButton("Reset Tracker")

        self.export_btn = QtWidgets.QPushButton("Export Video 🎬")
        self.export_btn.setFixedHeight(36)
        self.export_btn.setEnabled(False)

        self.export_data_btn = QtWidgets.QPushButton("Export Data 📊")
        self.export_data_btn.setToolTip(
            "Export per_frame/*.txt + per_track/*.json (no images)"
        )
        self.export_data_btn.setFixedHeight(36)
        self.export_data_btn.setEnabled(False)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Idle")
        self.progress_bar.setFixedHeight(18)

        self.edit_btn   = QtWidgets.QPushButton("Edit (E)")
        self.delete_btn = QtWidgets.QPushButton("Delete (Del)")

        self.id_spin = QtWidgets.QSpinBox()
        self.id_spin.setPrefix("Track ID: ")
        self.id_spin.setRange(-1, 99999)
        self.id_spin.setValue(-1)
        self.id_spin.setEnabled(False)

        self.conf_spin = QtWidgets.QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 0.99)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.5)
        self.conf_spin.setPrefix("conf=")

        self.frame_skip_spin = QtWidgets.QSpinBox()
        self.frame_skip_spin.setPrefix("Track every ")
        self.frame_skip_spin.setSuffix(" frames")
        self.frame_skip_spin.setRange(1, 30)
        self.frame_skip_spin.setValue(1)

        self.show_trails_chk = QtWidgets.QCheckBox("Show trajectories")
        self.show_trails_chk.setChecked(True)

        self.trail_len_spin = QtWidgets.QSpinBox()
        self.trail_len_spin.setPrefix("Trail: ")
        self.trail_len_spin.setSuffix(" frames")
        self.trail_len_spin.setRange(5, 9999)
        self.trail_len_spin.setValue(60)

        # ---- Individuals ----
        self.individuals_panel = IndividualsPanel()

        self.only_assigned_chk = QtWidgets.QCheckBox("Assigned tracks only")
        self.only_assigned_chk.setToolTip(
            "Leave unassigned fragments out of per_frame/ and per_track/.\n"
            "individuals/ never contains them: they belong to no animal."
        )

        # ---- Layout ----
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(6)
        lv.addWidget(self.canvas, stretch=1)
        lv.addWidget(self.frame_slider)

        content = QtWidgets.QHBoxLayout()
        content.setContentsMargins(0, 0, 0, 0)
        content.setSpacing(10)
        content.addWidget(left, stretch=1)
        content.addWidget(self._build_side_panel(), stretch=0)

        page = QtWidgets.QVBoxLayout(self)
        page.setContentsMargins(8, 8, 8, 8)
        page.setSpacing(8)
        page.addLayout(content, stretch=1)
        page.addWidget(self._build_transport())

    def _build_side_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(panel)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(8)
        # No AlignTop here any more: the individuals area carries the stretch,
        # so it grows with the window instead of a trailing spacer doing it.

        src = QtWidgets.QGroupBox("Source")
        sl = QtWidgets.QVBoxLayout(src)
        sl.addWidget(self.open_video_btn)
        sl.addWidget(self.open_images_btn)

        trk = QtWidgets.QGroupBox("Tracking")
        tl = QtWidgets.QVBoxLayout(trk)
        tl.addWidget(self.step_btn)
        tl.addWidget(self.progress_bar)
        tl.addWidget(self.conf_spin)
        tl.addWidget(self.frame_skip_spin)
        tl.addWidget(self.reset_tracker_btn)

        vis = QtWidgets.QGroupBox("Visualisation")
        vl = QtWidgets.QVBoxLayout(vis)
        vl.addWidget(self.show_trails_chk)
        vl.addWidget(self.trail_len_spin)

        edit = QtWidgets.QGroupBox("Edit selected")
        el = QtWidgets.QVBoxLayout(edit)
        el.addWidget(self.edit_btn)
        el.addWidget(self.id_spin)
        el.addWidget(self.delete_btn)

        # A long session can hold dozens of individuals, so the panel scrolls
        # instead of pushing Export off the bottom of the window.
        ind_scroll = QtWidgets.QScrollArea()
        ind_scroll.setWidget(self.individuals_panel)
        ind_scroll.setWidgetResizable(True)
        ind_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        ind_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        ind_scroll.setMinimumHeight(220)

        exp = QtWidgets.QGroupBox("Export")
        xl = QtWidgets.QVBoxLayout(exp)
        xl.addWidget(self.only_assigned_chk)
        xl.addWidget(self.export_btn)
        xl.addWidget(self.export_data_btn)

        for grp in (src, trk, vis, edit):
            v.addWidget(grp)
        v.addWidget(ind_scroll, stretch=1)
        v.addWidget(exp)

        # The scroll area reports a small width hint, and horizontal scrolling
        # is off, so the floor has to be set here or the chips get clipped.
        panel.setMinimumWidth(250)
        panel.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        return panel

    def _build_transport(self) -> QtWidgets.QWidget:
        bar = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(bar)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(10)
        h.addStretch(1)
        for w in (self.prev_btn, self.play_btn, self.pause_btn, self.next_btn):
            h.addWidget(w)
        h.addSpacing(20)
        for w in (self.zoom_out_btn, self.zoom_in_btn, self.zoom_fit_btn):
            h.addWidget(w)
        h.addStretch(1)
        return bar

    def _wire_signals(self):
        self.open_video_btn.clicked.connect(self._open_video)
        self.open_images_btn.clicked.connect(self._open_folder)
        self.prev_btn.clicked.connect(self.prev_frame)
        self.next_btn.clicked.connect(self.next_frame)
        self.play_btn.clicked.connect(self.play)
        self.pause_btn.clicked.connect(self.pause)
        self.zoom_in_btn.clicked.connect(self._zoom_in)
        self.zoom_out_btn.clicked.connect(self._zoom_out)
        self.zoom_fit_btn.clicked.connect(self.canvas.zoom_fit)
        self.step_btn.clicked.connect(self._step_tracker)
        self.reset_tracker_btn.clicked.connect(self._reset_tracker)
        self.edit_btn.clicked.connect(self._toggle_edit)
        self.delete_btn.clicked.connect(self._delete_selected)
        self.export_btn.clicked.connect(self._export_video)
        self.export_data_btn.clicked.connect(self._export_data)
        self.show_trails_chk.toggled.connect(self._on_visu_changed)
        self.trail_len_spin.valueChanged.connect(self._on_visu_changed)
        self.id_spin.valueChanged.connect(self._on_id_spin_changed)

        ip = self.individuals_panel
        ip.track_selected.connect(self._select_track_id)
        ip.create_requested.connect(self._individual_create)
        ip.assign_requested.connect(self._individual_assign)
        ip.unassign_requested.connect(self._individual_unassign)
        ip.rename_requested.connect(self._individual_rename)
        ip.delete_requested.connect(self._individual_delete)

    # ==================== Launcher hook ====================

    def set_launcher(self, launcher):
        self._launcher = launcher

    def _status(self, msg: str):
        if self._launcher:
            self._launcher.statusBar().showMessage(msg, 5000)

    def _task(self) -> str:
        """Project task; decides the tracker's detection layout."""
        return normalize_task(self._cfg().get("task_type"))

    def _cfg(self) -> dict:
        return self._launcher.project_config() if self._launcher else {}

    # ==================== Individuals — paths & persistence ====================

    def _clip_export_dir(self) -> Optional[str]:
        """``<project>/exports/<clip>/`` for the loaded clip.

        The clip folder is the source's stem, so re-opening the same video
        finds its own mapping again. Returns None when there is no project or
        no source, in which case nothing is persisted rather than being written
        to an arbitrary place.
        """
        if self.source is None or self._launcher is None:
            return None
        project_dir = getattr(self._launcher, "project_dir", lambda: None)()
        if not project_dir:
            return None
        clip = Path(getattr(self.source, "path", "") or self.source.name()).stem
        if not clip:
            return None
        return str(Path(project_dir) / "exports" / clip)

    def _save_individuals(self):
        """Persist after every mutation: grouping is slow, redoing it is worse."""
        if not self._export_dir:
            return
        try:
            self.individuals.save(individuals_path(self._export_dir))
        except OSError as exc:
            self._status(f"Could not save individuals: {exc}")

    def _load_individuals(self):
        """Load this clip's mapping, reporting IDs that no longer exist."""
        self.individuals.clear()
        self._invalidate_grouping_views()
        if not self._export_dir:
            self._status(
                "No project open: individuals will not be saved to disk."
            )
            return
        path = individuals_path(self._export_dir)
        if not self.individuals.load(path):
            return
        stale = self.individuals.stale_track_ids(self._all_cached_track_ids())
        msg = f"Loaded {len(self.individuals)} individual(s)."
        if stale:
            # Not dropped: the tracks reappear as soon as the clip is re-tracked
            # with the same settings, and silently discarding the grouping would
            # be worse than carrying a stale entry.
            msg += (f" {len(stale)} assigned track(s) not in the current cache"
                    " — re-run the tracker to see them.")
        self._status(msg)

    # ==================== Individuals — derived views ====================

    def _all_cached_track_ids(self) -> List[int]:
        return sorted(self._frames_by_track().keys())

    def _invalidate_track_views(self):
        """Call when ``track_cache`` changed (new frames, IDs, deletions)."""
        self._frames_by_track_cache = None
        self._conflicts_cache = None
        self._panel_state = None

    def _invalidate_grouping_views(self):
        """Call when the grouping changed but the detections did not."""
        self._conflicts_cache = None
        self._panel_state = None

    def _frames_by_track(self) -> Dict[int, set]:
        """``{track_id: {frames}}`` over the whole cache, memoised."""
        if self._frames_by_track_cache is None:
            fbt: Dict[int, set] = defaultdict(set)
            for frame_idx, annots in self.track_cache.items():
                for b in annots:
                    if not b.deleted and b.track_id >= 0:
                        fbt[int(b.track_id)].add(int(frame_idx))
            self._frames_by_track_cache = dict(fbt)
        return self._frames_by_track_cache

    def _conflicts(self) -> Dict[int, int]:
        """``{uid: n_frames}`` where two tracks of one animal overlap."""
        if self._conflicts_cache is None:
            self._conflicts_cache = self.individuals.frame_conflicts(
                self._frames_by_track()
            )
        return self._conflicts_cache

    def _frame_track_ids(self) -> List[int]:
        """Track IDs visible on the current frame."""
        annots = self.track_cache.get(self.current_idx, [])
        return sorted({int(b.track_id) for b in annots
                       if not b.deleted and b.track_id >= 0})

    def _individual_label(self, track_id: int) -> Optional[str]:
        ind = self.individuals.individual_of(track_id)
        return ind.name if ind else None

    # ==================== Individuals — selection bridge ====================

    def _selected_track_id(self) -> Optional[int]:
        """Track ID of the selected annotation, or None."""
        annots = self.track_cache.get(self.current_idx, [])
        if self.selected_idx is None or not (
            0 <= self.selected_idx < len(annots)
        ):
            return None
        tid = int(annots[self.selected_idx].track_id)
        return tid if tid >= 0 else None

    def _select_track_id(self, track_id: int):
        """Select the annotation carrying ``track_id`` on the current frame."""
        annots = self.track_cache.get(self.current_idx, [])
        for i, b in enumerate(annots):
            if not b.deleted and int(b.track_id) == int(track_id):
                self.selected_idx = i
                self._update_id_spin()
                self._redraw()
                self._refresh_individuals()
                return

    def _refresh_individuals(self, force: bool = False):
        """Redraw the panel, skipping the rebuild when nothing visible moved.

        ``_read_frame`` runs on every playback tick, and rebuilding a few dozen
        buttons at 25 fps is wasted work; the state key makes that a no-op.
        """
        tids = self._frame_track_ids()
        selected = self._selected_track_id()
        conflicts = self._conflicts()
        state = (
            tuple(tids), selected,
            tuple(sorted(conflicts.items())),
            tuple((i.uid, i.name, tuple(i.track_ids), i.notes)
                  for i in self.individuals.all()),
        )
        if not force and state == self._panel_state:
            return
        self._panel_state = state
        self.individuals_panel.refresh(
            self.individuals, tids, selected, conflicts
        )

    # ==================== Individuals — actions ====================

    def _individual_create(self):
        tid = self._selected_track_id()
        if tid is None:
            return
        ind = self.individuals.create([tid])
        self._after_grouping_change(f"Track {tid} → new individual {ind.name}.")

    def _individual_assign(self, uid: int):
        tid = self._selected_track_id()
        if tid is None:
            return
        if not self.individuals.assign(tid, int(uid)):
            return
        ind = self.individuals.get(int(uid))
        name = ind.name if ind else uid
        self._after_grouping_change(f"Track {tid} → {name}.")

    def _individual_unassign(self):
        tid = self._selected_track_id()
        if tid is None or not self.individuals.unassign(tid):
            return
        self._after_grouping_change(f"Track {tid} detached.")

    def _individual_rename(self, uid: int):
        ind = self.individuals.get(int(uid))
        if ind is None:
            return
        result = ask_rename(self, ind)
        if result is None:
            return
        name, notes = result
        if name.strip():
            self.individuals.rename(int(uid), name)
        self.individuals.set_notes(int(uid), notes)
        self._after_grouping_change(f"Renamed to {ind.name}.")

    def _individual_delete(self, uid: int):
        ind = self.individuals.get(int(uid))
        if ind is None:
            return
        if ind.track_ids:
            reply = QtWidgets.QMessageBox.question(
                self, "Delete individual",
                f"Delete '{ind.name}'?\n\n"
                f"Its {len(ind.track_ids)} track(s) become unassigned; "
                "no detection is lost.",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No,
            )
            if reply != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        name = ind.name
        self.individuals.delete(int(uid))
        self._after_grouping_change(f"Deleted {name}.")

    def _after_grouping_change(self, message: str):
        """Shared tail of every grouping action: persist, recolour, report."""
        self._invalidate_grouping_views()
        self._save_individuals()
        self._redraw()
        self._refresh_individuals(force=True)
        self._status(message)

    # ==================== Source loading ====================

    def _open_video(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Video", "",
            "Video (*.mp4 *.avi *.mov *.mkv);;All (*)",
        )
        if p:
            self._load_source(VideoSource(p))

    def _open_folder(self):
        f = QtWidgets.QFileDialog.getExistingDirectory(self, "Open Image Folder")
        if f:
            self._load_source(ImageFolderSource(f))

    def _load_source(self, src):
        if self.source:
            try:
                self.source.close()
            except Exception:
                pass
        self.source = src
        self.total_frames = src.count()
        self.current_idx = 0
        self.current_frame_bgr = None
        self.track_cache.clear()
        self.selected_idx = None
        self.mode = "select"
        self._pending_id_changes.clear()
        self.frame_slider.setRange(0, max(0, self.total_frames - 1))
        self.frame_slider.setValue(0)
        # Nothing is tracked yet, so there is no mapping to protect: reset
        # without the stale-mapping prompt, then load the clip's own file.
        self._reset_tracker(confirm=False, quiet=True)
        self._export_dir = self._clip_export_dir()
        self._load_individuals()
        self._read_frame(0)
        self._status(f"Loaded: {src.name()} | {self.total_frames} frames")

    # ==================== Frame read / display ====================

    def _read_frame(self, idx: int):
        if not self.source:
            return
        idx = max(0, min(idx, self.total_frames - 1))
        frame = self.source.read(idx)
        if frame is None:
            return
        self.current_idx = idx
        self.current_frame_bgr = ensure_bgr_u8(frame)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(idx)
        self.frame_slider.blockSignals(False)
        self.selected_idx = None
        self._update_step_btn()
        self._update_export_btns()
        self._update_id_spin()
        self._redraw()
        self._refresh_individuals()
        if self._launcher:
            self._launcher.update_title()

    def _redraw(self):
        if self.current_frame_bgr is None:
            return
        annots = self.track_cache.get(self.current_idx, [])
        trajectories = self.traj_snapshots.get(self.current_idx, {})
        annotated = draw_tracked_annotations(
            self.current_frame_bgr, annots, self.selected_idx,
            trajectories,
            trail_length=self.trail_len_spin.value(),
            show_trails=self.show_trails_chk.isChecked(),
            # Every fragment of one animal shares its colour; unassigned ones
            # fall back to their per-track hue and so stay distinguishable.
            color_of=self.individuals.color_for_track,
            label_of=self._individual_label,
        )
        self.canvas.set_frame(annotated)

    def _on_visu_changed(self, *_):
        self._redraw()

    # ==================== Navigation ====================

    def prev_frame(self):
        if self.source and self.current_idx > 0:
            self.pause()
            self._read_frame(self.current_idx - 1)

    def next_frame(self):
        if self.source and self.current_idx < self.total_frames - 1:
            self.pause()
            self._read_frame(self.current_idx + 1)

    def _on_slider(self):
        if self.source:
            self.pause()
            self._read_frame(self.frame_slider.value())

    def play(self):
        if not self.source or self.playing:
            return
        self._play_fps = max(1.0, float(self.source.fps() or 25.0))
        self._play_anchor_time = time.monotonic()
        self._play_anchor_idx = self.current_idx
        self._play_busy = False
        self.playing = True
        self.canvas.set_fast_scaling(True)
        self.play_timer.start(max(5, int(1000 / self._play_fps)))

    def pause(self):
        if self.playing:
            self.play_timer.stop()
            self.playing = False
            self.canvas.set_fast_scaling(False)
            self._redraw()

    def _on_play_tick(self):
        """Advance playback on wall-clock time, dropping late frames."""
        if self._play_busy:
            return
        self._play_busy = True
        try:
            elapsed = time.monotonic() - self._play_anchor_time
            target = self._play_anchor_idx + int(elapsed * self._play_fps)
            target = max(self.current_idx + 1, target)
            if target >= self.total_frames:
                self.pause()
                return
            self._read_frame(target)
        finally:
            self._play_busy = False

    # ==================== Zoom ====================

    def _zoom_in(self):
        self.canvas.zoom_step(+1)

    def _zoom_out(self):
        self.canvas.zoom_step(-1)

    # ==================== Tracker — sync edits ====================

    def _sync_edits_to_tracker(self):
        """Push user edits (delete / move / ID change) back into the
        BoxMOT tracker's internal STracks so the next ``.update()`` call
        starts from the corrected state."""
        if self.tracker is None or self.last_tracked_idx < 0:
            return

        annots = self.track_cache.get(self.last_tracked_idx, [])

        deleted_tids = {b.track_id for b in annots
                        if b.deleted and b.track_id >= 0}
        if not (self._pending_id_changes or deleted_tids or self._edited_tids):
            return   # nothing was edited: leave the tracker alone

        # 1. Apply pending ID changes
        if self._pending_id_changes:
            for attr in ("active_tracks", "lost_stracks"):
                pool = getattr(self.tracker, attr, None)
                if pool is None:
                    continue
                for st in pool:
                    old_tid = int(getattr(st, "id", -1))
                    if old_tid in self._pending_id_changes:
                        try:
                            st.id = self._pending_id_changes[old_tid]
                        except Exception:
                            pass
            self._pending_id_changes.clear()

        # 2. Remove deleted tracks
        if deleted_tids:
            for attr in ("active_tracks", "lost_stracks"):
                pool = getattr(self.tracker, attr, None)
                if pool is None:
                    continue
                to_rm = [st for st in pool
                         if int(getattr(st, "id", -1)) in deleted_tids]
                for st in to_rm:
                    pool.remove(st)

        # 3. Update positions for surviving tracks
        active_map: Dict[int, Tuple[float, float, float, float]] = {}
        for b in annots:
            if (not b.deleted and b.track_id >= 0
                    and b.track_id in self._edited_tids):
                pts = b.poly.reshape(-1, 2)
                x1, y1 = float(pts[:, 0].min()), float(pts[:, 1].min())
                x2, y2 = float(pts[:, 0].max()), float(pts[:, 1].max())
                active_map[b.track_id] = (x1, y1, x2, y2)

        if active_map:
            for attr in ("active_tracks", "lost_stracks"):
                pool = getattr(self.tracker, attr, None)
                if pool is None:
                    continue
                for st in pool:
                    tid = int(getattr(st, "id", -1))
                    if tid in active_map:
                        try:
                            update_strack_bbox(st, *active_map[tid])
                        except Exception:
                            pass

        self._edited_tids.clear()

    # ==================== Tracker — step ====================

    def _update_step_btn(self):
        self.step_btn.setEnabled(
            self.source is not None
            and self.current_idx > self.last_tracked_idx
        )

    def _update_export_btns(self):
        ok = (
            bool(self.track_cache)
            and not self._tracking_running
            and not self._exporting
        )
        self.export_btn.setEnabled(ok)
        self.export_data_btn.setEnabled(ok)

    def unload_project(self):
        """Release the loaded clip and every tracker cache.

        Tracks, trajectories and CMC snapshots are all tied to one clip inside
        one project, so switching project must wipe them rather than let them
        be exported into the new project's folder.
        """
        self.pause()
        if self.source:
            try:
                self.source.close()
            except Exception:
                pass
        self.source = None

        self.total_frames = 0
        self.current_idx = 0
        self.current_frame_bgr = None

        self.tracker = None
        self.last_tracked_idx = -1
        self.track_cache.clear()
        self.traj_snapshots.clear()
        self.cmc_cache.clear()
        self._pending_id_changes.clear()
        self._edited_tids.clear()

        self.selected_idx = None
        self.mode = "select"
        self.dragging = False
        self.drag_start_img = None
        self.orig_poly = None
        self.vertex_drag_idx = None

        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.setValue(0)
        self.frame_slider.blockSignals(False)

        self.individuals.clear()
        self._export_dir = None
        self._invalidate_track_views()

        self.canvas.set_frame(None)
        self._update_step_btn()
        self._update_export_btns()
        self._refresh_individuals(force=True)

    def is_busy(self) -> bool:
        """True while a tracking step or a video export is still running."""
        return bool(self._tracking_running or self._exporting)

    def _reset_tracker(self, *, confirm: bool = True, quiet: bool = False):
        """Drop the tracker and every cache tied to its raw IDs.

        ``confirm`` is keyword-only on purpose: ``clicked`` would otherwise
        hand its ``checked`` boolean straight into it and silently skip the
        prompt.
        """
        n_assigned = sum(len(i.track_ids) for i in self.individuals.all())

        if confirm and n_assigned:
            # A reset makes BoxMOT reallocate IDs from 1, so track 7 after the
            # reset is a different animal than track 7 before it. Keeping the
            # mapping is legitimate (re-tracking with identical settings
            # reproduces the IDs), so the choice is the user's.
            box = QtWidgets.QMessageBox(self)
            box.setWindowTitle("Reset tracker")
            box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
            box.setText(
                f"{len(self.individuals)} individual(s) reference "
                f"{n_assigned} track(s)."
            )
            box.setInformativeText(
                "Resetting reallocates raw track IDs, so the current grouping "
                "may end up pointing at the wrong animals.\n\n"
                "Clear it, or keep it and re-track with the same settings?"
            )
            clear_btn = box.addButton(
                "Reset and clear grouping",
                QtWidgets.QMessageBox.ButtonRole.DestructiveRole,
            )
            keep_btn = box.addButton(
                "Reset and keep grouping",
                QtWidgets.QMessageBox.ButtonRole.AcceptRole,
            )
            cancel_btn = box.addButton(
                QtWidgets.QMessageBox.StandardButton.Cancel
            )
            box.setDefaultButton(keep_btn)
            box.exec()

            clicked = box.clickedButton()
            if clicked is cancel_btn:
                return
            if clicked is clear_btn:
                self.individuals.clear()
                self._save_individuals()

        self.tracker = None
        self.last_tracked_idx = -1
        self.track_cache.clear()
        self.traj_snapshots.clear()
        self.cmc_cache.clear()
        self.selected_idx = None
        self._pending_id_changes.clear()
        self._edited_tids.clear()
        self._invalidate_track_views()
        self._update_step_btn()
        self._update_export_btns()
        self._redraw()
        self._refresh_individuals(force=True)
        if not quiet:
            self._status("Tracker reset.")

    def _set_ui_locked(self, locked: bool):
        self._tracking_running = locked
        enabled = not locked
        for w in (
            self.open_video_btn, self.open_images_btn,
            self.prev_btn, self.next_btn,
            self.play_btn, self.pause_btn,
            self.frame_slider, self.reset_tracker_btn,
            self.edit_btn, self.delete_btn,
            self.conf_spin, self.frame_skip_spin,
        ):
            w.setEnabled(enabled)
        self.id_spin.setEnabled(enabled and self.selected_idx is not None)
        self.individuals_panel.setEnabled(enabled)
        ok = enabled and bool(self.track_cache)
        self.export_btn.setEnabled(ok)
        self.export_data_btn.setEnabled(ok)

    def _step_tracker(self):
        if not self.source or self.current_idx <= self.last_tracked_idx:
            return
        if self._tracking_running:
            return
        if self.tracker is None:
            self.tracker = build_tracker(self._cfg(), self._task())
            self.last_tracked_idx = -1
            self._status("Tracker initialised (BoTSORT AABB).")

        self._sync_edits_to_tracker()

        start = self.last_tracked_idx + 1
        end = self.current_idx
        cfg = self._cfg()

        self._set_ui_locked(True)
        self.step_btn.setEnabled(False)
        self.step_btn.setText("Tracking...")
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("0/?")

        self._wt = QtCore.QThread(self)
        self._wk = TrackingStepWorker(
            source=self.source,
            start_idx=start, end_idx=end,
            tracker=self.tracker,
            model_path=cfg.get("model_path", ""),
            conf=float(self.conf_spin.value()),
            imgsz=int(cfg.get("imgsz", 1024)),
            frame_skip=self.frame_skip_spin.value(),
            task=self._task(),
        )
        self._wk.moveToThread(self._wt)
        self._wt.started.connect(self._wk.run)
        self._wk.frame_tracked.connect(self._on_frame_tracked)
        self._wk.traj_snapshot.connect(self._on_traj_snapshot)
        self._wk.cmc_snapshot.connect(self._on_cmc_snapshot)
        self._wk.progress.connect(self._on_progress)
        self._wk.finished.connect(self._on_done)
        self._wk.error.connect(self._on_error)
        self._wk.finished.connect(self._wt.quit)
        self._wk.error.connect(self._wt.quit)
        self._wt.finished.connect(self._wk.deleteLater)
        self._wt.finished.connect(self._wt.deleteLater)
        self._wt.start()

    def _on_frame_tracked(self, idx: int, obbs):
        self.track_cache[idx] = list(obbs)
        self.last_tracked_idx = max(self.last_tracked_idx, idx)
        self._invalidate_track_views()
        if idx == self.current_idx:
            self._redraw()

    def _on_traj_snapshot(self, idx: int, snap):
        self.traj_snapshots[idx] = dict(snap)
        if idx == self.current_idx:
            self._redraw()

    def _on_cmc_snapshot(self, idx: int, warp):
        if warp is not None:
            self.cmc_cache[idx] = np.asarray(warp, dtype=np.float64)
        else:
            self.cmc_cache[idx] = None

    def _on_progress(self, done: int, total: int):
        pct = int(done / max(total, 1) * 100)
        self.progress_bar.setValue(pct)
        self.progress_bar.setFormat(f"{done}/{total}")

    def _on_done(self):
        self._set_ui_locked(False)
        self.step_btn.setText("Step Tracker ▶▶")
        self._update_step_btn()
        self._update_export_btns()
        self.progress_bar.setFormat("Done")
        self._redraw()
        self._refresh_individuals(force=True)
        last_snap = self.traj_snapshots.get(self.last_tracked_idx, {})
        self._status(
            f"Tracked to frame {self.last_tracked_idx}. "
            f"{len(last_snap)} unique IDs."
        )

    def _on_error(self, msg: str):
        self._set_ui_locked(False)
        self.step_btn.setText("Step Tracker ▶▶")
        self._update_step_btn()
        self._update_export_btns()
        self.progress_bar.setFormat("Error")
        self._status(f"Tracking error: {msg}")
        QtWidgets.QMessageBox.critical(self, "Tracking Error", msg)

    # ==================== Video export ====================

    def _export_video(self):
        if not self.source or not self.track_cache:
            return
        if self._tracking_running or self._exporting:
            return

        default_name = "tracked_output.mp4"
        if hasattr(self.source, "path"):
            src_path = Path(self.source.path)
            default_name = str(src_path.with_name(src_path.stem + "_tracked.mp4"))
        out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export tracked video", default_name,
            "Video (*.mp4);;All (*)",
        )
        if not out_path:
            return

        fps = self.source.fps() or 25
        self._exporting = True
        self._set_ui_locked(True)
        self.export_btn.setEnabled(False)
        self.export_btn.setText("Exporting...")
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Export 0/?")

        self._exp_thread = QtCore.QThread(self)
        self._exp_worker = VideoExportWorker(
            source=self.source,
            track_cache=dict(self.track_cache),
            traj_snapshots=dict(self.traj_snapshots),
            output_path=out_path,
            fps=fps,
            trail_length=self.trail_len_spin.value(),
            show_trails=self.show_trails_chk.isChecked(),
        )
        self._exp_worker.moveToThread(self._exp_thread)
        self._exp_thread.started.connect(self._exp_worker.run)
        self._exp_worker.progress.connect(self._on_export_progress)
        self._exp_worker.finished.connect(self._on_export_done)
        self._exp_worker.error.connect(self._on_export_error)
        self._exp_worker.finished.connect(self._exp_thread.quit)
        self._exp_worker.error.connect(self._exp_thread.quit)
        self._exp_thread.finished.connect(self._exp_worker.deleteLater)
        self._exp_thread.finished.connect(self._exp_thread.deleteLater)
        self._exp_thread.start()

    def _on_export_progress(self, done: int, total: int):
        pct = int(done / max(total, 1) * 100)
        self.progress_bar.setValue(pct)
        self.progress_bar.setFormat(f"Export {done}/{total}")

    def _on_export_done(self, path: str):
        self._exporting = False
        self._set_ui_locked(False)
        self.export_btn.setText("Export Video 🎬")
        self._update_export_btns()
        self.progress_bar.setFormat("Export done")
        self._status(f"Video exported → {path}")
        QtWidgets.QMessageBox.information(
            self, "Export complete", f"Video saved to:\n{path}"
        )

    def _on_export_error(self, msg: str):
        self._exporting = False
        self._set_ui_locked(False)
        self.export_btn.setText("Export Video 🎬")
        self._update_export_btns()
        self.progress_bar.setFormat("Export error")
        self._status(f"Export error: {msg}")
        QtWidgets.QMessageBox.critical(self, "Export Error", msg)

    # ==================== Data export ====================

    def _export_data(self):
        if not self.track_cache:
            return
        if self._tracking_running or self._exporting:
            return

        # Pre-fill with the clip's own export folder so the default answer is
        # the one the mapping and the post-processing script both expect.
        if self._export_dir:
            Path(self._export_dir).mkdir(parents=True, exist_ok=True)
        out_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select output folder for tracking data",
            self._export_dir or "",
        )
        if not out_dir:
            return

        pf_dir = Path(out_dir) / "per_frame"
        pt_dir = Path(out_dir) / "per_track"
        pf_dir.mkdir(parents=True, exist_ok=True)
        pt_dir.mkdir(parents=True, exist_ok=True)

        # ── CMC affine matrices (one entry per tracked frame) ──
        cmc_out: Dict[str, Any] = {}
        for fidx in sorted(self.cmc_cache.keys()):
            mat = self.cmc_cache[fidx]
            cmc_out[str(fidx)] = mat.tolist() if mat is not None else None
        if cmc_out:
            cmc_path = Path(out_dir) / "cmc_transforms.json"
            cmc_path.write_text(
                json.dumps(cmc_out, indent=2, ensure_ascii=False)
            )

        per_track: Dict[int, list] = defaultdict(list)
        n_frames = 0
        n_detections = 0

        header = (
            "# track_id centroid_x centroid_y "
            "bbox_x1 bbox_y1 bbox_x2 bbox_y2 "
            "obb_x1 obb_y1 obb_x2 obb_y2 obb_x3 obb_y3 obb_x4 obb_y4 "
            "confidence class_id\n"
        )

        only_assigned = self.only_assigned_chk.isChecked()
        n_skipped_tracks = 0
        if only_assigned:
            n_skipped_tracks = len(self.individuals.unassigned(
                self._all_cached_track_ids()
            ))

        for frame_idx in sorted(self.track_cache.keys()):
            annots = self.track_cache[frame_idx]
            active = [b for b in annots if not b.deleted and b.track_id >= 0]
            if only_assigned:
                active = [
                    b for b in active
                    if self.individuals.individual_of(b.track_id) is not None
                ]
            if not active:
                continue

            lines = [header]
            for b in active:
                pts = b.poly.reshape(4, 2)
                cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
                bx1, by1 = float(pts[:, 0].min()), float(pts[:, 1].min())
                bx2, by2 = float(pts[:, 0].max()), float(pts[:, 1].max())
                obb_flat = " ".join(
                    f"{pts[i, j]:.2f}"
                    for i in range(4) for j in range(2)
                )
                lines.append(
                    f"{b.track_id} "
                    f"{cx:.2f} {cy:.2f} "
                    f"{bx1:.2f} {by1:.2f} {bx2:.2f} {by2:.2f} "
                    f"{obb_flat} "
                    f"{b.conf:.4f} {b.cls_id}\n"
                )

                obb_xywhr = obb_to_xywhr(b.poly)
                frame_cmc = self.cmc_cache.get(frame_idx)
                cmc_entry = (
                    frame_cmc.tolist() if frame_cmc is not None else None
                )

                per_track[b.track_id].append({
                    "frame": frame_idx,
                    "centroid": [round(cx, 2), round(cy, 2)],
                    "bbox": [round(bx1, 2), round(by1, 2),
                             round(bx2, 2), round(by2, 2)],
                    "obb": [
                        [round(float(pts[i, 0]), 2),
                         round(float(pts[i, 1]), 2)]
                        for i in range(4)
                    ],
                    "obb_xywhr": {
                        "cx":         obb_xywhr[0],
                        "cy":         obb_xywhr[1],
                        "width":      obb_xywhr[2],
                        "height":     obb_xywhr[3],
                        "angle_deg":  obb_xywhr[4],
                    },
                    "cmc_affine": cmc_entry,
                    "confidence": round(b.conf, 4),
                    "class_id":   b.cls_id,
                })

            txt_path = pf_dir / f"frame_{frame_idx:06d}.txt"
            txt_path.write_text("".join(lines))
            n_frames += 1
            n_detections += len(active)

        for tid, detections in sorted(per_track.items()):
            record = {
                "track_id":       tid,
                "num_detections": len(detections),
                "first_frame":    detections[0]["frame"],
                "last_frame":     detections[-1]["frame"],
                "detections":     detections,
            }
            json_path = pt_dir / f"track_{tid:04d}.json"
            json_path.write_text(
                json.dumps(record, indent=2, ensure_ascii=False)
            )

        # ── Individuals: one merged file per animal ──
        # Same collision rule as track_postprocess.merge_tracks (highest
        # confidence wins). No interpolation or smoothing here: those stay in
        # the script, to be run over these files as an opt-in pass.
        written: List[dict] = []
        n_dropped = 0
        if len(self.individuals):
            written = self.individuals.export_individuals(
                individuals_dir(out_dir), per_track
            )
            n_dropped = sum(w["dropped"] for w in written)
            # Keep the identity next to the data it describes, and leave a
            # pasteable groups list for the script's hardcoded `groups`.
            self.individuals.save(individuals_path(out_dir))
            # ...and refresh the canonical per-clip copy, which is a different
            # file whenever the user exported somewhere else.
            self._save_individuals()
            self.individuals.export_groups_json(
                str(Path(out_dir) / "groups.json")
            )

        n_tracks = len(per_track)
        n_cmc = sum(1 for v in self.cmc_cache.values() if v is not None)
        self._status(
            f"Exported {n_detections} detections across {n_frames} frames, "
            f"{n_tracks} tracks, {len(written)} individuals, "
            f"{n_cmc} CMC matrices → {out_dir}"
        )

        lines = [
            f"Exported to: {out_dir}",
            "",
            f"per_frame/  → {n_frames} files",
            f"per_track/  → {n_tracks} JSON files",
            f"cmc_transforms.json → {n_cmc} matrices",
            f"Total detections: {n_detections}",
        ]
        if written:
            lines += [
                "",
                f"individuals/ → {len(written)} merged files",
                f"  {n_dropped} detection(s) dropped on frame collisions "
                "(highest confidence kept)",
                "groups.json  → pasteable into track_postprocess.py",
            ]
        elif len(self.individuals):
            lines += ["", "individuals/ → nothing written: no track grouped."]
        if n_skipped_tracks:
            lines += ["", f"{n_skipped_tracks} unassigned track(s) skipped."]

        QtWidgets.QMessageBox.information(
            self, "Data export complete", "\n".join(lines),
        )

    # ==================== Selection / editing ====================

    def _toggle_edit(self):
        self.mode = "edit" if self.mode != "edit" else "select"
        self._status(f"Mode: {self.mode}")

    def _delete_selected(self):
        annots = self.track_cache.get(self.current_idx, [])
        if self.selected_idx is None or self.selected_idx >= len(annots):
            return
        annots[self.selected_idx].deleted = True
        self.selected_idx = None
        self._invalidate_track_views()
        self._update_id_spin()
        self._redraw()
        self._refresh_individuals()

    def _update_id_spin(self):
        annots = self.track_cache.get(self.current_idx, [])
        valid = (
            self.selected_idx is not None
            and 0 <= self.selected_idx < len(annots)
            and not annots[self.selected_idx].deleted
        )
        self.id_spin.blockSignals(True)
        if valid:
            self.id_spin.setEnabled(True)
            self.id_spin.setValue(annots[self.selected_idx].track_id)
        else:
            self.id_spin.setEnabled(False)
            self.id_spin.setValue(-1)
        self.id_spin.blockSignals(False)

    def _on_id_spin_changed(self, val: int):
        annots = self.track_cache.get(self.current_idx, [])
        if self.selected_idx is None or self.selected_idx >= len(annots):
            return
        old_id = annots[self.selected_idx].track_id
        if old_id != val:
            annots[self.selected_idx].track_id = val
            if old_id >= 0:
                self._pending_id_changes[old_id] = val
            self._invalidate_track_views()
        self._redraw()
        self._refresh_individuals()

    def _pick_annot(self, x: float, y: float) -> Optional[int]:
        annots = self.track_cache.get(self.current_idx, [])
        best, ba = None, None
        for i, b in enumerate(annots):
            if b.deleted:
                continue
            pts = b.poly.reshape(-1, 2).astype(np.float32)
            if cv2.pointPolygonTest(pts, (x, y), False) >= 0:
                a = cv2.contourArea(pts.astype(np.int32))
                if best is None or a < ba:
                    best, ba = i, a
        return best

    def _pick_vertex(self, x: float, y: float, tol: float = 10.0) -> Optional[int]:
        annots = self.track_cache.get(self.current_idx, [])
        if self.selected_idx is None or self.selected_idx >= len(annots):
            return None
        pts = annots[self.selected_idx].poly.reshape(-1, 2)
        for i in range(pts.shape[0]):
            if np.hypot(pts[i, 0] - x, pts[i, 1] - y) <= tol:
                return i
        return None

    def _translate_selected(self, dx: float, dy: float):
        annots = self.track_cache.get(self.current_idx, [])
        if self.selected_idx is None or self.selected_idx >= len(annots):
            return
        if self.orig_poly is None:
            return
        annots[self.selected_idx].poly = (
            self.orig_poly + np.array([dx, dy], dtype=np.float32)
        ).astype(np.float32)
        self._mark_edited(annots[self.selected_idx])

    def _mark_edited(self, box):
        """Remember that this track's geometry was corrected by hand.

        Only marked tracks get their Kalman state rewritten before the next
        step, so untouched tracks keep predicting normally.
        """
        tid = int(getattr(box, "track_id", -1))
        if tid >= 0:
            self._edited_tids.add(tid)

    def _set_vertex(self, vi: int, x: float, y: float):
        annots = self.track_cache.get(self.current_idx, [])
        if self.selected_idx is None or self.selected_idx >= len(annots):
            return
        self._mark_edited(annots[self.selected_idx])
        p = annots[self.selected_idx].poly.copy()
        p[vi] = [x, y]
        annots[self.selected_idx].poly = p.astype(np.float32)

    # ==================== Canvas mouse handlers ====================

    def _busy(self) -> bool:
        return self._tracking_running or self._exporting

    def _on_canvas_mouse_press(self, event, xi: float, yi: float):
        if self._busy() or self.current_frame_bgr is None:
            return
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return

        # Dragging redraws on every move; keep resampling cheap until release.
        self.canvas.set_fast_scaling(True)

        hit = self._pick_annot(xi, yi)
        if hit is None:
            self.selected_idx = None
            self._update_id_spin()
            self._redraw()
            self._refresh_individuals()
            return

        self.selected_idx = hit
        self._update_id_spin()
        self._redraw()
        self._refresh_individuals()

        # Edit-mode: try to pick a vertex first
        if self.mode == "edit":
            v = self._pick_vertex(xi, yi)
            if v is not None:
                self.vertex_drag_idx = v
                self.dragging = True
                return

        # Otherwise start translating the whole polygon
        annots = self.track_cache.get(self.current_idx, [])
        if self.selected_idx < len(annots):
            self.dragging = True
            self.drag_start_img = (xi, yi)
            self.orig_poly = annots[self.selected_idx].poly.copy()

    def _on_canvas_mouse_move(self, event, xi: float, yi: float):
        if self._busy() or not self.dragging:
            return
        if self.vertex_drag_idx is not None:
            self._set_vertex(self.vertex_drag_idx, xi, yi)
        elif self.drag_start_img is not None:
            self._translate_selected(
                xi - self.drag_start_img[0],
                yi - self.drag_start_img[1],
            )
        self._redraw()

    def _on_canvas_mouse_release(self, event, xi: float, yi: float):
        if not self.playing:
            self.canvas.set_fast_scaling(False)
        if not self.dragging:
            self._redraw()
            return
        self.dragging = False
        self.vertex_drag_idx = None
        self.drag_start_img = None
        self.orig_poly = None
        self._redraw()

    # ==================== Keyboard (forwarded from launcher) ====================

    def set_space_held(self, held: bool):
        """Forward space-key state from the launcher to the canvas."""
        self.canvas.set_space_held(held)