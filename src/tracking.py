"""
Tracking page — BoxMOT BoTSORT with AABB detections.
OBB detections are converted to axis-aligned bounding boxes for the tracker,
then track IDs are mapped back onto the original OBBs.
Trajectories are read from the tracker's internal STrack history_observations.

Exports:
  per_frame/frame_XXXXXX.txt  — one line per detection
  per_track/track_XXXX.json   — full history per track ID
"""

import os
import json
import math
from typing import Optional, List, Dict, Tuple, Any
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .utils import (
    OBBOX, FrameSource, VideoSource, ImageFolderSource,
    cvimg_to_qimage, ensure_bgr_u8,
)
from .workers import YOLO_MODEL_PATH

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    from boxmot.trackers import BotSort
    BOXMOT_AVAILABLE = True
except ImportError:
    BOXMOT_AVAILABLE = False

import torch


# ---------------------------------------------------------------------------
# Tracker factory
# ---------------------------------------------------------------------------

def build_tracker(cfg: dict):
    if not BOXMOT_AVAILABLE:
        raise RuntimeError("boxmot is not installed. pip install boxmot")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    reid_weights = Path(cfg.get("reid_weights", "osnet_x0_25_msmt17.pt"))
    return BotSort(
        reid_weights=reid_weights,
        device=device,
        half=False,
        with_reid=cfg.get("with_reid", True),
        track_high_thresh=float(cfg.get("track_high_thresh", 0.6)),
        track_low_thresh=float(cfg.get("track_low_thresh", 0.1)),
        new_track_thresh=float(cfg.get("new_track_thresh", 0.7)),
        track_buffer=int(cfg.get("track_buffer", 30)),
        match_thresh=float(cfg.get("match_thresh", 0.8)),
        proximity_thresh=float(cfg.get("proximity_thresh", 0.5)),
        appearance_thresh=float(cfg.get("appearance_thresh", 0.25)),
    )


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

def _make_palette(n: int = 64) -> List[Tuple[int, int, int]]:
    pal = []
    for i in range(n):
        h = int(180 * i / n)
        s = 200 + (i % 3) * 25
        v = 220 + (i % 2) * 35
        bgr = cv2.cvtColor(
            np.array([[[h, min(s, 255), min(v, 255)]]], dtype=np.uint8),
            cv2.COLOR_HSV2BGR,
        )[0, 0]
        pal.append(tuple(int(c) for c in bgr))
    return pal

_PALETTE = _make_palette(64)

def track_color(tid: int) -> Tuple[int, int, int]:
    return (0, 200, 255) if tid < 0 else _PALETTE[tid % len(_PALETTE)]


# ---------------------------------------------------------------------------
# OBB → AABB conversion
# ---------------------------------------------------------------------------

def obb_to_aabb_row(box: OBBOX) -> np.ndarray:
    pts = box.poly.reshape(-1, 2)
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    return np.array([x1, y1, x2, y2, box.conf, box.cls_id], dtype=np.float32)


def obb_centroid(box: OBBOX) -> Tuple[float, float]:
    pts = box.poly.reshape(-1, 2)
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())


# ---------------------------------------------------------------------------
# OBB → xywhr (centre, width, height, angle in degrees)
# ---------------------------------------------------------------------------

def obb_to_xywhr(poly: np.ndarray) -> Tuple[float, float, float, float, float]:
    """Return (cx, cy, w, h, angle_deg) from a 4-point OBB polygon."""
    pts = poly.reshape(4, 2).astype(np.float32)
    (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
    return (round(cx, 2), round(cy, 2),
            round(w, 2), round(h, 2), round(angle, 2))


# ---------------------------------------------------------------------------
# Update a BotSort STrack's Kalman state from an AABB
# ---------------------------------------------------------------------------

def _update_strack_bbox(strack, x1, y1, x2, y2):
    """Best-effort update of a STrack's internal Kalman mean to match a new AABB.
    BotSort state format is typically [cx, cy, aspect_ratio, h, vx, vy, va, vh].
    """
    mean = getattr(strack, "mean", None)
    if mean is None or len(mean) < 4:
        return
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = max(x2 - x1, 1e-6)
    h = max(y2 - y1, 1e-6)
    mean[0] = cx
    mean[1] = cy
    mean[2] = w / h          # aspect ratio
    mean[3] = h
    # Zero velocities so the Kalman prediction doesn't drift from old momentum
    if len(mean) >= 8:
        mean[4] = 0.0
        mean[5] = 0.0
        mean[6] = 0.0
        mean[7] = 0.0


# ---------------------------------------------------------------------------
# Extract trajectories from tracker internal STracks
# ---------------------------------------------------------------------------

def extract_trajectories_from_tracker(tracker) -> Dict[int, List[Tuple[float, float]]]:
    trajectories: Dict[int, List[Tuple[float, float]]] = {}
    seen = set()
    all_stracks = []
    for attr in ("active_tracks", "lost_stracks"):
        pool = getattr(tracker, attr, None)
        if not pool:
            continue
        for st in pool:
            obj_id = id(st)
            if obj_id not in seen:
                seen.add(obj_id)
                all_stracks.append(st)

    for strack in all_stracks:
        tid = int(getattr(strack, "id", -1))
        if tid < 0:
            continue
        centers = []
        obs = getattr(strack, "history_observations", None)
        if obs:
            for box in obs:
                box_arr = np.asarray(box, dtype=np.float32).ravel()
                if len(box_arr) >= 4:
                    cx = float((box_arr[0] + box_arr[2]) / 2)
                    cy = float((box_arr[1] + box_arr[3]) / 2)
                    centers.append((cx, cy))
        mean = getattr(strack, "mean", None)
        if mean is not None:
            try:
                xyxy = strack.xyxy
                box_arr = np.asarray(xyxy, dtype=np.float32).ravel()
                if len(box_arr) >= 4:
                    cx = float((box_arr[0] + box_arr[2]) / 2)
                    cy = float((box_arr[1] + box_arr[3]) / 2)
                    centers.append((cx, cy))
            except Exception:
                pass
        if centers:
            trajectories[tid] = centers
    return trajectories


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_tracked_annotations(
    img_bgr: np.ndarray,
    annots: List[OBBOX],
    selected_idx: Optional[int],
    trajectories: Dict[int, List[Tuple[float, float]]],
    trail_length: int = 60,
    show_trails: bool = True,
) -> np.ndarray:
    out = img_bgr.copy()

    if show_trails:
        for tid, centers in trajectories.items():
            if tid < 0:
                continue
            color = track_color(tid)
            recent = centers[-trail_length:] if len(centers) > trail_length else centers
            if len(recent) < 2:
                continue
            coords = np.array(recent, dtype=np.int32)
            n = len(coords)
            for j in range(1, n):
                alpha = j / n
                thick = max(1, int(1 + 2 * alpha))
                c = tuple(int(v * (0.3 + 0.7 * alpha)) for v in color)
                cv2.line(out, tuple(coords[j - 1]), tuple(coords[j]),
                         c, thick, cv2.LINE_AA)
            cv2.circle(out, tuple(coords[-1]), 4, color, -1, cv2.LINE_AA)

    for i, b in enumerate(annots):
        if b.deleted:
            continue
        pts = b.poly.reshape(-1, 2).astype(int)
        tid = b.track_id

        if selected_idx is not None and i == selected_idx:
            color, thick = (255, 0, 255), 5
        else:
            color, thick = track_color(tid), 4

        cv2.polylines(out, [pts], True, color, thick, cv2.LINE_AA)

        label = f"ID:{tid}" if tid >= 0 else "?"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        x0, y0 = int(pts[0, 0]), int(pts[0, 1]) - 6
        cv2.rectangle(out, (x0, y0 - th - 4), (x0 + tw + 6, y0 + 2), color, -1)
        cv2.putText(out, label, (x0 + 3, y0 - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    return out


# ---------------------------------------------------------------------------
# IoU helper
# ---------------------------------------------------------------------------

def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, 0:1], b[:, 0:1].T)
    y1 = np.maximum(a[:, 1:2], b[:, 1:2].T)
    x2 = np.minimum(a[:, 2:3], b[:, 2:3].T)
    y2 = np.minimum(a[:, 3:4], b[:, 3:4].T)
    inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-6)


# ---------------------------------------------------------------------------
# Tracking step worker
# ---------------------------------------------------------------------------

class TrackingStepWorker(QtCore.QObject):
    frame_tracked = QtCore.Signal(int, object)
    traj_snapshot = QtCore.Signal(int, object)
    cmc_snapshot  = QtCore.Signal(int, object)   # frame_idx, 2×3 or 3×3 ndarray
    progress      = QtCore.Signal(int, int)
    finished      = QtCore.Signal()
    error         = QtCore.Signal(str)

    def __init__(self, source, start_idx, end_idx, tracker,
                 model_path, conf, imgsz, frame_skip=1):
        super().__init__()
        self.source = source
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.tracker = tracker
        self.model_path = model_path
        self.conf = conf
        self.imgsz = imgsz
        self.frame_skip = max(1, frame_skip)

    @staticmethod
    def _extract_obbs(res) -> List[OBBOX]:
        boxes: List[OBBOX] = []
        has_obb = hasattr(res, "obb") and res.obb is not None and len(res.obb) > 0
        if has_obb:
            obb = res.obb
            polys = getattr(obb, "xyxyxyxy", None)
            cls = getattr(obb, "cls", None)
            conf_vals = getattr(obb, "conf", None)
            if polys is not None and len(polys) > 0:
                P = polys.cpu().numpy() if hasattr(polys, "cpu") else np.asarray(polys)
                C = cls.cpu().numpy() if hasattr(cls, "cpu") else np.zeros(len(P))
                S = conf_vals.cpu().numpy() if hasattr(conf_vals, "cpu") else np.ones(len(P))
                for p, c, s in zip(P, C, S):
                    boxes.append(OBBOX(poly=p.reshape(4, 2).astype(np.float32),
                                       cls_id=int(c), conf=float(s)))
            else:
                xywhr = getattr(obb, "xywhr", None)
                if xywhr is not None and len(xywhr) > 0:
                    X = xywhr.cpu().numpy() if hasattr(xywhr, "cpu") else np.asarray(xywhr)
                    C = cls.cpu().numpy() if hasattr(cls, "cpu") else np.zeros(len(X))
                    S = conf_vals.cpu().numpy() if hasattr(conf_vals, "cpu") else np.ones(len(X))
                    for (cx, cy, w, h, rad), c, s in zip(X, C, S):
                        rect = np.array([[-w/2, -h/2], [w/2, -h/2],
                                         [w/2, h/2], [-w/2, h/2]], dtype=np.float32)
                        cos_r, sin_r = np.cos(rad), np.sin(rad)
                        R = np.array([[cos_r, -sin_r], [sin_r, cos_r]], dtype=np.float32)
                        pts = rect @ R.T + np.array([cx, cy], dtype=np.float32)
                        boxes.append(OBBOX(poly=pts, cls_id=int(c), conf=float(s)))
        elif res.boxes is not None and len(res.boxes) > 0:
            from .utils import rect_to_poly_xyxy
            xyxy = res.boxes.xyxy.cpu().numpy()
            C = res.boxes.cls.cpu().numpy()
            S = res.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), c, s in zip(xyxy, C, S):
                boxes.append(OBBOX(poly=rect_to_poly_xyxy(x1, y1, x2, y2),
                                   cls_id=int(c), conf=float(s)))
        return boxes

    @staticmethod
    def _assign_ids(obbs, det_aabbs, tracks):
        if tracks is None or len(tracks) == 0:
            return obbs
        trk_ids = tracks[:, 4].astype(int)
        if tracks.shape[1] >= 8:
            det_indices = tracks[:, 7].astype(int)
            for row, di in enumerate(det_indices):
                if 0 <= di < len(obbs):
                    obbs[di].track_id = int(trk_ids[row])
            return obbs
        if not obbs or len(det_aabbs) == 0:
            return obbs
        trk_boxes = tracks[:, :4]
        ious = _iou_matrix(det_aabbs[:, :4], trk_boxes)
        used = set()
        for ti in range(len(trk_boxes)):
            best_det = int(ious[:, ti].argmax())
            if best_det not in used and ious[best_det, ti] > 0.3:
                obbs[best_det].track_id = int(trk_ids[ti])
                used.add(best_det)
        return obbs

    @staticmethod
    def _extract_cmc_matrix(tracker) -> Optional[np.ndarray]:
        """Best-effort extraction of the Camera Motion Compensation warp matrix
        from a BoxMOT tracker.  Returns a 2×3 (affine) or 3×3 (homography)
        numpy array, or None if unavailable."""
        
       
        mat = getattr(tracker, "warp", None)
        if mat is not None and isinstance(mat, np.ndarray):
            return mat.copy()
        return None

    @QtCore.Slot()
    def run(self):
        try:
            if YOLO is None:
                raise RuntimeError("ultralytics not installed")
            model = YOLO(self.model_path)
            frames = list(range(self.start_idx, self.end_idx + 1, self.frame_skip))
            if frames[-1] != self.end_idx:
                frames.append(self.end_idx)
            total = len(frames)
            for i, idx in enumerate(frames):
                frame = self.source.read(idx)
                if frame is None:
                    self.frame_tracked.emit(idx, [])
                    self.traj_snapshot.emit(idx, {})
                    self.progress.emit(i + 1, total)
                    continue
                frame = ensure_bgr_u8(frame)
                results = model.predict(source=frame, imgsz=self.imgsz,
                                        conf=self.conf, verbose=False)
                obbs = self._extract_obbs(results[0])
                if obbs:
                    det_aabbs = np.stack([obb_to_aabb_row(b) for b in obbs])
                else:
                    det_aabbs = np.empty((0, 6), dtype=np.float32)
                tracks = self.tracker.update(det_aabbs, frame)
                obbs = self._assign_ids(obbs, det_aabbs, tracks)
                # ── Trajectories from tracker internal STracks ──
                snap = extract_trajectories_from_tracker(self.tracker)
                # ── Extract CMC warp matrix (affine 2×3 or homography 3×3) ──
                warp = self._extract_cmc_matrix(self.tracker)
                self.frame_tracked.emit(idx, obbs)
                self.traj_snapshot.emit(idx, snap)
                self.cmc_snapshot.emit(idx, warp)
                self.progress.emit(i + 1, total)
            self.finished.emit()
        except Exception as e:
            import traceback; traceback.print_exc()
            self.error.emit(str(e))


# ---------------------------------------------------------------------------
# Video export worker
# ---------------------------------------------------------------------------

class VideoExportWorker(QtCore.QObject):
    """Renders every tracked frame with annotations + trajectories to a .mp4."""
    progress = QtCore.Signal(int, int)
    finished = QtCore.Signal(str)
    error    = QtCore.Signal(str)

    def __init__(self, source: FrameSource, track_cache: Dict[int, List[OBBOX]],
                 traj_snapshots: Dict[int, Dict[int, List[Tuple[float, float]]]],
                 output_path: str, fps: float,
                 trail_length: int = 60, show_trails: bool = True):
        super().__init__()
        self.source = source
        self.track_cache = track_cache
        self.traj_snapshots = traj_snapshots
        self.output_path = output_path
        self.fps = fps
        self.trail_length = trail_length
        self.show_trails = show_trails

    @QtCore.Slot()
    def run(self):
        try:
            if not self.track_cache:
                self.error.emit("Nothing to export — run the tracker first."); return
            last_idx = max(self.track_cache.keys())
            total = last_idx + 1
            sample = self.source.read(0)
            if sample is None:
                self.error.emit("Cannot read first frame."); return
            sample = ensure_bgr_u8(sample)
            h, w = sample.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (w, h))
            if not writer.isOpened():
                self.error.emit(f"Cannot open VideoWriter for {self.output_path}"); return
            sorted_keys = sorted(self.track_cache.keys())

            def _closest_earlier(idx, keys):
                lo, hi, best = 0, len(keys) - 1, None
                while lo <= hi:
                    mid = (lo + hi) // 2
                    if keys[mid] <= idx:
                        best = keys[mid]; lo = mid + 1
                    else:
                        hi = mid - 1
                return best

            for idx in range(total):
                frame = self.source.read(idx)
                if frame is None:
                    writer.write(np.zeros((h, w, 3), dtype=np.uint8))
                    self.progress.emit(idx + 1, total); continue
                frame = ensure_bgr_u8(frame)
                annots = self.track_cache.get(idx, [])
                snap_key = _closest_earlier(idx, sorted_keys)
                trajectories = self.traj_snapshots.get(snap_key, {}) if snap_key is not None else {}
                rendered = draw_tracked_annotations(
                    frame, annots, selected_idx=None,
                    trajectories=trajectories,
                    trail_length=self.trail_length,
                    show_trails=self.show_trails,
                )
                writer.write(rendered)
                self.progress.emit(idx + 1, total)
            writer.release()
            self.finished.emit(self.output_path)
        except Exception as e:
            import traceback; traceback.print_exc()
            self.error.emit(str(e))


# ---------------------------------------------------------------------------
# Tracking page widget
# ---------------------------------------------------------------------------

class TrackingPage(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._launcher = None

        self.source: Optional[FrameSource] = None
        self.total_frames: int = 0
        self.current_idx: int = 0
        self.current_frame_bgr: Optional[np.ndarray] = None
        self.play_timer = QtCore.QTimer(self)
        self.play_timer.timeout.connect(self._on_play_tick)
        self.playing = False

        self.zoom = 1.0
        self.min_zoom, self.max_zoom = 0.25, 8.0
        self.pan_img = np.array([0.0, 0.0], dtype=np.float32)

        self.tracker = None
        self.last_tracked_idx: int = -1
        self.track_cache: Dict[int, List[OBBOX]] = {}
        self.traj_snapshots: Dict[int, Dict[int, List[Tuple[float, float]]]] = {}
        self.cmc_cache: Dict[int, Optional[np.ndarray]] = {}   # frame → affine 2×3 or 3×3

        self.selected_idx: Optional[int] = None
        self.draw_map: Dict[str, float] = {"scale": 1.0, "xoff": 0, "yoff": 0}
        self.space_held = False
        self.mode = "select"
        self.dragging = False
        self.drag_start_img = None
        self.orig_poly = None
        self.vertex_drag_idx = None

        # ── Pending edit bookkeeping for tracker sync ──
        self._pending_id_changes: Dict[int, int] = {}   # old_tid → new_tid

        # ================ UI ================
        self.canvas = QtWidgets.QLabel()
        self.canvas.setMouseTracking(True)
        self.canvas.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.canvas.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.canvas.setStyleSheet("background:#111; border:1px solid #333;")
        self.canvas.setMinimumSize(720, 405)
        self.canvas.installEventFilter(self)

        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.sliderReleased.connect(self._on_slider)

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

        # ── Export video button ──
        self.export_btn = QtWidgets.QPushButton("Export Video 🎬")
        self.export_btn.setFixedHeight(36)
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._export_video)

        # ── Export data button ──
        self.export_data_btn = QtWidgets.QPushButton("Export Data 📊")
        self.export_data_btn.setToolTip(
            "Export per_frame/*.txt + per_track/*.json (no images)")
        self.export_data_btn.setFixedHeight(36)
        self.export_data_btn.setEnabled(False)
        self.export_data_btn.clicked.connect(self._export_data)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100); self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Idle")
        self.progress_bar.setFixedHeight(18)

        self.edit_btn   = QtWidgets.QPushButton("Edit (E)")
        self.delete_btn = QtWidgets.QPushButton("Delete (Del)")

        self.id_spin = QtWidgets.QSpinBox()
        self.id_spin.setPrefix("Track ID: ")
        self.id_spin.setRange(-1, 99999); self.id_spin.setValue(-1)
        self.id_spin.setEnabled(False)
        self.id_spin.valueChanged.connect(self._on_id_spin_changed)

        self.conf_spin = QtWidgets.QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 0.99); self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.5); self.conf_spin.setPrefix("conf=")

        self.frame_skip_spin = QtWidgets.QSpinBox()
        self.frame_skip_spin.setPrefix("Track every ")
        self.frame_skip_spin.setSuffix(" frames")
        self.frame_skip_spin.setRange(1, 30)
        self.frame_skip_spin.setValue(1)

        self._tracking_running = False
        self._exporting = False

        self.show_trails_chk = QtWidgets.QCheckBox("Show trajectories")
        self.show_trails_chk.setChecked(True)
        self.show_trails_chk.toggled.connect(lambda _: self._redraw())

        self.trail_len_spin = QtWidgets.QSpinBox()
        self.trail_len_spin.setPrefix("Trail: "); self.trail_len_spin.setSuffix(" frames")
        self.trail_len_spin.setRange(5, 9999); self.trail_len_spin.setValue(60)
        self.trail_len_spin.valueChanged.connect(lambda _: self._redraw())

        self.open_video_btn.clicked.connect(self._open_video)
        self.open_images_btn.clicked.connect(self._open_folder)
        self.prev_btn.clicked.connect(self.prev_frame)
        self.next_btn.clicked.connect(self.next_frame)
        self.play_btn.clicked.connect(self.play)
        self.pause_btn.clicked.connect(self.pause)
        self.zoom_in_btn.clicked.connect(lambda: self.zoom_step(+1))
        self.zoom_out_btn.clicked.connect(lambda: self.zoom_step(-1))
        self.zoom_fit_btn.clicked.connect(self.zoom_fit)
        self.step_btn.clicked.connect(self._step_tracker)
        self.reset_tracker_btn.clicked.connect(self._reset_tracker)
        self.edit_btn.clicked.connect(self._toggle_edit)
        self.delete_btn.clicked.connect(self._delete_selected)

        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0); lv.setSpacing(6)
        lv.addWidget(self.canvas, stretch=1)
        lv.addWidget(self.frame_slider)

        content = QtWidgets.QHBoxLayout()
        content.setContentsMargins(0, 0, 0, 0); content.setSpacing(10)
        content.addWidget(left, stretch=1)
        content.addWidget(self._build_side(), stretch=0)

        page = QtWidgets.QVBoxLayout(self)
        page.setContentsMargins(8, 8, 8, 8); page.setSpacing(8)
        page.addLayout(content, stretch=1)
        page.addWidget(self._build_transport())

    def _build_side(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(panel)
        v.setContentsMargins(8, 8, 8, 8); v.setSpacing(8)
        v.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        src = QtWidgets.QGroupBox("Source")
        sl = QtWidgets.QVBoxLayout(src)
        sl.addWidget(self.open_video_btn); sl.addWidget(self.open_images_btn)

        trk = QtWidgets.QGroupBox("Tracking")
        tl = QtWidgets.QVBoxLayout(trk)
        tl.addWidget(self.step_btn); tl.addWidget(self.progress_bar)
        tl.addWidget(self.conf_spin); tl.addWidget(self.frame_skip_spin)
        tl.addWidget(self.reset_tracker_btn)

        vis = QtWidgets.QGroupBox("Visualisation")
        vl = QtWidgets.QVBoxLayout(vis)
        vl.addWidget(self.show_trails_chk); vl.addWidget(self.trail_len_spin)

        edit = QtWidgets.QGroupBox("Edit selected")
        el = QtWidgets.QVBoxLayout(edit)
        el.addWidget(self.edit_btn); el.addWidget(self.id_spin)
        el.addWidget(self.delete_btn)

        exp = QtWidgets.QGroupBox("Export")
        xl = QtWidgets.QVBoxLayout(exp)
        xl.addWidget(self.export_btn)
        xl.addWidget(self.export_data_btn)

        v.addWidget(src); v.addWidget(trk); v.addWidget(vis)
        v.addWidget(edit); v.addWidget(exp)
        v.addStretch(1)
        panel.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed,
                            QtWidgets.QSizePolicy.Policy.Expanding)
        return panel

    def _build_transport(self) -> QtWidgets.QWidget:
        bar = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(bar)
        h.setContentsMargins(0, 0, 0, 0); h.setSpacing(10)
        h.addStretch(1)
        for w in (self.prev_btn, self.play_btn, self.pause_btn, self.next_btn):
            h.addWidget(w)
        h.addSpacing(20)
        for w in (self.zoom_out_btn, self.zoom_in_btn, self.zoom_fit_btn):
            h.addWidget(w)
        h.addStretch(1)
        return bar

    def set_launcher(self, launcher):
        self._launcher = launcher

    def _status(self, msg):
        if self._launcher:
            self._launcher.statusBar().showMessage(msg, 5000)

    def _cfg(self) -> dict:
        return self._launcher.project_config() if self._launcher else {}

    # ==================================================================
    # Source
    # ==================================================================

    def _open_video(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Video", "",
            "Video (*.mp4 *.avi *.mov *.mkv);;All (*)")
        if p:
            self._load_source(VideoSource(p))

    def _open_folder(self):
        f = QtWidgets.QFileDialog.getExistingDirectory(self, "Open Image Folder")
        if f:
            self._load_source(ImageFolderSource(f))

    def _load_source(self, src):
        if self.source:
            try: self.source.close()
            except: pass
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
        self._reset_tracker()
        self._read_frame(0)
        self._status(f"Loaded: {src.name()} | {self.total_frames} frames")

    # ==================================================================
    # Frame read / display
    # ==================================================================

    def _read_frame(self, idx):
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
        if self._launcher:
            self._launcher.update_title()

    def _redraw(self):
        if self.current_frame_bgr is None:
            return
        annots = self.track_cache.get(self.current_idx, [])
        trajectories = self.traj_snapshots.get(self.current_idx, {})
        annotated = draw_tracked_annotations(
            self.current_frame_bgr.copy(), annots, self.selected_idx,
            trajectories,
            trail_length=self.trail_len_spin.value(),
            show_trails=self.show_trails_chk.isChecked(),
        )
        self._show_frame(annotated)

    def _show_frame(self, frame_bgr):
        qimg = cvimg_to_qimage(frame_bgr)
        iw, ih = qimg.width(), qimg.height()
        lw, lh = self.canvas.width(), self.canvas.height()
        base = min(lw / iw, lh / ih) if iw and ih else 1.0
        sc = base * self.zoom
        dw, dh = int(iw * sc), int(ih * sc)
        xoff = (lw - dw) // 2
        yoff = (lh - dh) // 2
        cvs = QtGui.QPixmap(lw, lh)
        cvs.fill(QtGui.QColor(17, 17, 17))
        p = QtGui.QPainter(cvs)
        scaled = QtGui.QPixmap.fromImage(qimg).scaled(
            dw, dh, QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation)
        p.drawPixmap(int(xoff - self.pan_img[0] * sc),
                     int(yoff - self.pan_img[1] * sc), scaled)
        p.end()
        self.draw_map = {
            "scale": sc, "xoff": xoff, "yoff": yoff,
            "img_w": iw, "img_h": ih,
            "panx": float(self.pan_img[0]), "pany": float(self.pan_img[1]),
            "base": base, "lbl_w": lw, "lbl_h": lh,
        }
        self.canvas.setPixmap(cvs)

    # ==================================================================
    # Navigation
    # ==================================================================

    def prev_frame(self):
        if self.source and self.current_idx > 0:
            self.pause(); self._read_frame(self.current_idx - 1)

    def next_frame(self):
        if self.source and self.current_idx < self.total_frames - 1:
            self.pause(); self._read_frame(self.current_idx + 1)

    def _on_slider(self):
        if self.source:
            self.pause(); self._read_frame(self.frame_slider.value())

    def play(self):
        if not self.source or self.playing: return
        fps = self.source.fps() or 25
        self.play_timer.start(max(15, int(1000 / fps)))
        self.playing = True

    def pause(self):
        if self.playing:
            self.play_timer.stop(); self.playing = False

    def _on_play_tick(self):
        if self.current_idx + 1 >= self.total_frames:
            self.pause(); return
        self._read_frame(self.current_idx + 1)

    # ==================================================================
    # Zoom / pan
    # ==================================================================

    def display_to_image(self, xd, yd):
        m = self.draw_map
        if not m: return None, None
        s = m["scale"]
        xi = (xd - (m["xoff"] - m["panx"] * s)) / s
        yi = (yd - (m["yoff"] - m["pany"] * s)) / s
        if xi < 0 or yi < 0 or xi >= m.get("img_w", 1) or yi >= m.get("img_h", 1):
            return None, None
        return float(xi), float(yi)

    def zoom_fit(self):
        self.zoom = 1.0; self.pan_img[:] = 0.0; self._redraw()

    def zoom_step(self, d, anchor=None):
        if self.current_frame_bgr is None: return
        m = self.draw_map
        if not m or "base" not in m: return
        if anchor is None:
            anchor = QtCore.QPointF(self.canvas.width() / 2, self.canvas.height() / 2)
        step = 1.25 if d > 0 else 0.8
        nz = float(np.clip(self.zoom * step, self.min_zoom, self.max_zoom))
        if abs(nz - self.zoom) < 1e-6: return
        xd, yd = float(anchor.x()), float(anchor.y())
        xi, yi = self.display_to_image(int(xd), int(yd))
        if xi is None:
            self.zoom = nz; self._redraw(); return
        self.zoom = nz
        ns = float(m["base"]) * self.zoom
        xoff = (m["lbl_w"] - m["img_w"] * ns) / 2.0
        yoff = (m["lbl_h"] - m["img_h"] * ns) / 2.0
        self.pan_img[0] = (xoff + xi * ns - xd) / ns
        self.pan_img[1] = (yoff + yi * ns - yd) / ns
        self._redraw()

    # ==================================================================
    # Tracker — sync edits into BoxMOT internal state
    # ==================================================================

    def _sync_edits_to_tracker(self):
        """Push user edits (delete / move / ID change) from track_cache back
        into the BoxMOT tracker's internal STracks so the next .update() call
        starts from the corrected state."""
        if self.tracker is None or self.last_tracked_idx < 0:
            return

        annots = self.track_cache.get(self.last_tracked_idx, [])

        # ── 1. Apply pending ID changes ──
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

        # ── 2. Remove deleted tracks ──
        deleted_tids = {b.track_id for b in annots
                        if b.deleted and b.track_id >= 0}
        if deleted_tids:
            for attr in ("active_tracks", "lost_stracks"):
                pool = getattr(self.tracker, attr, None)
                if pool is None:
                    continue
                to_rm = [st for st in pool
                         if int(getattr(st, "id", -1)) in deleted_tids]
                for st in to_rm:
                    pool.remove(st)

        # ── 3. Update positions for surviving tracks ──
        active_map: Dict[int, Tuple[float, float, float, float]] = {}
        for b in annots:
            if not b.deleted and b.track_id >= 0:
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
                            _update_strack_bbox(st, *active_map[tid])
                        except Exception:
                            pass

    # ==================================================================
    # Tracker — step
    # ==================================================================

    def _update_step_btn(self):
        self.step_btn.setEnabled(
            self.source is not None and self.current_idx > self.last_tracked_idx)

    def _update_export_btns(self):
        ok = (bool(self.track_cache)
              and not self._tracking_running
              and not self._exporting)
        self.export_btn.setEnabled(ok)
        self.export_data_btn.setEnabled(ok)

    def _reset_tracker(self):
        self.tracker = None
        self.last_tracked_idx = -1
        self.track_cache.clear()
        self.traj_snapshots.clear()
        self.cmc_cache.clear()
        self.selected_idx = None
        self._pending_id_changes.clear()
        self._update_step_btn()
        self._update_export_btns()
        self._redraw()
        self._status("Tracker reset.")

    def _set_ui_locked(self, locked: bool):
        self._tracking_running = locked
        enabled = not locked
        self.open_video_btn.setEnabled(enabled)
        self.open_images_btn.setEnabled(enabled)
        self.prev_btn.setEnabled(enabled)
        self.next_btn.setEnabled(enabled)
        self.play_btn.setEnabled(enabled)
        self.pause_btn.setEnabled(enabled)
        self.frame_slider.setEnabled(enabled)
        self.reset_tracker_btn.setEnabled(enabled)
        self.edit_btn.setEnabled(enabled)
        self.delete_btn.setEnabled(enabled)
        self.id_spin.setEnabled(enabled and self.selected_idx is not None)
        self.conf_spin.setEnabled(enabled)
        self.frame_skip_spin.setEnabled(enabled)
        ok = enabled and bool(self.track_cache)
        self.export_btn.setEnabled(ok)
        self.export_data_btn.setEnabled(ok)

    def _step_tracker(self):
        if not self.source or self.current_idx <= self.last_tracked_idx:
            return
        if self._tracking_running:
            return
        if self.tracker is None:
            self.tracker = build_tracker(self._cfg())
            self.last_tracked_idx = -1
            self._status("Tracker initialised (BoTSORT AABB).")

        # ── Sync any user edits into the tracker before stepping ──
        self._sync_edits_to_tracker()

        start = self.last_tracked_idx + 1
        end = self.current_idx
        cfg = self._cfg()
        frame_skip = self.frame_skip_spin.value()

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
            model_path=cfg.get("model_path", YOLO_MODEL_PATH),
            conf=float(self.conf_spin.value()),
            imgsz=int(cfg.get("imgsz", 1024)),
            frame_skip=frame_skip,
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

    def _on_frame_tracked(self, idx, obbs):
        self.track_cache[idx] = list(obbs)
        self.last_tracked_idx = max(self.last_tracked_idx, idx)
        if idx == self.current_idx:
            self._redraw()

    def _on_traj_snapshot(self, idx, snap):
        self.traj_snapshots[idx] = dict(snap)
        if idx == self.current_idx:
            self._redraw()

    def _on_cmc_snapshot(self, idx, warp):
        if warp is not None:
            self.cmc_cache[idx] = np.asarray(warp, dtype=np.float64)
        else:
            self.cmc_cache[idx] = None

    def _on_progress(self, done, total):
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
        last_snap = self.traj_snapshots.get(self.last_tracked_idx, {})
        self._status(f"Tracked to frame {self.last_tracked_idx}. "
                     f"{len(last_snap)} unique IDs.")

    def _on_error(self, msg):
        self._set_ui_locked(False)
        self.step_btn.setText("Step Tracker ▶▶")
        self._update_step_btn()
        self._update_export_btns()
        self.progress_bar.setFormat("Error")
        self._status(f"Tracking error: {msg}")
        QtWidgets.QMessageBox.critical(self, "Tracking Error", msg)

    # ==================================================================
    # Video export
    # ==================================================================

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
            "Video (*.mp4);;All (*)")
        if not out_path:
            return
        fps = (self.source.fps() or 25)
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
            output_path=out_path, fps=fps,
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

    def _on_export_progress(self, done, total):
        pct = int(done / max(total, 1) * 100)
        self.progress_bar.setValue(pct)
        self.progress_bar.setFormat(f"Export {done}/{total}")

    def _on_export_done(self, path):
        self._exporting = False
        self._set_ui_locked(False)
        self.export_btn.setText("Export Video 🎬")
        self._update_export_btns()
        self.progress_bar.setFormat("Export done")
        self._status(f"Video exported → {path}")
        QtWidgets.QMessageBox.information(
            self, "Export complete", f"Video saved to:\n{path}")

    def _on_export_error(self, msg):
        self._exporting = False
        self._set_ui_locked(False)
        self.export_btn.setText("Export Video 🎬")
        self._update_export_btns()
        self.progress_bar.setFormat("Export error")
        self._status(f"Export error: {msg}")
        QtWidgets.QMessageBox.critical(self, "Export Error", msg)

    # ==================================================================
    # Data export  (per_frame/*.txt  +  per_track/*.json)
    # ==================================================================

    def _export_data(self):
        if not self.track_cache:
            return
        if self._tracking_running or self._exporting:
            return

        out_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select output folder for tracking data")
        if not out_dir:
            return

        pf_dir = Path(out_dir) / "per_frame"
        pt_dir = Path(out_dir) / "per_track"
        pf_dir.mkdir(parents=True, exist_ok=True)
        pt_dir.mkdir(parents=True, exist_ok=True)

        # ── Export CMC affine matrices (frame-to-frame warp) ──
        cmc_out: Dict[str, Any] = {}
        for fidx in sorted(self.cmc_cache.keys()):
            mat = self.cmc_cache[fidx]
            if mat is not None:
                cmc_out[str(fidx)] = mat.tolist()
            else:
                cmc_out[str(fidx)] = None
        if cmc_out:
            cmc_path = Path(out_dir) / "cmc_transforms.json"
            cmc_path.write_text(
                json.dumps(cmc_out, indent=2, ensure_ascii=False))

        # Collect per-track data while writing per-frame files
        per_track: Dict[int, list] = defaultdict(list)
        n_frames = 0
        n_detections = 0

        header = (
            "# track_id centroid_x centroid_y "
            "bbox_x1 bbox_y1 bbox_x2 bbox_y2 "
            "obb_x1 obb_y1 obb_x2 obb_y2 obb_x3 obb_y3 obb_x4 obb_y4 "
            "confidence class_id\n"
        )

        for frame_idx in sorted(self.track_cache.keys()):
            annots = self.track_cache[frame_idx]
            active = [b for b in annots if not b.deleted and b.track_id >= 0]
            if not active:
                continue

            lines = [header]
            for b in active:
                pts = b.poly.reshape(4, 2)
                cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
                bx1, by1 = float(pts[:, 0].min()), float(pts[:, 1].min())
                bx2, by2 = float(pts[:, 0].max()), float(pts[:, 1].max())
                obb_flat = " ".join(f"{pts[i, j]:.2f}"
                                    for i in range(4) for j in range(2))
                line = (
                    f"{b.track_id} "
                    f"{cx:.2f} {cy:.2f} "
                    f"{bx1:.2f} {by1:.2f} {bx2:.2f} {by2:.2f} "
                    f"{obb_flat} "
                    f"{b.conf:.4f} {b.cls_id}\n"
                )
                lines.append(line)

                # ── Compute OBB parametric form (cx, cy, w, h, angle) ──
                obb_xywhr = obb_to_xywhr(b.poly)

                # ── CMC warp matrix for this frame (if available) ──
                frame_cmc = self.cmc_cache.get(frame_idx)
                cmc_entry = frame_cmc.tolist() if frame_cmc is not None else None

                # Accumulate for per_track
                per_track[b.track_id].append({
                    "frame": frame_idx,
                    "centroid": [round(cx, 2), round(cy, 2)],
                    "bbox": [round(bx1, 2), round(by1, 2),
                             round(bx2, 2), round(by2, 2)],
                    "obb": [[round(float(pts[i, 0]), 2),
                             round(float(pts[i, 1]), 2)] for i in range(4)],
                    "obb_xywhr": {
                        "cx": obb_xywhr[0],
                        "cy": obb_xywhr[1],
                        "width": obb_xywhr[2],
                        "height": obb_xywhr[3],
                        "angle_deg": obb_xywhr[4],
                    },
                    "cmc_affine": cmc_entry,
                    "confidence": round(b.conf, 4),
                    "class_id": b.cls_id,
                })

            txt_path = pf_dir / f"frame_{frame_idx:06d}.txt"
            txt_path.write_text("".join(lines))
            n_frames += 1
            n_detections += len(active)

        # Write per-track JSON files
        for tid, detections in sorted(per_track.items()):
            record = {
                "track_id": tid,
                "num_detections": len(detections),
                "first_frame": detections[0]["frame"],
                "last_frame": detections[-1]["frame"],
                "detections": detections,
            }
            json_path = pt_dir / f"track_{tid:04d}.json"
            json_path.write_text(
                json.dumps(record, indent=2, ensure_ascii=False))

        n_tracks = len(per_track)
        n_cmc = sum(1 for v in self.cmc_cache.values() if v is not None)
        self._status(
            f"Exported {n_detections} detections across {n_frames} frames, "
            f"{n_tracks} tracks, {n_cmc} CMC matrices → {out_dir}")
        QtWidgets.QMessageBox.information(
            self, "Data export complete",
            f"Exported to: {out_dir}\n\n"
            f"per_frame/  → {n_frames} files\n"
            f"per_track/  → {n_tracks} JSON files\n"
            f"cmc_transforms.json → {n_cmc} matrices\n"
            f"Total detections: {n_detections}",
        )

    # ==================================================================
    # Selection / editing
    # ==================================================================

    def _toggle_edit(self):
        self.mode = "edit" if self.mode != "edit" else "select"
        self._status(f"Mode: {self.mode}")

    def _delete_selected(self):
        annots = self.track_cache.get(self.current_idx, [])
        if self.selected_idx is None or self.selected_idx >= len(annots):
            return
        annots[self.selected_idx].deleted = True
        self.selected_idx = None
        self._update_id_spin(); self._redraw()

    def _update_id_spin(self):
        annots = self.track_cache.get(self.current_idx, [])
        if (self.selected_idx is not None
                and 0 <= self.selected_idx < len(annots)
                and not annots[self.selected_idx].deleted):
            self.id_spin.setEnabled(True)
            self.id_spin.blockSignals(True)
            self.id_spin.setValue(annots[self.selected_idx].track_id)
            self.id_spin.blockSignals(False)
        else:
            self.id_spin.setEnabled(False)
            self.id_spin.blockSignals(True)
            self.id_spin.setValue(-1)
            self.id_spin.blockSignals(False)

    def _on_id_spin_changed(self, val):
        annots = self.track_cache.get(self.current_idx, [])
        if self.selected_idx is not None and self.selected_idx < len(annots):
            old_id = annots[self.selected_idx].track_id
            if old_id != val:
                annots[self.selected_idx].track_id = val
                # Remember old→new so _sync_edits_to_tracker can find the STrack
                if old_id >= 0:
                    self._pending_id_changes[old_id] = val
            self._redraw()

    def _pick_annot(self, x, y):
        annots = self.track_cache.get(self.current_idx, [])
        best, ba = None, None
        for i, b in enumerate(annots):
            if b.deleted: continue
            pts = b.poly.reshape(-1, 2).astype(np.float32)
            if cv2.pointPolygonTest(pts, (x, y), False) >= 0:
                a = cv2.contourArea(pts.astype(np.int32))
                if best is None or a < ba:
                    best, ba = i, a
        return best

    def _pick_vertex(self, x, y, tol=10):
        annots = self.track_cache.get(self.current_idx, [])
        if self.selected_idx is None or self.selected_idx >= len(annots): return None
        pts = annots[self.selected_idx].poly.reshape(-1, 2)
        for i in range(pts.shape[0]):
            if np.hypot(pts[i, 0] - x, pts[i, 1] - y) <= tol: return i
        return None

    def _translate_selected(self, dx, dy):
        annots = self.track_cache.get(self.current_idx, [])
        if self.selected_idx is None or self.selected_idx >= len(annots): return
        annots[self.selected_idx].poly = (
            self.orig_poly + np.array([dx, dy], dtype=np.float32)
        ).astype(np.float32)

    def _set_vertex(self, vi, x, y):
        annots = self.track_cache.get(self.current_idx, [])
        if self.selected_idx is None or self.selected_idx >= len(annots): return
        p = annots[self.selected_idx].poly.copy()
        p[vi] = [x, y]
        annots[self.selected_idx].poly = p.astype(np.float32)

    # ==================================================================
    # Event filter
    # ==================================================================

    def eventFilter(self, obj, event):
        if obj is not self.canvas:
            return super().eventFilter(obj, event)

        if self._tracking_running or self._exporting:
            if event.type() in (
                QtCore.QEvent.Type.MouseButtonPress,
                QtCore.QEvent.Type.MouseMove,
                QtCore.QEvent.Type.MouseButtonRelease,
                QtCore.QEvent.Type.MouseButtonDblClick,
                QtCore.QEvent.Type.Wheel,
            ):
                return True
            return super().eventFilter(obj, event)

        if event.type() == QtCore.QEvent.Type.Wheel:
            d = event.angleDelta().y()
            if d > 0: self.zoom_step(+1, anchor=event.position())
            elif d < 0: self.zoom_step(-1, anchor=event.position())
            return True

        if event.type() not in (
            QtCore.QEvent.Type.MouseButtonPress,
            QtCore.QEvent.Type.MouseMove,
            QtCore.QEvent.Type.MouseButtonRelease,
        ):
            return super().eventFilter(obj, event)

        if self.current_frame_bgr is None or not hasattr(event, "position"):
            return False
        pos = event.position()
        xd, yd = int(pos.x()), int(pos.y())
        xi, yi = self.display_to_image(xd, yd)
        if xi is None: return False

        if event.type() == QtCore.QEvent.Type.MouseButtonPress:
            if event.button() == QtCore.Qt.MouseButton.LeftButton and self.space_held:
                self._pan_dragging = True; self._pan_last = (xd, yd); return True
        if event.type() == QtCore.QEvent.Type.MouseMove:
            if getattr(self, "_pan_dragging", False):
                s = self.draw_map.get("scale", 1.0)
                self.pan_img[0] -= (xd - self._pan_last[0]) / s
                self.pan_img[1] -= (yd - self._pan_last[1]) / s
                self._pan_last = (xd, yd); self._redraw(); return True
        if event.type() == QtCore.QEvent.Type.MouseButtonRelease:
            if getattr(self, "_pan_dragging", False):
                self._pan_dragging = False; return True

        if event.type() == QtCore.QEvent.Type.MouseButtonPress:
            if event.button() == QtCore.Qt.MouseButton.LeftButton:
                hit = self._pick_annot(xi, yi)
                if hit is not None:
                    self.selected_idx = hit
                    self._update_id_spin(); self._redraw()
                    if self.mode == "edit":
                        v = self._pick_vertex(xi, yi)
                        if v is not None:
                            self.vertex_drag_idx = v; self.dragging = True; return True
                    annots = self.track_cache.get(self.current_idx, [])
                    if self.selected_idx < len(annots):
                        self.dragging = True
                        self.drag_start_img = (xi, yi)
                        self.orig_poly = annots[self.selected_idx].poly.copy()
                    return True
                else:
                    self.selected_idx = None
                    self._update_id_spin(); self._redraw(); return True

        if event.type() == QtCore.QEvent.Type.MouseMove and self.dragging:
            if self.vertex_drag_idx is not None:
                self._set_vertex(self.vertex_drag_idx, xi, yi)
            elif self.drag_start_img:
                self._translate_selected(xi - self.drag_start_img[0],
                                         yi - self.drag_start_img[1])
            self._redraw(); return True

        if event.type() == QtCore.QEvent.Type.MouseButtonRelease and self.dragging:
            self.dragging = False; self.vertex_drag_idx = None
            self.drag_start_img = None; self.orig_poly = None
            self._redraw(); return True

        return super().eventFilter(obj, event)

    def keyPressEvent(self, e):
        if self._tracking_running or self._exporting:
            return
        if e.key() == QtCore.Qt.Key.Key_Space: self.space_held = True
        else: super().keyPressEvent(e)

    def keyReleaseEvent(self, e):
        if self._tracking_running or self._exporting:
            return
        if e.key() == QtCore.Qt.Key.Key_Space: self.space_held = False
        else: super().keyReleaseEvent(e)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.current_frame_bgr is not None:
            self._redraw()