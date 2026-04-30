"""Tracking workers — Qt QObjects that run in background threads.

* :class:`TrackingStepWorker` runs YOLO + tracker for a range of frames,
  emitting per-frame results.
* :class:`VideoExportWorker` re-renders the cached tracking output to an
  ``.mp4`` file.

Both workers cache the YOLO model on the class so successive runs avoid
reloading weights.
"""

from typing import Dict, List, Tuple

import cv2
import numpy as np
from PySide6 import QtCore

from ..utils import (
    OBBOX, FrameSource,
    ensure_bgr_u8, rect_to_poly_xyxy,
)
from ..workers import resolve_model_path
from .tracking_helpers import (
    draw_tracked_annotations,
    extract_cmc_matrix,
    extract_trajectories_from_tracker,
    iou_matrix,
    obb_to_aabb_row,
)

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


# ==================== Tracking step worker ====================

class TrackingStepWorker(QtCore.QObject):
    """Run YOLO detection + tracker update over ``[start_idx, end_idx]``."""

    frame_tracked = QtCore.Signal(int, object)
    traj_snapshot = QtCore.Signal(int, object)
    cmc_snapshot  = QtCore.Signal(int, object)   # frame_idx, 2×3/3×3 ndarray
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

    # ---- Class-level model cache (keyed by path) ----

    @classmethod
    def _get_model(cls, model_path: str):
        if not hasattr(cls, "_model") or cls._model_path != model_path:
            print(f"[TrackingStepWorker] Loading model: {model_path}")
            cls._model = YOLO(model_path)
            cls._model_path = model_path
        return cls._model

    # ---- OBB extraction from ultralytics result ----

    @staticmethod
    def _extract_obbs(res) -> List[OBBOX]:
        boxes: List[OBBOX] = []
        has_obb = (
            hasattr(res, "obb")
            and res.obb is not None
            and len(res.obb) > 0
        )
        if has_obb:
            obb = res.obb
            polys = getattr(obb, "xyxyxyxy", None)
            cls = getattr(obb, "cls", None)
            conf_vals = getattr(obb, "conf", None)
            if polys is not None and len(polys) > 0:
                P = (polys.cpu().numpy() if hasattr(polys, "cpu")
                     else np.asarray(polys))
                C = (cls.cpu().numpy() if hasattr(cls, "cpu")
                     else np.zeros(len(P)))
                S = (conf_vals.cpu().numpy() if hasattr(conf_vals, "cpu")
                     else np.ones(len(P)))
                for p, c, s in zip(P, C, S):
                    boxes.append(OBBOX(
                        poly=p.reshape(4, 2).astype(np.float32),
                        cls_id=int(c), conf=float(s),
                    ))
            else:
                xywhr = getattr(obb, "xywhr", None)
                if xywhr is not None and len(xywhr) > 0:
                    X = (xywhr.cpu().numpy() if hasattr(xywhr, "cpu")
                         else np.asarray(xywhr))
                    C = (cls.cpu().numpy() if hasattr(cls, "cpu")
                         else np.zeros(len(X)))
                    S = (conf_vals.cpu().numpy()
                         if hasattr(conf_vals, "cpu") else np.ones(len(X)))
                    for (cx, cy, w, h, rad), c, s in zip(X, C, S):
                        rect = np.array(
                            [[-w / 2, -h / 2], [w / 2, -h / 2],
                             [w / 2, h / 2], [-w / 2, h / 2]],
                            dtype=np.float32,
                        )
                        cos_r, sin_r = np.cos(rad), np.sin(rad)
                        R = np.array(
                            [[cos_r, -sin_r], [sin_r, cos_r]],
                            dtype=np.float32,
                        )
                        pts = rect @ R.T + np.array(
                            [cx, cy], dtype=np.float32
                        )
                        boxes.append(OBBOX(
                            poly=pts, cls_id=int(c), conf=float(s),
                        ))
        elif res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            C = res.boxes.cls.cpu().numpy()
            S = res.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), c, s in zip(xyxy, C, S):
                boxes.append(OBBOX(
                    poly=rect_to_poly_xyxy(x1, y1, x2, y2),
                    cls_id=int(c), conf=float(s),
                ))
        return boxes

    # ---- Detection → track-ID assignment ----

    @staticmethod
    def _assign_ids(obbs, det_aabbs, tracks):
        if tracks is None or len(tracks) == 0:
            return obbs
        trk_ids = tracks[:, 4].astype(int)
        # Preferred path: BoxMOT returns the original detection index in col 7
        if tracks.shape[1] >= 8:
            det_indices = tracks[:, 7].astype(int)
            for row, di in enumerate(det_indices):
                if 0 <= di < len(obbs):
                    obbs[di].track_id = int(trk_ids[row])
            return obbs
        # Fallback: greedy IoU match between detections and tracks
        if not obbs or len(det_aabbs) == 0:
            return obbs
        trk_boxes = tracks[:, :4]
        ious = iou_matrix(det_aabbs[:, :4], trk_boxes)
        used = set()
        for ti in range(len(trk_boxes)):
            best_det = int(ious[:, ti].argmax())
            if best_det not in used and ious[best_det, ti] > 0.3:
                obbs[best_det].track_id = int(trk_ids[ti])
                used.add(best_det)
        return obbs

    # ---- Run loop ----

    @QtCore.Slot()
    def run(self):
        try:
            if YOLO is None:
                raise RuntimeError("ultralytics not installed")

            model_path = resolve_model_path(self.model_path, "obb")
            model = self._get_model(model_path)

            frames = list(range(self.start_idx, self.end_idx + 1, self.frame_skip))
            if frames and frames[-1] != self.end_idx:
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
                results = model.predict(
                    source=frame,
                    imgsz=self.imgsz,
                    conf=self.conf,
                    verbose=False,
                )
                obbs = self._extract_obbs(results[0])
                if obbs:
                    det_aabbs = np.stack([obb_to_aabb_row(b) for b in obbs])
                else:
                    det_aabbs = np.empty((0, 6), dtype=np.float32)

                tracks = self.tracker.update(det_aabbs, frame)
                obbs = self._assign_ids(obbs, det_aabbs, tracks)

                snap = extract_trajectories_from_tracker(self.tracker)
                warp = extract_cmc_matrix(self.tracker)

                self.frame_tracked.emit(idx, obbs)
                self.traj_snapshot.emit(idx, snap)
                self.cmc_snapshot.emit(idx, warp)
                self.progress.emit(i + 1, total)

            self.finished.emit()

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


# ==================== Video export worker ====================

class VideoExportWorker(QtCore.QObject):
    """Render every tracked frame with annotations + trajectories to a .mp4."""

    progress = QtCore.Signal(int, int)
    finished = QtCore.Signal(str)
    error    = QtCore.Signal(str)

    def __init__(
        self,
        source: FrameSource,
        track_cache: Dict[int, List[OBBOX]],
        traj_snapshots: Dict[int, Dict[int, List[Tuple[float, float]]]],
        output_path: str,
        fps: float,
        trail_length: int = 60,
        show_trails: bool = True,
    ):
        super().__init__()
        self.source = source
        self.track_cache = track_cache
        self.traj_snapshots = traj_snapshots
        self.output_path = output_path
        self.fps = fps
        self.trail_length = trail_length
        self.show_trails = show_trails

    @staticmethod
    def _closest_earlier(idx: int, keys: list):
        """Binary search: largest key in ``keys`` that is ``<= idx``."""
        lo, hi, best = 0, len(keys) - 1, None
        while lo <= hi:
            mid = (lo + hi) // 2
            if keys[mid] <= idx:
                best = keys[mid]
                lo = mid + 1
            else:
                hi = mid - 1
        return best

    @QtCore.Slot()
    def run(self):
        try:
            if not self.track_cache:
                self.error.emit("Nothing to export — run the tracker first.")
                return

            last_idx = max(self.track_cache.keys())
            total = last_idx + 1

            sample = self.source.read(0)
            if sample is None:
                self.error.emit("Cannot read first frame.")
                return
            sample = ensure_bgr_u8(sample)
            h, w = sample.shape[:2]

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                self.output_path, fourcc, self.fps, (w, h)
            )
            if not writer.isOpened():
                self.error.emit(
                    f"Cannot open VideoWriter for {self.output_path}"
                )
                return

            sorted_keys = sorted(self.track_cache.keys())

            for idx in range(total):
                frame = self.source.read(idx)
                if frame is None:
                    writer.write(np.zeros((h, w, 3), dtype=np.uint8))
                    self.progress.emit(idx + 1, total)
                    continue

                frame = ensure_bgr_u8(frame)
                annots = self.track_cache.get(idx, [])
                snap_key = self._closest_earlier(idx, sorted_keys)
                trajectories = (
                    self.traj_snapshots.get(snap_key, {})
                    if snap_key is not None else {}
                )
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
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))
