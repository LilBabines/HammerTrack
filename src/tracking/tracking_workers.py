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
    ensure_bgr_u8, parse_pose_result, rect_to_poly_xyxy,
    translate_annotation, two_stage_detect,
)
from ..tasks import TASK_OBB, TASK_POSE, normalize_task
from ..workers import resolve_model_path
from .tracking_helpers import (
    draw_tracked_annotations,
    extract_cmc_matrix,
    extract_trajectories_from_tracker,
    boxes_to_dets,
    iou_matrix,
    parse_track_results,
    suppress_duplicate_detections,
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
                 model_path, conf, imgsz, frame_skip=1, task=TASK_OBB,
                 input_nms: float = 0.7, two_stage: bool = False,
                 region_conf: float = 0.10, max_regions: int = 8):
        super().__init__()
        self.source = source
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.tracker = tracker
        self.model_path = model_path
        self.conf = conf
        self.imgsz = imgsz
        self.frame_skip = max(1, frame_skip)
        # Must match the tracker built by build_tracker(): the detection
        # layout fed to update() has to agree with the tracker's own mode.
        self.task = normalize_task(task)
        self._is_obb = self.task == TASK_OBB
        self._is_pose = self.task == TASK_POSE
        # Overlap above which a detection is dropped as a duplicate, before
        # the tracker ever sees it. 0 disables the pass.
        self.input_nms = float(input_nms)
        self._n_suppressed = 0
        # Two-stage: propose regions on the frame, then re-detect inside each
        # one at native resolution. Costs one forward pass per region, so it is
        # opt-in.
        self.two_stage = bool(two_stage)
        self.region_conf = float(region_conf)
        self.max_regions = int(max_regions)
        self._two_stage_stats = {}

    # ---- Class-level model cache (keyed by path) ----

    @classmethod
    def _get_model(cls, model_path: str):
        if not hasattr(cls, "_model") or cls._model_path != model_path:
            print(f"[TrackingStepWorker] Loading model: {model_path}")
            cls._model = YOLO(model_path)
            cls._model_path = model_path
        return cls._model

    # ---- OBB extraction from ultralytics result ----

    def _extract_obbs(self, res) -> List[OBBOX]:
        """Annotations from one ultralytics result, keypoints included.

        BoxMOT tracks the bounding box only — it has no notion of landmarks.
        That does not stop the keypoints from riding along: they are carried on
        the same ``OBBOX`` the tracker stamps its ID onto, so display and
        export get them for free while the motion model keeps working on boxes.
        """
        if self._is_pose:
            # Shared with DetectionWorker so both paths apply the identical
            # "all points at the origin means no keypoints" guard.
            return parse_pose_result(res)

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

    # ---- Two-stage helpers ----

    def _run_two_stage(self, model, frame: np.ndarray) -> List[OBBOX]:
        """Region proposal then native-resolution refinement on one frame."""
        def predict_full(image: np.ndarray) -> List[OBBOX]:
            res = model.predict(source=image, imgsz=self.imgsz,
                                conf=self.region_conf, verbose=False)
            return self._extract_obbs(res[0])

        def predict_crop(image: np.ndarray) -> List[OBBOX]:
            res = model.predict(source=image, imgsz=self.imgsz,
                                conf=self.conf, verbose=False)
            return self._extract_obbs(res[0])

        boxes, stats = two_stage_detect(
            frame,
            predict_frame=predict_full,     # permissive: proposals only
            predict_region=predict_crop,    # the user's real conf threshold
            imgsz=self.imgsz,
            nms_threshold=max(self.input_nms, 0.1),
            max_regions=self.max_regions,
            # Proposals the refinement missed must still clear the real
            # threshold: a 0.10-confidence leftover fed to BoxMOT spawns a
            # phantom track that nothing downstream can remove.
            keep_conf=self.conf,
            translate=translate_annotation,
        )
        for key, value in stats.items():
            self._two_stage_stats[key] = (
                self._two_stage_stats.get(key, 0) + value
            )
        return boxes

    # ---- Detection → track-ID assignment ----

    @staticmethod
    def _assign_ids(obbs, det_aabbs, tracks):
        """Map tracker IDs back onto the source annotations.

        Layout-agnostic: ``parse_track_results`` hands back ids, det indices
        and axis-aligned boxes whether the tracker ran in OBB or AABB mode.
        """
        ids, det_inds, trk_boxes = parse_track_results(tracks)
        if len(ids) == 0:
            return obbs

        # Preferred path: the tracker tells us which detection each track came
        # from, so the mapping is exact.
        if (det_inds >= 0).any():
            for row, di in enumerate(det_inds):
                if 0 <= di < len(obbs):
                    obbs[di].track_id = int(ids[row])
            return obbs

        # Fallback for coasting-only frames: greedy IoU match.
        if not obbs or len(det_aabbs) == 0:
            return obbs
        ious = iou_matrix(det_aabbs[:, :4], trk_boxes)
        used = set()
        for ti in range(len(trk_boxes)):
            best_det = int(ious[:, ti].argmax())
            if best_det not in used and ious[best_det, ti] > 0.3:
                obbs[best_det].track_id = int(ids[ti])
                used.add(best_det)
        return obbs

    # ---- Run loop ----

    @QtCore.Slot()
    def run(self):
        try:
            if YOLO is None:
                raise RuntimeError("ultralytics not installed")

            model_path = resolve_model_path(self.model_path, self.task)
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
                if self.two_stage:
                    obbs = self._run_two_stage(model, frame)
                else:
                    results = model.predict(
                        source=frame,
                        imgsz=self.imgsz,
                        conf=self.conf,
                        verbose=False,
                    )
                    obbs = self._extract_obbs(results[0])

                # Deduplicate BEFORE the tracker. Two boxes on one shark would
                # otherwise each claim a track ID, so the same animal ends up
                # with several IDs, several skeletons and several trails — and
                # no later stage can undo that, because the identities are
                # already distinct by then.
                obbs, removed = suppress_duplicate_detections(
                    obbs, self.input_nms
                )
                self._n_suppressed += removed

                # Feed the tracker in its own layout: oriented rows keep the
                # angle inside the motion model instead of flattening it.
                dets = boxes_to_dets(obbs, self._is_obb)
                tracks = self.tracker.update(dets, frame)

                # The IoU fallback always works on axis-aligned boxes.
                det_aabbs = (boxes_to_dets(obbs, False) if obbs
                             else np.empty((0, 6), dtype=np.float32))
                obbs = self._assign_ids(obbs, det_aabbs, tracks)

                snap = extract_trajectories_from_tracker(self.tracker)
                warp = extract_cmc_matrix(self.tracker)

                self.frame_tracked.emit(idx, obbs)
                self.traj_snapshot.emit(idx, snap)
                self.cmc_snapshot.emit(idx, warp)
                self.progress.emit(i + 1, total)

            if self._n_suppressed:
                print(f"[TrackingStepWorker] input_nms={self.input_nms}: "
                      f"{self._n_suppressed} duplicate detection(s) dropped "
                      f"over {total} frame(s)")
            if self.two_stage and self._two_stage_stats:
                st = self._two_stage_stats
                print(f"[TrackingStepWorker] two-stage over {total} frame(s): "
                      f"{st.get('regions', 0)} region(s), "
                      f"pass1={st.get('pass1', 0)} pass2={st.get('pass2', 0)}, "
                      f"{st.get('kept_pass1', 0)} kept unrefined, "
                      f"{st.get('below_conf', 0)} below conf, "
                      f"{st.get('edge_dropped', 0)} edge-clipped dropped, "
                      f"{st.get('suppressed', 0)} duplicate(s) removed")
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