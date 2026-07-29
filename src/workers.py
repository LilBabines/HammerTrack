import os
import sys
import io
from typing import List, Optional
import time

import cv2
import numpy as np
from PySide6 import QtCore

# -------- YOLO (Ultralytics) --------
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# ------ Local imports ------
from .tasks import (
    TASK_DETECT, TASK_POSE,
    default_model_for, is_pretrained_name, normalize_task, validate_pretrained,
)
from .utils import (
    OBBOX, ensure_bgr_u8, parse_pose_result, rect_to_poly_xyxy,
    translate_annotation, two_stage_detect,
)

_RESERVED_TRAIN_ARGS = frozenset({
        "data", "epochs", "imgsz", "batch", "project", "name",
        "exist_ok", "verbose", "seed", "task", "model", "resume",
    })


def resolve_model_path(model_path: str, task: str = TASK_DETECT) -> str:
    """Return usable inference weights for ``task``.

    ``model_path`` wins when it points at an existing file (fine-tuned weights)
    or is already an official pretrained name that ultralytics can download on
    demand. Otherwise the task's default pretrained checkpoint is used.
    """
    if model_path and os.path.isfile(model_path):
        return model_path
    if is_pretrained_name(model_path):
        # Not on disk yet, but ultralytics fetches it on first use.
        return model_path

    fallback = default_model_for(task)
    if model_path:
        print(
            f"[resolve_model_path] '{model_path}' not found — "
            f"falling back to '{fallback}' (task={task})"
        )
    return fallback


# ---------------------------------------------------------------------------
# Stdout capture helper — thread-safe relay to a Qt signal
# ---------------------------------------------------------------------------

class _StdoutCapture(io.TextIOBase):
    """Captures writes to stdout and relays each line via a Qt signal,
    while still forwarding to the original stdout."""

    def __init__(self, signal: QtCore.SignalInstance, original_stdout):
        super().__init__()
        self._signal = signal
        self._original = original_stdout

    def write(self, text: str):
        if self._original:
            self._original.write(text)
        if text and text.strip():
            self._signal.emit(text.rstrip("\n"))
        return len(text) if text else 0

    def flush(self):
        if self._original:
            self._original.flush()

    def isatty(self):
        return False


# ---------------------------------------------------------------------------
# Detection (YOLO detect / obb / pose)
# ---------------------------------------------------------------------------
class DetectionWorker(QtCore.QObject):
    """Run detection on a single frame using YOLO.

    Accepts EITHER:
      - source_path (str): path to an image file → passed directly to YOLO
      - frame_bgr (np.ndarray): BGR uint8 array (e.g. from video capture)

    In single-pass mode ``source_path`` takes priority, since YOLO then handles
    its own I/O and decoding. In two-stage mode a numpy array is required —
    regions have to be cropped out of actual pixels — so a path-only source is
    read here instead.
    """
    finished = QtCore.Signal(object, object, object)
    error = QtCore.Signal(str)

    def __init__(
        self,
        frame_idx: int,
        frame_bgr: np.ndarray = None,
        conf: float = 0.5,
        imgsz: int = 1024,
        model_path: str = "",
        source_path: str = None,
        task: str = TASK_DETECT,
        two_stage: bool = False,
        region_conf: float = 0.10,
        max_regions: int = 8,
    ):
        super().__init__()
        self.frame_idx = frame_idx
        self.frame_bgr = frame_bgr
        self.conf = conf
        self.model_path = model_path
        self.imgsz = imgsz
        self.source_path = source_path
        self.task = normalize_task(task)
        self.two_stage = bool(two_stage)
        self.region_conf = float(region_conf)
        self.max_regions = int(max_regions)

    @classmethod
    def clear_model_cache(cls):
        """Drop the cached model (and free its GPU memory).

        The cache lives on the class and is shared by every page, so it has to
        be dropped when the project changes: the next project may well be a
        different task, and keeping the old weights resident wastes VRAM.
        """
        for attr in ("_model", "_model_path", "_model_task"):
            if hasattr(cls, attr):
                delattr(cls, attr)

    @classmethod
    def _get_model(cls, model_path: str):
        """Return a cached YOLO model, loading it only when the path changes."""
        if not hasattr(cls, "_model") or cls._model_path != model_path:
            print(f"[DetectionWorker] Loading model: {model_path}")
            cls._model = YOLO(model_path)
            cls._model_path = model_path
            cls._model_task = getattr(cls._model, "task", "detect")
            print(f"[DetectionWorker] Model task: {cls._model_task}")
        return cls._model

    # ------------------------------------------------------------------
    # Result → annotations
    # ------------------------------------------------------------------

    def _extract(self, res) -> List[OBBOX]:
        """Annotations from one ultralytics result, whatever the task.

        Pulled out of ``run()`` because the two-stage path runs it once per
        region as well as once on the whole frame. A single entry point also
        guarantees the proposal pass and the refinement pass read a result the
        same way.
        """
        has_obb = (
            hasattr(res, "obb")
            and res.obb is not None
            and len(res.obb) > 0
        )
        has_boxes = res.boxes is not None and len(res.boxes) > 0

        # --- Pose path: axis-aligned box + keypoints ---
        if self.task == TASK_POSE and has_boxes:
            # Shared with the tracking worker, so both apply the identical
            # "all points at the origin means no keypoints" guard.
            return parse_pose_result(res)

        if has_obb:
            return self._extract_obb(res.obb)

        # --- AABB fallback ---
        if has_boxes:
            return self._extract_aabb(res.boxes)

        return []

    @staticmethod
    def _extract_obb(obb) -> List[OBBOX]:
        """Oriented boxes, from the 4-point form or the xywhr fallback."""
        boxes: List[OBBOX] = []
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
            return boxes

        xywhr = getattr(obb, "xywhr", None)
        if xywhr is None or len(xywhr) == 0:
            return boxes

        X = (xywhr.cpu().numpy() if hasattr(xywhr, "cpu")
             else np.asarray(xywhr))
        C = (cls.cpu().numpy() if hasattr(cls, "cpu") else np.zeros(len(X)))
        S = (conf_vals.cpu().numpy() if hasattr(conf_vals, "cpu")
             else np.ones(len(X)))
        for (cx, cy, w, h, rad), c, s in zip(X, C, S):
            rect = np.array(
                [[-w / 2, -h / 2], [w / 2, -h / 2],
                 [w / 2, h / 2], [-w / 2, h / 2]],
                dtype=np.float32,
            )
            cos_r, sin_r = np.cos(rad), np.sin(rad)
            R = np.array([[cos_r, -sin_r], [sin_r, cos_r]], dtype=np.float32)
            pts = rect @ R.T + np.array([cx, cy], dtype=np.float32)
            boxes.append(OBBOX(poly=pts, cls_id=int(c), conf=float(s)))
        return boxes

    @staticmethod
    def _extract_aabb(res_boxes) -> List[OBBOX]:
        """Axis-aligned boxes: detect task, or a pose model with no keypoints."""
        boxes: List[OBBOX] = []
        xyxy = res_boxes.xyxy.cpu().numpy()
        C = res_boxes.cls.cpu().numpy()
        S = res_boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), c, s in zip(xyxy, C, S):
            boxes.append(OBBOX(
                poly=rect_to_poly_xyxy(x1, y1, x2, y2),
                cls_id=int(c), conf=float(s),
            ))
        return boxes

    # ------------------------------------------------------------------
    # Sources
    # ------------------------------------------------------------------

    @staticmethod
    def _as_u8(img: np.ndarray) -> np.ndarray:
        """Minimal dtype guard. YOLO wants uint8 BGR — no RGB swap here."""
        if img.dtype == np.uint8:
            return img
        if img.dtype == np.uint16:
            return (img / 256).astype(np.uint8)
        return img.astype(np.uint8)

    def _resolve_source(self):
        """Single-pass source: the file path when available, else the array."""
        if self.source_path and os.path.isfile(self.source_path):
            return self.source_path
        if self.frame_bgr is not None:
            return self._as_u8(self.frame_bgr)
        raise RuntimeError("No source_path and no frame_bgr provided.")

    def _resolve_frame(self) -> np.ndarray:
        """Two-stage source: always an array, since regions must be cropped.

        Letting YOLO read the file itself is cheaper, but then there are no
        pixels here to cut the regions out of — so a path-only source is decoded
        and normalised at this point instead.
        """
        if self.frame_bgr is not None:
            return self._as_u8(self.frame_bgr)
        if self.source_path and os.path.isfile(self.source_path):
            img = cv2.imread(self.source_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise RuntimeError(f"Cannot read image: {self.source_path}")
            return ensure_bgr_u8(img)
        raise RuntimeError("No source_path and no frame_bgr provided.")

    def _predict(self, model, image: np.ndarray, conf: float) -> List[OBBOX]:
        res = model.predict(
            source=image, imgsz=self.imgsz, conf=conf, verbose=False,
        )
        return self._extract(res[0])

    # ------------------------------------------------------------------

    @QtCore.Slot()
    def run(self):
        try:
            model = self._get_model(self.model_path)
            names = getattr(model, "names", None)

            if self.two_stage:
                frame = self._resolve_frame()
                boxes, stats = two_stage_detect(
                    frame,
                    # Permissive: the first pass only has to notice something
                    # is there, and its boxes are proposals, not results.
                    predict_frame=lambda img: self._predict(
                        model, img, self.region_conf),
                    # The refinement pass uses the real threshold.
                    predict_region=lambda img: self._predict(
                        model, img, self.conf),
                    imgsz=self.imgsz,
                    max_regions=self.max_regions,
                    keep_conf=self.conf,
                    translate=translate_annotation,
                )
                print(
                    f"[DetectionWorker] two-stage frame {self.frame_idx}: "
                    f"{stats['regions']} region(s), pass1={stats['pass1']} "
                    f"pass2={stats['pass2']}, "
                    f"{stats['kept_pass1']} kept unrefined, "
                    f"{stats['below_conf']} below conf, "
                    f"{stats['edge_dropped']} edge-clipped, "
                    f"{stats['suppressed']} duplicate(s) removed"
                )
            else:
                res = model.predict(
                    source=self._resolve_source(),
                    imgsz=self.imgsz,
                    conf=self.conf,
                    verbose=False,
                )[0]
                boxes = self._extract(res)

            print(f"[DetectionWorker] Emitting {len(boxes)} boxes")
            self.finished.emit(self.frame_idx, names, boxes)

        except Exception as e:
            import traceback
            print(f"[DetectionWorker] EXCEPTION:\n{traceback.format_exc()}")
            self.error.emit(str(e))


# ---------------------------------------------------------------------------
# Detection fine-tuning
# ---------------------------------------------------------------------------

class DetectFinetuneWorker(QtCore.QObject):
    """Build a YOLO dataset from verified polygons and fine-tune the model.

    Signals:
        progress(str, float)          — message + progress in [0, 1]
        epoch_metrics(int, int, dict) — current_epoch, total_epochs, metrics
        log_line(str)                 — a line of console output
        finished(str)                 — path to best.pt
        error(str)
    """
    progress = QtCore.Signal(str, float)
    epoch_metrics = QtCore.Signal(int, int, object)
    log_line = QtCore.Signal(str)
    finished = QtCore.Signal(str)
    error = QtCore.Signal(str)

    def __init__(
        self,
        class_names: List[str],
        base_model: str,
        data_yaml: str,
        task: str = TASK_DETECT,
        out_root: Optional[str] = None,
        epochs: int = 20,
        imgsz: int = 1024,
        batch: int = 8,
        overrides: Optional[dict] = None,
        seed: int = 1337,
    ):
        super().__init__()
        self.class_names = class_names
        # Always an official pretrained checkpoint name, never custom weights
        # (see the validation in run()).
        self.base_model = base_model
        self.task = normalize_task(task)
        self.out_root = out_root or os.path.join(os.getcwd(), "finetune_runs")
        self.epochs = int(epochs)
        self.imgsz = int(imgsz)
        self.batch = int(batch)
        self.overrides = dict(overrides or {})
        self.seed = int(seed)
        self.data_yaml = data_yaml

    def _warn_if_flip_idx_missing(self):
        """Warn when data.yaml lacks flip_idx, which mutes flip augmentation."""
        try:
            with open(self.data_yaml, "r", encoding="utf-8") as f:
                content = f.read()
        except OSError:
            return
        if "flip_idx" not in content:
            self.log_line.emit(
                "WARNING: data.yaml has no 'flip_idx' — ultralytics will "
                "disable flip augmentation for this pose run."
            )

    @QtCore.Slot()
    def run(self):
        # Capture stdout so ultralytics console output goes to the GUI
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = _StdoutCapture(self.log_line, original_stdout)
        sys.stderr = _StdoutCapture(self.log_line, original_stderr)

        try:
            if YOLO is None:
                raise RuntimeError(
                    "Ultralytics is not installed. `pip install ultralytics`"
                )
            if not self.class_names:
                raise ValueError(
                    "class_names is empty; cannot write dataset.yaml."
                )
            if not os.path.isfile(self.data_yaml):
                raise FileNotFoundError(
                    f"Dataset config not found: {self.data_yaml}\n"
                    f"Export the dataset before training."
                )

            # Fine-tuning must always restart from the official pretrained
            # backbone. Training on top of a previous run's weights would
            # compound drift across active-learning iterations.
            ok, reason = validate_pretrained(self.base_model, self.task)
            if not ok:
                raise ValueError(reason)

            ts = time.strftime("%Y%m%d-%H%M%S")
            run_dir = os.path.join(self.out_root, f"run-{ts}")

            # Bare official names are downloaded by ultralytics on first use.
            self.log_line.emit(
                f"=== Base model: {self.base_model} "
                f"(pretrained, task={self.task}) ==="
            )
            model = YOLO(self.base_model)

            loaded_task = getattr(model, "task", None)
            if loaded_task and loaded_task != self.task:
                raise ValueError(
                    f"Loaded model reports task '{loaded_task}' but the "
                    f"project task is '{self.task}'."
                )

            # --- Register ultralytics callbacks for per-epoch progress ---
            total_epochs = self.epochs
            worker_ref = self  # prevent GC issues in closure

            def _on_fit_epoch_end(trainer):
                epoch = trainer.epoch + 1
                metrics = {}

                for k, v in (getattr(trainer, "metrics", None) or {}).items():
                    try:
                        metrics[str(k)] = float(v)
                    except (TypeError, ValueError):
                        continue

                # tloss = epoch mean; loss_items is the last batch only.
                tloss = getattr(trainer, "tloss", None)
                if tloss is not None:
                    try:
                        # dict {'train/box_loss': float, ...}, but a list of
                        # names when tloss is None — hence the isinstance guard.
                        items = trainer.label_loss_items(tloss)
                        if isinstance(items, dict):
                            for k, v in items.items():
                                try:
                                    metrics[str(k)] = float(v)
                                except (TypeError, ValueError):
                                    continue
                    except Exception:
                        pass

                worker_ref.progress.emit(
                    f"Epoch {epoch}/{total_epochs}", epoch / total_epochs
                )
                worker_ref.epoch_metrics.emit(epoch, total_epochs, metrics)

            model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)

            self.progress.emit("Starting training...", 0.0)
            self.log_line.emit(
                f"=== Training started: {total_epochs} epochs, "
                f"imgsz={self.imgsz}, batch={self.batch} ==="
            )

            # Flips are safe for pose too: ultralytics remaps keypoints with
            # data.yaml's flip_idx on BOTH the vertical and horizontal flip.
            # It silently disables them when flip_idx is missing, so the check
            # below turns that into a visible warning instead.
            if self.task == TASK_POSE:
                self._warn_if_flip_idx_missing()

            safe = {k: v for k, v in self.overrides.items()
                    if k not in _RESERVED_TRAIN_ARGS}
            dropped = sorted(set(self.overrides) - set(safe))
            if dropped:
                self.log_line.emit(
                    f"WARNING: ignoring reserved train() override(s): "
                    f"{', '.join(dropped)}"
                )
            if safe:
                self.log_line.emit(
                    "=== Overrides: "
                    + ", ".join(f"{k}={v}" for k, v in sorted(safe.items()))
                    + " ==="
                )

            model.train(
                data=self.data_yaml,
                epochs=self.epochs,
                imgsz=self.imgsz,
                batch=self.batch,
                project=run_dir,
                name="finetune",
                exist_ok=True,
                verbose=True,
                seed=self.seed,
                **safe,
            )

            # Locate best weights
            weights_dir = os.path.join(run_dir, "finetune", "weights")
            best_pt = os.path.join(weights_dir, "best.pt")
            if not os.path.isfile(best_pt):
                last_pt = os.path.join(weights_dir, "last.pt")
                if os.path.isfile(last_pt):
                    best_pt = last_pt
                else:
                    raise RuntimeError(
                        "Training finished but no weights found."
                    )

            self.progress.emit("Training complete!", 1.0)
            self.log_line.emit(
                f"=== Training complete — weights: {best_pt} ==="
            )
            self.finished.emit(best_pt)

        except Exception as e:
            self.error.emit(str(e))
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr