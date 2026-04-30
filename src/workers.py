import os
import sys
import io
from typing import List, Optional
import time

import numpy as np
from PySide6 import QtCore

# -------- YOLO (Ultralytics) --------
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# ------ Local imports ------
from .utils import OBBOX, rect_to_poly_xyxy


YOLO_MODEL_PATH = ""

# ---------------------------------------------------------------------------
# Default fallback weights — ultralytics auto-downloads these on first use.
# Used by `resolve_model_path` when the configured path is empty or missing.
# ---------------------------------------------------------------------------
DEFAULT_MODEL_DETECT = "yolo26m.pt"
DEFAULT_MODEL_OBB    = "yolo26m-obb.pt"


def resolve_model_path(model_path: str, task: str = "detect") -> str:
    """Return ``model_path`` if it points to an existing file on disk,
    otherwise a sensible default that ultralytics will auto-download.

    Args:
        model_path: User-configured path (may be empty or missing).
        task:       Either ``"obb"`` or ``"detect"`` (anything else is treated
                    as ``"detect"``).

    Defaults:
        task == "obb"  → ``DEFAULT_MODEL_OBB``  (currently ``yolo26m-obb.pt``)
        otherwise      → ``DEFAULT_MODEL_DETECT`` (currently ``yolo26m.pt``)
    """
    if model_path and os.path.isfile(model_path):
        return model_path
    fallback = DEFAULT_MODEL_OBB if task == "obb" else DEFAULT_MODEL_DETECT
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
# Detection (YOLO-OBB)
# ---------------------------------------------------------------------------
class DetectionWorker(QtCore.QObject):
    """Run oriented-bounding-box detection on a single frame using YOLO-OBB.

    Accepts EITHER:
      - source_path (str): path to an image file → passed directly to YOLO
      - frame_bgr (np.ndarray): BGR uint8 array (e.g. from video capture)

    When source_path is given, it takes priority (YOLO handles its own I/O).
    """
    finished = QtCore.Signal(object, object, object)
    error = QtCore.Signal(str)

    def __init__(
        self,
        frame_idx: int,
        frame_bgr: np.ndarray = None,
        conf: float = 0.5,
        imgsz: int = 1024,
        model_path: str = YOLO_MODEL_PATH,
        source_path: str = None,
    ):
        super().__init__()
        self.frame_idx = frame_idx
        self.frame_bgr = frame_bgr
        self.conf = conf
        self.model_path = model_path
        self.imgsz = imgsz
        self.source_path = source_path

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

    @QtCore.Slot()
    def run(self):
        try:
            model = self._get_model(self.model_path)

            # --- Choose source: file path preferred, numpy fallback ---
            if self.source_path and os.path.isfile(self.source_path):
                source = self.source_path
            elif self.frame_bgr is not None:
                # YOLO expects RGB uint8 when given a numpy array
                bgr = self.frame_bgr
                # Safety: ensure uint8 (basic conversion only)
                if bgr.dtype != np.uint8:
                    if bgr.dtype == np.uint16:
                        bgr = (bgr / 256).astype(np.uint8)
                    else:
                        bgr = bgr.astype(np.uint8)
                # YOLO's internal pipeline expects BGR (it does its own conversion)
                # Passing BGR directly — do NOT convert to RGB here
                source = bgr
            else:
                raise RuntimeError("No source_path and no frame_bgr provided.")

            # --- Predict ---
            results = model.predict(
                source=source,
                imgsz=self.imgsz,
                conf=self.conf,
                verbose=False,
            )
            res = results[0]
            names = getattr(model, "names", None)

            # --- Debug ---
            has_obb = (
                hasattr(res, "obb")
                and res.obb is not None
                and len(res.obb) > 0
            )
            has_boxes = res.boxes is not None and len(res.boxes) > 0

            boxes: List[OBBOX] = []

            # --- OBB path ---
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

            # --- AABB fallback ---
            elif has_boxes:
                xyxy = res.boxes.xyxy.cpu().numpy()
                C = res.boxes.cls.cpu().numpy()
                S = res.boxes.conf.cpu().numpy()
                for (x1, y1, x2, y2), c, s in zip(xyxy, C, S):
                    boxes.append(OBBOX(
                        poly=rect_to_poly_xyxy(x1, y1, x2, y2),
                        cls_id=int(c), conf=float(s),
                    ))

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
    """Build a YOLO-OBB dataset from verified polygons and fine-tune the model.

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
        base_model_path: str,
        out_root: Optional[str] = None,
        epochs: int = 20,
        imgsz: int = 1024,
        batch: int = 8,
        val_split: float = 0.1,
        seed: int = 1337,
        data_yaml: str = "datasets/datasets_build/dataset.yaml",
    ):
        super().__init__()
        self.class_names = class_names
        self.base_model_path = base_model_path
        self.out_root = out_root or os.path.join(os.getcwd(), "finetune_runs")
        self.epochs = int(epochs)
        self.imgsz = int(imgsz)
        self.batch = int(batch)
        self.val_split = float(val_split)
        self.seed = int(seed)
        self.data_yaml = data_yaml

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
            if not os.path.isfile(self.base_model_path):
                raise FileNotFoundError(
                    f"Base model not found: {self.base_model_path}"
                )
            if not self.class_names:
                raise ValueError(
                    "class_names is empty; cannot write dataset.yaml."
                )

            ts = time.strftime("%Y%m%d-%H%M%S")
            run_dir = os.path.join(self.out_root, f"run-{ts}")

            model = YOLO(self.base_model_path)

            # --- Register ultralytics callbacks for per-epoch progress ---
            total_epochs = self.epochs
            worker_ref = self  # prevent GC issues in closure

            def _on_fit_epoch_end(trainer):
                """Called by ultralytics at the end of each epoch (after val)."""
                epoch = trainer.epoch + 1
                metrics = {}

                # Collect available metrics from the trainer
                if hasattr(trainer, "metrics") and trainer.metrics:
                    for k, v in trainer.metrics.items():
                        try:
                            metrics[k] = float(v)
                        except (TypeError, ValueError):
                            pass

                # Also grab the last training loss values
                if (hasattr(trainer, "loss_items")
                        and trainer.loss_items is not None):
                    loss_names = getattr(trainer, "loss_names", None)
                    loss_vals = trainer.loss_items
                    if hasattr(loss_vals, "cpu"):
                        loss_vals = loss_vals.cpu().numpy()
                    if loss_names and len(loss_names) == len(loss_vals):
                        for name, val in zip(loss_names, loss_vals):
                            metrics[f"train/{name}"] = float(val)

                frac = epoch / total_epochs
                worker_ref.progress.emit(
                    f"Epoch {epoch}/{total_epochs}", frac
                )
                worker_ref.epoch_metrics.emit(epoch, total_epochs, metrics)

            model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)

            self.progress.emit("Starting training...", 0.0)
            self.log_line.emit(
                f"=== Training started: {total_epochs} epochs, "
                f"imgsz={self.imgsz}, batch={self.batch} ==="
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
                flipud=0.5,
                fliplr=0.5,
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