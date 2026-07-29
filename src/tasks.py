"""
Task definitions shared across the app.

HammerTrack supports three YOLO tasks. The task is chosen when a project is
created and is **immutable** afterwards: annotations, label files and model
weights are all task-specific, so switching would silently invalidate an
existing dataset. The task is baked into the project folder name
(``<name>_<task>``) so it stays obvious on disk.

* ``detect`` — axis-aligned boxes      → label: ``cls cx cy w h``
* ``obb``    — oriented boxes          → label: ``cls x1 y1 x2 y2 x3 y3 x4 y4``
* ``pose``   — box + N keypoints       → label: ``cls cx cy w h x1 y1 ... xN yN``
"""

import re
from typing import List, Optional, Tuple

TASK_DETECT = "detect"
TASK_OBB = "obb"
TASK_POSE = "pose"

TASKS = (TASK_DETECT, TASK_OBB, TASK_POSE)

TASK_LABELS = {
    TASK_DETECT: "detect — axis-aligned boxes (HBB)",
    TASK_OBB: "obb — oriented boxes",
    TASK_POSE: "pose — box + keypoints",
}

# Filename suffix ultralytics uses for each task's pretrained checkpoints.
TASK_SUFFIX = {
    TASK_DETECT: "",
    TASK_OBB: "-obb",
    TASK_POSE: "-pose",
}

# Model scales offered for fine-tuning, smallest to largest.
MODEL_SCALES = ("n", "s", "m", "l", "x")
MODEL_FAMILY = "yolo26"
DEFAULT_SCALE = "m"

# Default ultralytics weights per task (auto-downloaded on first use).
DEFAULT_MODELS = {
    task: f"{MODEL_FAMILY}{DEFAULT_SCALE}{suffix}.pt"
    for task, suffix in TASK_SUFFIX.items()
}

# Official ultralytics checkpoint naming, e.g. "yolo26m-pose.pt", "yolo11n.pt".
# Anything that does not match is treated as custom weights.
_PRETRAINED_RE = re.compile(
    r"^yolo(?:26|11|v10|v9|v8|v5)([nsmlx])u?(-obb|-pose|-seg|-cls)?\.pt$"
)

# Keypoints are stored as (x, y) only: every annotated point is considered
# visible, so the third "visibility" dimension carries no information here.
KPT_DIMS = 2

# How the bounding box of a pose instance is obtained.
POSE_BBOX_AUTO = "auto"      # derived from the keypoint extent
POSE_BBOX_MANUAL = "manual"  # drawn by hand before placing the keypoints
POSE_BBOX_MODES = (POSE_BBOX_AUTO, POSE_BBOX_MANUAL)


# Training knobs forwarded to ultralytics ``train()``, with the defaults this
# tool applies when a project config does not carry them. Kept here, and not
# in the settings panel, so the panel and the training page cannot disagree:
# a project saved before these settings existed must still train the way the
# UI claims it will.
#
# These are NOT the ultralytics defaults. ``scale`` is lowered (0.5 rescales
# an already tiny animal down to half its size half the time) and ``degrees``
# is raised (footage shot straight down has no privileged heading, and
# rotation preserves left/right so keypoints need no remapping).
DEFAULT_TRAIN_OVERRIDES = {
    "degrees": 180.0,
    "scale": 0.2,
    "translate": 0.1,
    "fliplr": 0.5,
    "flipud": 0.5,
    "mosaic": 1.0,
    "close_mosaic": 10,
    "hsv_v": 0.4,
    "hsv_s": 0.7,
    "hsv_h": 0.015,
}


def normalize_task(task: Optional[str]) -> str:
    """Return a valid task string, falling back to ``obb`` for legacy configs.

    Older projects stored ``"auto"`` (task inferred from the model). Those are
    mapped to ``obb``, which is what the tool did in practice back then.
    """
    if task in TASKS:
        return task
    return TASK_OBB


def default_model_for(task: str) -> str:
    return DEFAULT_MODELS.get(normalize_task(task), DEFAULT_MODELS[TASK_DETECT])


def pretrained_choices(task: str) -> List[str]:
    """Every official pretrained checkpoint usable as a base for ``task``."""
    suffix = TASK_SUFFIX[normalize_task(task)]
    return [f"{MODEL_FAMILY}{scale}{suffix}.pt" for scale in MODEL_SCALES]


def is_pretrained_name(name: str) -> bool:
    """True when ``name`` is a bare official ultralytics checkpoint name.

    A path (custom or fine-tuned weights such as ``.../weights/best.pt``) is
    deliberately rejected: fine-tuning must always restart from the official
    pretrained backbone rather than from a previous run's output.
    """
    if not name or "/" in name or "\\" in name:
        return False
    return bool(_PRETRAINED_RE.match(name))


def pretrained_task(name: str) -> Optional[str]:
    """Task a pretrained checkpoint name targets, or None if not official."""
    match = _PRETRAINED_RE.match(name or "")
    if not match:
        return None
    suffix = match.group(2) or ""
    for task, task_suffix in TASK_SUFFIX.items():
        if suffix == task_suffix:
            return task
    return None   # -seg / -cls: valid ultralytics weights, unsupported here


def validate_pretrained(name: str, task: str) -> Tuple[bool, str]:
    """Check a fine-tune base model. Returns ``(ok, reason)``.

    ``reason`` is empty when ok, otherwise a message meant for the user.
    """
    task = normalize_task(task)
    if not name:
        return False, "No base model set for fine-tuning."
    if not is_pretrained_name(name):
        return False, (
            f"'{name}' is not an official ultralytics pretrained checkpoint.\n"
            f"Fine-tuning must start from pretrained weights such as "
            f"'{default_model_for(task)}', not from custom or previously "
            f"fine-tuned weights."
        )
    found = pretrained_task(name)
    if found != task:
        return False, (
            f"'{name}' is a '{found or 'unsupported'}' checkpoint but this "
            f"project's task is '{task}'.\nUse e.g. "
            f"'{default_model_for(task)}'."
        )
    return True, ""


def project_folder_name(name: str, task: str) -> str:
    """Build the on-disk project folder name: ``<name>_<task>``.

    Already-suffixed names are left untouched so the function is idempotent.
    """
    task = normalize_task(task)
    suffix = f"_{task}"
    clean = name.strip().replace(" ", "_")
    return clean if clean.endswith(suffix) else f"{clean}{suffix}"


def task_from_folder_name(folder: str) -> Optional[str]:
    """Recover the task from a ``<name>_<task>`` folder, or None if absent."""
    for task in TASKS:
        if folder.endswith(f"_{task}"):
            return task
    return None


# ---------------------------------------------------------------------------
# Keypoint helpers
# ---------------------------------------------------------------------------

def default_keypoint_names(n: int) -> List[str]:
    return [f"kpt_{i}" for i in range(max(0, n))]


def default_flip_idx(n: int) -> List[int]:
    """Identity mapping with keypoints 0 and 1 swapped.

    For a hammerhead, keypoints 0 and 1 are the left and right tips of the
    cephalofoil, so a horizontal flip must exchange them; every other point is
    on the body axis and maps to itself. Adjust in the project settings if the
    keypoint layout differs.
    """
    idx = list(range(max(0, n)))
    if n >= 2:
        idx[0], idx[1] = idx[1], idx[0]
    return idx


def parse_flip_idx(text: str, n: int) -> List[int]:
    """Parse a comma-separated flip_idx string, or fall back to the default.

    Falls back silently when the input is malformed, out of range, or not a
    permutation, so a typo in the settings can never corrupt a training run.
    """
    try:
        idx = [int(part) for part in text.replace(" ", "").split(",") if part]
    except ValueError:
        return default_flip_idx(n)
    if len(idx) != n or sorted(idx) != list(range(n)):
        return default_flip_idx(n)
    return idx


def format_flip_idx(idx: List[int]) -> str:
    return ", ".join(str(i) for i in idx)