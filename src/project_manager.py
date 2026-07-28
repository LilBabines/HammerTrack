"""
Project management — handles project folders and config persistence.

A *project* is a directory under ``PROJECTS_ROOT`` containing a dataset,
fine-tune runs, exports and a ``config.json`` file. ``ProjectManager``
hides all of this from the rest of the app: list / create projects,
load / save their configs.

Every project is bound to a single YOLO task, chosen at creation time and
immutable afterwards. The task is part of the folder name (``<name>_<task>``)
so two tasks can coexist side by side without their datasets ever mixing.

This module is pure I/O (no Qt, no Ultralytics) so it can be reused or
unit-tested standalone.
"""

import json
import os
from typing import List, Optional

from .tasks import (
    TASK_OBB,
    TASKS,
    POSE_BBOX_AUTO,
    default_flip_idx,
    default_keypoint_names,
    default_model_for,
    normalize_task,
    project_folder_name,
    task_from_folder_name,
)


PROJECTS_ROOT = os.path.join(os.getcwd(), "projects")

# Sub-folders created inside every new project.
_PROJECT_SUBFOLDERS = (
    "datasets/images/train",
    "datasets/images/val",
    "datasets/labels/train",
    "datasets/labels/val",
    "finetune_runs",
    "exports",
)

# Default number of keypoints for a new pose project.
DEFAULT_NUM_KEYPOINTS = 5


class ProjectManager:
    """Manages project directories and their config files."""

    def __init__(self, root: str = PROJECTS_ROOT):
        self.root = root
        os.makedirs(self.root, exist_ok=True)

    # ---------------- Listing & creation ----------------

    def list_projects(self) -> List[str]:
        if not os.path.isdir(self.root):
            return []
        return sorted(
            d for d in os.listdir(self.root)
            if os.path.isdir(os.path.join(self.root, d))
        )

    def create_project(self, name: str, task: str = TASK_OBB) -> str:
        """Create ``<name>_<task>`` (if missing) and return its full path.

        Safe to call on an existing project: folders use ``exist_ok`` and the
        config is only written when absent, so a stored task is never
        overwritten.
        """
        folder = project_folder_name(name, task)
        proj_dir = self.project_dir(folder)
        os.makedirs(proj_dir, exist_ok=True)
        for sub in _PROJECT_SUBFOLDERS:
            os.makedirs(os.path.join(proj_dir, sub), exist_ok=True)

        cfg_path = os.path.join(proj_dir, "config.json")
        if not os.path.exists(cfg_path):
            self.save_config(folder, self._default_config(folder, task))
        return proj_dir

    def ensure_project(self, folder: str) -> str:
        """Ensure the sub-folders of an already-existing project exist.

        Used when selecting a project: it must not invent a task, so the task
        is resolved from the config / folder name instead of being defaulted.
        """
        return self.create_project(folder, self.project_task(folder))

    def project_dir(self, folder: str) -> str:
        return os.path.join(self.root, folder)

    # ---------------- Task ----------------

    def project_task(self, folder: str) -> str:
        """Return the task a project is bound to.

        Resolution order: the stored ``task_type``, then the folder suffix,
        then ``obb`` for legacy projects created before task selection existed.
        """
        cfg_path = os.path.join(self.root, folder, "config.json")
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    stored = json.load(f).get("task_type")
                if stored in TASKS:
                    return stored
            except (OSError, json.JSONDecodeError):
                pass
        return task_from_folder_name(folder) or TASK_OBB

    # ---------------- Config I/O ----------------

    def load_config(self, folder: str) -> dict:
        cfg_path = os.path.join(self.root, folder, "config.json")
        task = self.project_task(folder)
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            # Backfill keys introduced after this project was created, then pin
            # the task to its resolved value (legacy configs stored "auto").
            for key, value in self._default_config(folder, task).items():
                cfg.setdefault(key, value)
            cfg["task_type"] = task
            return cfg
        return self._default_config(folder, task)

    def save_config(self, folder: str, cfg: dict):
        # The task is immutable: never let a caller write a different one.
        cfg = dict(cfg)
        cfg["task_type"] = self.project_task(folder)
        cfg_path = os.path.join(self.root, folder, "config.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)

    # ---------------- Defaults ----------------

    def _default_config(self, folder: str, task: Optional[str] = None) -> dict:
        proj = self.project_dir(folder)
        task = normalize_task(task)
        n_kpt = DEFAULT_NUM_KEYPOINTS
        return {
            "project_name":       folder,
            "dataset_dir":        os.path.join(proj, "datasets"),
            "finetune_dir":       os.path.join(proj, "finetune_runs"),
            "model_path":         "",
            "default_model":      default_model_for(task),
            "class_names":        ["object"],
            "task_type":          task,          # "detect", "obb" or "pose"
            # Pose-only settings (ignored by the other tasks)
            "num_keypoints":      n_kpt,
            "keypoint_names":     default_keypoint_names(n_kpt),
            "flip_idx":           default_flip_idx(n_kpt),
            "pose_bbox_mode":     POSE_BBOX_AUTO,
            # Training
            "epochs":             20,
            "imgsz":              1024,
            "batch":              16,
            "val_split":          0.1,
            "conf_threshold":     0.5,
            # Tracking
            "tracker_type":       "botsort",
            "reid_weights":       "osnet_x0_25_msmt17.pt",
            "with_reid":          True,
            "track_high_thresh":  0.6,
            "track_low_thresh":   0.1,
            "new_track_thresh":   0.7,
            "track_buffer":       30,
            "match_thresh":       0.8,
            "proximity_thresh":   0.5,
            "appearance_thresh":  0.25,
        }