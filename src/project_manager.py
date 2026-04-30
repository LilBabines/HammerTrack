"""
Project management — handles project folders and config persistence.

A *project* is a directory under ``PROJECTS_ROOT`` containing a dataset,
fine-tune runs, exports and a ``config.json`` file. ``ProjectManager``
hides all of this from the rest of the app: list / create projects,
load / save their configs.

This module is pure I/O (no Qt, no Ultralytics) so it can be reused or
unit-tested standalone.
"""

import json
import os
from typing import List

from .workers import YOLO_MODEL_PATH


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

    def create_project(self, name: str) -> str:
        proj_dir = self.project_dir(name)
        os.makedirs(proj_dir, exist_ok=True)
        for sub in _PROJECT_SUBFOLDERS:
            os.makedirs(os.path.join(proj_dir, sub), exist_ok=True)
        cfg_path = os.path.join(proj_dir, "config.json")
        if not os.path.exists(cfg_path):
            self.save_config(name, self._default_config(name))
        return proj_dir

    def project_dir(self, name: str) -> str:
        return os.path.join(self.root, name)

    # ---------------- Config I/O ----------------

    def load_config(self, name: str) -> dict:
        cfg_path = os.path.join(self.root, name, "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return self._default_config(name)

    def save_config(self, name: str, cfg: dict):
        cfg_path = os.path.join(self.root, name, "config.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)

    # ---------------- Defaults ----------------

    def _default_config(self, name: str) -> dict:
        proj = self.project_dir(name)
        return {
            "project_name":       name,
            "dataset_dir":        os.path.join(proj, "datasets"),
            "finetune_dir":       os.path.join(proj, "finetune_runs"),
            "model_path":         YOLO_MODEL_PATH,
            "class_names":        ["object"],
            "task_type":          "auto",          # "auto", "obb", "detect"
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
