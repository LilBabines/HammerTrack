"""
train_yolo.py
=============

Train a YOLO model from the command line, outside the GUI.

Like the in-app fine-tuning, training always starts from an official
ultralytics pretrained checkpoint rather than from custom or previously
fine-tuned weights; ``--allow-custom-model`` opts out when resuming a run
deliberately.

Run `python scripts/train_yolo.py --help` for all options.
"""

import argparse
import os
import sys

import yaml
from ultralytics import YOLO

# Make `src` importable when the script is run from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tasks import (  # noqa: E402
    TASKS, TASK_POSE, default_model_for, normalize_task, validate_pretrained,
)


def check_data_yaml(path: str, task: str) -> dict:
    """Load data.yaml and fail early on the mistakes that waste a whole run."""
    if not os.path.isfile(path):
        sys.exit(f"Dataset config not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if "names" not in data:
        sys.exit(f"{path}: missing 'names'.")

    if task == TASK_POSE:
        kpt_shape = data.get("kpt_shape")
        if not kpt_shape or len(kpt_shape) != 2:
            sys.exit(f"{path}: pose training requires 'kpt_shape: [N, dims]'.")
        flip_idx = data.get("flip_idx")
        if not flip_idx:
            print(
                f"WARNING: {path} has no 'flip_idx' — ultralytics will "
                f"disable flip augmentation for this pose run."
            )
        elif len(flip_idx) != kpt_shape[0]:
            sys.exit(
                f"{path}: flip_idx has {len(flip_idx)} entries but kpt_shape "
                f"declares {kpt_shape[0]} keypoints."
            )
        elif sorted(flip_idx) != list(range(kpt_shape[0])):
            sys.exit(f"{path}: flip_idx must be a permutation of 0..N-1.")

    return data


def parse_args():
    p = argparse.ArgumentParser(
        description="Train a YOLO26 model on a HammerTrack dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data", required=True,
                   help="Path to the dataset config (data.yaml).")
    p.add_argument("--task", required=True, choices=list(TASKS),
                   help="YOLO task the dataset was exported for.")
    p.add_argument("--model", default=None,
                   help="Pretrained checkpoint to start from. "
                        "Defaults to the task's yolo26m variant.")
    p.add_argument("--allow-custom-model", action="store_true",
                   help="Permit non-pretrained weights (e.g. resuming from a "
                        "previous best.pt). Off by default on purpose.")

    p.add_argument("--epochs",  type=int, default=200, help="Training epochs.")
    p.add_argument("--imgsz",   type=int, default=1024, help="Input image size.")
    p.add_argument("--batch",   type=int, default=8, help="Batch size, -1 for auto.")
    p.add_argument("--workers", type=int, default=8, help="Dataloader workers.")
    p.add_argument("--device",  default="0", help="Device: '0', '0,1' or 'cpu'.")
    p.add_argument("--seed",    type=int, default=1337, help="Random seed.")

    p.add_argument("--project", default="finetune_runs",
                   help="Results are saved under project/name.")
    p.add_argument("--name",    default="train_main", help="Run name.")

    p.add_argument("--flipud", type=float, default=0.5,
                   help="Vertical flip probability.")
    p.add_argument("--fliplr", type=float, default=0.5,
                   help="Horizontal flip probability.")
    p.add_argument("--cutmix", type=float, default=0.0,
                   help="CutMix probability.")
    return p.parse_args()


def main():
    args = parse_args()
    task = normalize_task(args.task)

    model_name = args.model or default_model_for(task)
    ok, reason = validate_pretrained(model_name, task)
    if not ok:
        if not args.allow_custom_model:
            sys.exit(f"{reason}\n\nPass --allow-custom-model to override.")
        print(f"WARNING: {reason}\nProceeding because --allow-custom-model was given.")

    check_data_yaml(args.data, task)

    print(f"Loading model : {model_name} (task={task})")
    model = YOLO(model_name)

    loaded_task = getattr(model, "task", None)
    if loaded_task and loaded_task != task:
        sys.exit(
            f"Model reports task '{loaded_task}' but --task is '{task}'."
        )

    print(f"Starting training on : {args.data}")
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        seed=args.seed,
        project=args.project,
        name=args.name,
        exist_ok=True,
        flipud=args.flipud,
        fliplr=args.fliplr,
        cutmix=args.cutmix,
    )


if __name__ == "__main__":
    main()