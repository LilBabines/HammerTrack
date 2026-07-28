"""
merge_csv_per_track.py
======================

Merge per-track JSON + angle CSVs + cohesion CSV into one CSV per track.

Output columns:
  frame, time_s,
  centroid_x, centroid_y, interpolated,
  obb_x0, obb_y0, obb_x1, obb_y1, obb_x2, obb_y2, obb_x3, obb_y3,
  angle_image, angle_absolute, cohesion

Track identity is the JSON filename stem (e.g. "shark_3"); the same string
must be used as the column name in the angle and cohesion CSVs.

Run `python scripts/merge_csv_per_track.py --help` for all options.
"""

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

import pandas as pd


CSV_HEADER = [
    "frame", "time_s",
    "centroid_x", "centroid_y", "interpolated",
    "obb_x0", "obb_y0", "obb_x1", "obb_y1",
    "obb_x2", "obb_y2", "obb_x3", "obb_y3",
    "angle_image", "angle_absolute", "cohesion",
]


def load_indexed_csv(path: str, label: str):
    """Load a CSV indexed by the ``frame`` column, or None if not provided."""
    if not path:
        return None
    if not os.path.isfile(path):
        sys.exit(f"{label} CSV not found: {path}")
    df = pd.read_csv(path)
    if "frame" not in df.columns:
        sys.exit(f"{label} CSV has no 'frame' column: {path}")
    return df.set_index("frame")


def lookup(df, frame: int, track_id: str, fmt: str = "{:.6f}") -> str:
    """Return the value at (frame, track_id) as a formatted string, or ''."""
    if df is None or track_id not in df.columns or frame not in df.index:
        return ""
    value = df.at[frame, track_id]
    return fmt.format(value) if pd.notna(value) else ""


def merge_track(path: str, fps: float, df_angle_img, df_angle_abs,
                df_cohesion, out_dir: Path) -> int:
    """Write one merged CSV for a single track JSON. Returns the row count."""
    track_id = Path(path).stem

    with open(path) as f:
        data = json.load(f)

    for df, label in ((df_angle_img, "angle_image"),
                      (df_angle_abs, "angle_absolute"),
                      (df_cohesion, "cohesion")):
        if df is not None and track_id not in df.columns:
            print(f"  WARNING: {track_id} not found in {label} CSV")

    out_path = out_dir / f"{track_id}.csv"
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_HEADER)

        for det in sorted(data["detections"], key=lambda d: d["frame"]):
            frame = det["frame"]
            cx, cy = det["centroid"]
            interp = det.get("interpolated", det["confidence"] == 0.0)

            obb = det.get("obb")
            if obb and len(obb) == 4:
                obb_cells = [f"{obb[i][j]:.2f}" for i in range(4) for j in range(2)]
            else:
                obb_cells = [""] * 8

            writer.writerow([
                frame, f"{frame / fps:.4f}",
                f"{cx:.2f}", f"{cy:.2f}", int(interp),
                *obb_cells,
                lookup(df_angle_img, frame, track_id),
                lookup(df_angle_abs, frame, track_id),
                lookup(df_cohesion, frame, track_id),
            ])

    n_dets = len(data["detections"])
    print(f"  {track_id}: {n_dets} rows -> {out_path}")
    return n_dets


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Merge post-processed track JSONs with angle and cohesion CSVs "
            "into one consolidated CSV per track."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--tracks", required=True,
                   help="Glob pattern OR directory of post-processed track .json files.")
    p.add_argument("--output-dir", required=True,
                   help="Output directory for the per-track merged CSVs.")
    p.add_argument("--angle-image-csv", default=None,
                   help="CSV of image-space angles (from scripts/angle.py).")
    p.add_argument("--angle-absolute-csv", default=None,
                   help="CSV of stabilized angles (from scripts/angle.py).")
    p.add_argument("--cohesion-csv", default=None,
                   help="Per-frame cohesion CSV (from scripts/cohesion.py).")
    p.add_argument("--fps", type=float, default=30.0,
                   help="Frame rate used to compute the time_s column.")
    p.add_argument("--pattern", default="*.json",
                   help="Glob pattern used when --tracks is a directory.")
    return p.parse_args()


def main():
    args = parse_args()

    if args.fps <= 0:
        sys.exit(f"--fps must be > 0, got {args.fps}")

    track_pattern = (str(Path(args.tracks) / args.pattern)
                     if os.path.isdir(args.tracks) else args.tracks)
    track_files = sorted(glob.glob(track_pattern))
    if not track_files:
        sys.exit(f"No track files found matching: {track_pattern}")

    df_angle_img = load_indexed_csv(args.angle_image_csv, "angle_image")
    df_angle_abs = load_indexed_csv(args.angle_absolute_csv, "angle_absolute")
    df_cohesion = load_indexed_csv(args.cohesion_csv, "cohesion")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(track_files)} track files")
    for path in track_files:
        merge_track(path, args.fps, df_angle_img, df_angle_abs,
                    df_cohesion, out_dir)

    print(f"\nDone! {len(track_files)} CSVs written to {out_dir}")


if __name__ == "__main__":
    main()