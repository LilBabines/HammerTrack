"""
Merge per-track JSON + angle CSVs + cohesion CSV → one CSV per track.

Output columns:
  frame, time_s,
  centroid_x, centroid_y, interpolated,
  obb_x0, obb_y0, obb_x1, obb_y1, obb_x2, obb_y2, obb_x3, obb_y3,
  angle_image, angle_absolute, cohesion
"""

import json
import glob
import csv
import os
import pandas as pd
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────
FPS = 30.0

TRACK_JSON_DIR = "projects/hammer/exports/2021_114/postp_tracks"
ANGLE_IMAGE_CSV = "projects/hammer/exports/2021_114/okok_angle_image.csv"
ANGLE_ABSOLUTE_CSV = "projects/hammer/exports/2021_114/okok_angle_absolute.csv"
COHESION_CSV = "projects/hammer/exports/2021_114/cohesion_results.csv"
OUTPUT_DIR = "projects/hammer/exports/2021_114/per_track_csv"

# ── Load angle & cohesion CSVs (indexed by frame) ───────────────────
df_angle_img = pd.read_csv(ANGLE_IMAGE_CSV).set_index("frame")
df_angle_abs = pd.read_csv(ANGLE_ABSOLUTE_CSV).set_index("frame")
df_cohesion = pd.read_csv(COHESION_CSV).set_index("frame")

# ── Process each track JSON ─────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
track_files = sorted(glob.glob(os.path.join(TRACK_JSON_DIR, "*.json")))

print(f"Found {len(track_files)} track files")

for path in track_files:
    track_id = Path(path).stem  # e.g. "shark_3"

    with open(path) as f:
        data = json.load(f)

    # Check column exists in each CSV
    has_angle_img = track_id in df_angle_img.columns
    has_angle_abs = track_id in df_angle_abs.columns
    has_cohesion = track_id in df_cohesion.columns

    if not has_angle_img:
        print(f"  WARNING: {track_id} not found in angle_image CSV")
    if not has_angle_abs:
        print(f"  WARNING: {track_id} not found in angle_absolute CSV")
    if not has_cohesion:
        print(f"  WARNING: {track_id} not found in cohesion CSV")

    out_path = os.path.join(OUTPUT_DIR, f"{track_id}.csv")
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "frame", "time_s",
            "centroid_x", "centroid_y", "interpolated",
            "obb_x0", "obb_y0", "obb_x1", "obb_y1",
            "obb_x2", "obb_y2", "obb_x3", "obb_y3",
            "angle_image", "angle_absolute", "cohesion",
        ])

        for det in sorted(data["detections"], key=lambda d: d["frame"]):
            f = det["frame"]
            time_s = f"{f / FPS:.4f}"

            cx, cy = det["centroid"]
            interp = det.get("interpolated", det["confidence"] == 0.0)

            obb = det.get("obb")
            if obb and len(obb) == 4:
                obb_flat = [obb[i][j] for i in range(4) for j in range(2)]
            else:
                obb_flat = [""] * 8

            a_img = ""
            if has_angle_img and f in df_angle_img.index:
                v = df_angle_img.at[f, track_id]
                if pd.notna(v):
                    a_img = f"{v:.6f}"

            a_abs = ""
            if has_angle_abs and f in df_angle_abs.index:
                v = df_angle_abs.at[f, track_id]
                if pd.notna(v):
                    a_abs = f"{v:.6f}"

            coh = ""
            if has_cohesion and f in df_cohesion.index:
                v = df_cohesion.at[f, track_id]
                if pd.notna(v):
                    coh = f"{v:.6f}"

            writer.writerow([
                f, time_s,
                f"{cx:.2f}", f"{cy:.2f}", int(interp),
                *[f"{v:.2f}" if v != "" else "" for v in obb_flat],
                a_img, a_abs, coh,
            ])

    n_dets = len(data["detections"])
    print(f"  {track_id}: {n_dets} rows → {out_path}")

print(f"\nDone! {len(track_files)} CSVs written to {OUTPUT_DIR}")
