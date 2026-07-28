"""
cohesion.py
===========

Compute a per-frame group-cohesion metric from per-individual track JSON files.

Input: either the GUI export (``<export_dir>/individuals/``) or the
post-processed tracks (``<export_dir>/postp_tracks/``). Both schemas share the
``detections`` list, so this script does not care which one it is given —
though running it on post-processed tracks is what you usually want, since
interpolation fills the frames where an individual would otherwise be absent.

Track identity is the *filename stem*, verbatim (see ``track_io.track_id``):
``individuals/shark_3.json`` -> column ``shark_3``, ``individuals/Bob.json``
-> column ``Bob``. Those column names are what ``angle.py`` and
``merge_csv_per_track.py`` look up, so nothing needs renaming downstream.

For each frame independently:
  - T = median of all bbox diagonals present in that frame (scale).
  - For each individual i present:
        cohesion_i = quantile_q( ||c_i - c_j|| for j != i ) / T
    where c_* are centroids and `q` is the quantile chosen with --quantile
    (default 0.25, i.e. the lower quartile of pairwise distances).
  - cohesion_global = mean of cohesion_i over present individuals.

Output: one CSV with columns
    frame, <track_id>..., T, cohesion_global

Run `python scripts/cohesion.py --help` for all options.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

import track_io


# =============================================================================
# Data loading
# =============================================================================

def build_frame_index(tracks: dict) -> dict:
    """For each track, build a dict {frame: detection}."""
    return {tid: track_io.build_frame_index(data) for tid, data in tracks.items()}


# =============================================================================
# Cohesion computation
# =============================================================================

def bbox_diagonal(bbox: list) -> float:
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return float(np.sqrt(w * w + h * h))


def compute_single_frame(frame: int, frame_index: dict, track_ids: list,
                         quantile: float = 0.25) -> dict:
    """100%-independent computation for ONE frame. Returns a CSV row."""
    row = {"frame": frame}

    # 1) Detections present in this frame
    detections = {}
    for tid in track_ids:
        if frame in frame_index[tid]:
            detections[tid] = frame_index[tid][frame]

    # 2) Need at least 2 individuals to define cohesion
    if len(detections) < 2:
        for tid in track_ids:
            row[tid] = np.nan
        row["T"] = np.nan
        row["cohesion_global"] = np.nan
        return row

    # 3) T = median bbox diagonal in this frame
    diags = [bbox_diagonal(det["bbox"]) for det in detections.values()]
    T = float(np.median(diags))

    # 4) Centroids of present individuals
    centroids = {
        tid: np.array(det["centroid"])
        for tid, det in detections.items()
    }

    # 5) Per-individual cohesion = quantile_q(pairwise distances) / T
    cohesion_values = []
    for tid in track_ids:
        if tid not in centroids:
            row[tid] = np.nan
            continue

        dists = [
            float(np.linalg.norm(centroids[tid] - centroids[tj]))
            for tj in centroids
            if tj != tid
        ]

        ci = float(np.quantile(dists, quantile)) / T if (dists and T > 0) else np.nan
        row[tid] = round(ci, 4) if not np.isnan(ci) else np.nan

        if not np.isnan(ci):
            cohesion_values.append(ci)

    # 6) Global cohesion = mean over present individuals
    row["T"] = round(T, 2)
    row["cohesion_global"] = (
        round(float(np.mean(cohesion_values)), 4) if cohesion_values else np.nan
    )
    return row


def compute_cohesion_per_frame(tracks: dict, output_path: str,
                               quantile: float = 0.25) -> pd.DataFrame:
    frame_index = build_frame_index(tracks)
    track_ids = sorted(frame_index.keys())

    all_frames = sorted(
        set(f for tid_frames in frame_index.values() for f in tid_frames)
    )

    print(f"Number of tracks : {len(track_ids)}")
    print(f"Number of frames : {len(all_frames)}")
    print(f"Track IDs        : {track_ids}")
    print(f"Quantile         : {quantile}\n")

    rows = [
        compute_single_frame(frame, frame_index, track_ids, quantile=quantile)
        for frame in all_frames
    ]

    df = pd.DataFrame(rows)

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"CSV exported -> {output_path}")
    print(f"Shape        : {df.shape}")
    return df


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Compute per-frame group cohesion from post-processed track JSON "
            "files. For each frame, cohesion_i = quantile of pairwise centroid "
            "distances divided by the median bbox diagonal of that frame."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--tracks", required=True,
                   help="Directory OR glob of per-individual track .json files "
                        "(<export_dir>/individuals/ or .../postp_tracks/).")
    p.add_argument("--output-csv", required=True,
                   help="Output path for the cohesion CSV.")
    p.add_argument("--pattern", default="*.json",
                   help="Glob pattern used when --tracks is a directory.")
    p.add_argument("--quantile", type=float, default=0.25,
                   help="Quantile in [0, 1] of pairwise distances used per individual.")
    return p.parse_args()


def main():
    args = parse_args()
    if not (0.0 <= args.quantile <= 1.0):
        sys.exit(f"--quantile must be in [0, 1], got {args.quantile}")

    tracks = track_io.load_tracks_by_id(args.tracks, pattern=args.pattern)
    print(f"Loaded {len(tracks)} individuals from {args.tracks}\n")
    compute_cohesion_per_frame(tracks, args.output_csv, quantile=args.quantile)


if __name__ == "__main__":
    main()