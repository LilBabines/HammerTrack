"""
track_io.py
===========

Shared I/O layer for the feature-extraction scripts.

**Single source of truth for the identity rule.** A track is identified by the
*stem of its JSON filename*, verbatim. That same string is used everywhere:
CSV column names, video labels, cohesion lookup, output filenames. No parsing,
no prefixing, no reconstruction from ``merged_track_ids`` — so an individual
the GUI named ``Bob`` stays ``Bob`` from one end of the pipeline to the other.

Expected layout, as written by the GUI (``tracking_page._export_data``):

    <export_dir>/
    ├── per_frame/            frame_XXXXXX.txt      (not used here)
    ├── per_track/            track_XXXX.json       raw tracker fragments
    ├── individuals/          <name>.json           fragments MERGED per animal
    ├── cmc_transforms.json   {"<frame>": affine 2x3}
    └── postp_tracks/         <name>.json           written by track_postprocess.py

``individuals/`` is the entry point of the feature-extraction pipeline: the
merge (identity) is done by the GUI, the numeric passes (outliers,
interpolation, smoothing) by ``track_postprocess.py``, which re-emits the same
stems into ``postp_tracks/``.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Keys carried by the GUI export that identify the *animal* rather than
# describe a detection. Every pass preserves them verbatim.
IDENTITY_KEYS = ("uid", "name", "notes", "color", "merged_track_ids")

# Columns of a cohesion CSV that are NOT per-track values. Anything else in
# the header is a track ID. ``cohesion_globale`` is the legacy French spelling,
# still recognised so previously generated CSVs keep working.
COHESION_RESERVED_COLUMNS = ("frame", "T", "cohesion_global", "cohesion_globale")


# =============================================================================
# Identity
# =============================================================================

def track_id(path) -> str:
    """Identity of a track file: its filename stem, verbatim.

    ``individuals/shark_3.json`` -> ``"shark_3"``
    ``individuals/Bob.json``     -> ``"Bob"``
    """
    return Path(path).stem


def check_column_safe(tid: str, path: str) -> None:
    """Warn if a track ID would collide with a reserved CSV column."""
    if tid in COHESION_RESERVED_COLUMNS:
        print(
            f"  WARNING: track '{tid}' ({path}) collides with a reserved CSV "
            f"column name {COHESION_RESERVED_COLUMNS}. Rename the individual "
            f"in the GUI, or the cohesion CSV will be ambiguous."
        )


# =============================================================================
# Loading
# =============================================================================

def resolve_track_files(tracks_arg: str, pattern: str = "*.json") -> List[str]:
    """Accept either a directory (uses ``pattern``) or a glob; return files.

    Exits with a message rather than silently processing nothing, which is the
    most common way to lose an afternoon on this pipeline.
    """
    if os.path.isdir(tracks_arg):
        track_pattern = str(Path(tracks_arg) / pattern)
    else:
        track_pattern = tracks_arg

    files = sorted(glob.glob(track_pattern))
    if not files:
        sys.exit(f"No track files found matching: {track_pattern}")
    return files


def load_track(path: str) -> Dict[str, Any]:
    """Load one track JSON and tag it with its identity.

    Accepts both schemas indifferently, since they share ``detections``:
      * ``individuals/<name>.json``  — GUI merge, no numeric pass
      * ``postp_tracks/<name>.json`` — after ``track_postprocess.py``
      * ``per_track/track_XXXX.json`` — a single raw fragment
    """
    with open(path) as f:
        data = json.load(f)

    if not isinstance(data, dict) or "detections" not in data:
        sys.exit(f"Not a track file (no 'detections' key): {path}")

    data["id"] = track_id(path)
    data["source_path"] = str(path)
    check_column_safe(data["id"], path)
    return data


def load_tracks(tracks_arg: str, pattern: str = "*.json") -> List[dict]:
    """Load every track file, sorted by path. IDs come from the stems."""
    return [load_track(p) for p in resolve_track_files(tracks_arg, pattern)]


def load_tracks_by_id(tracks_arg: str, pattern: str = "*.json") -> Dict[str, dict]:
    """Same as :func:`load_tracks`, keyed by track ID.

    Two files with the same stem in different directories would silently
    shadow each other, so that is an error.
    """
    tracks: Dict[str, dict] = {}
    for track in load_tracks(tracks_arg, pattern):
        tid = track["id"]
        if tid in tracks:
            sys.exit(
                f"Duplicate track ID '{tid}': "
                f"{tracks[tid]['source_path']} and {track['source_path']}"
            )
        tracks[tid] = track
    return tracks


def identity_payload(data: dict) -> dict:
    """The identity keys present in a loaded track, for pass-through on export."""
    return {k: data[k] for k in IDENTITY_KEYS if k in data}


def build_frame_index(track: dict) -> Dict[int, dict]:
    """``{frame: detection}`` for one loaded track."""
    return {int(d["frame"]): d for d in track["detections"]}


# =============================================================================
# Cohesion CSV
# =============================================================================

def cohesion_track_columns(columns) -> List[str]:
    """Per-track columns of a cohesion CSV: everything not reserved.

    Replaces the old ``startswith("shark_")`` heuristic, which broke as soon
    as an individual was named anything else.
    """
    return [c for c in columns if c not in COHESION_RESERVED_COLUMNS]


def warn_missing_columns(track_ids, columns, label: str) -> None:
    """Report track IDs absent from a CSV, and CSV columns with no track."""
    present = set(cohesion_track_columns(columns))
    wanted = set(track_ids)
    for tid in sorted(wanted - present):
        print(f"  WARNING: '{tid}' has no column in the {label} CSV")
    for col in sorted(present - wanted):
        print(f"  WARNING: {label} CSV column '{col}' matches no track file")
