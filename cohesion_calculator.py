import json
import glob
import numpy as np
import pandas as pd
from pathlib import Path


def load_tracks(json_dir: str, pattern: str = "*.json") -> dict:
    """Charge tous les fichiers JSON de tracks et retourne un dict {track_id: data}."""
    tracks = {}
    for fpath in sorted(glob.glob(str(Path(json_dir) / pattern))):
        with open(fpath, "r") as f:
            data = json.load(f)

        track_id = fpath[:-5].split("_")[-1]
        tracks[track_id] = data
    return tracks


def build_frame_index(tracks: dict) -> dict:
    """Pour chaque track, construit un dict {frame: detection}."""
    frame_index = {}
    for tid, data in tracks.items():
        frame_index[tid] = {}
        for det in data["detections"]:
            frame_index[tid][det["frame"]] = det
    return frame_index


def bbox_diagonal(bbox: list) -> float:
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return np.sqrt(w**2 + h**2)


def compute_single_frame(frame: int, frame_index: dict, track_ids: list) -> dict:
    """
    Calcul 100% indépendant pour UNE frame.
    Retourne une ligne du CSV.
    """
    row = {"frame": frame}

    # 1) Extraire les détections présentes dans cette frame
    detections = {}
    for tid in track_ids:
        if frame in frame_index[tid]:
            detections[tid] = frame_index[tid][frame]

    # 2) T = médiane des diagonales des bboxes de CETTE frame
    if len(detections) < 2:
        # Pas assez de requins pour calculer une cohésion
        for tid in track_ids:
            row[f"shark_{tid}"] = np.nan
        row["T"] = np.nan
        row["cohesion_globale"] = np.nan
        return row

    diags = [bbox_diagonal(det["bbox"]) for det in detections.values()]
    T = float(np.median(diags))

    # 3) Centroids des requins présents dans CETTE frame
    centroids = {
        tid: np.array(det["centroid"])
        for tid, det in detections.items()
    }

    # 4) Pour chaque requin i présent : cohesion_i = median(dist(i,j) pour j présent) / T
    cohesion_values = []
    for tid in track_ids:
        if tid not in centroids:
            row[f"shark_{tid}"] = np.nan
            continue

        dists = [
            np.linalg.norm(centroids[tid] - centroids[tj])
            for tj in centroids
            if tj != tid
        ]

        ci = float(np.quantile(dists,0.25)) / T if (dists and T > 0) else np.nan
        row[f"shark_{tid}"] = round(ci, 4) if not np.isnan(ci) else np.nan

        if not np.isnan(ci):
            cohesion_values.append(ci)

    # 5) Cohésion globale de CETTE frame = moyenne des cohesion_i
    row["T"] = round(T, 2)
    row["cohesion_globale"] = round(float(np.mean(cohesion_values)), 4) if cohesion_values else np.nan

    return row


def compute_cohesion_per_frame(tracks: dict, output_path: str = "cohesion_per_frame.csv"):
    frame_index = build_frame_index(tracks)
    track_ids = sorted(frame_index.keys())

    # Toutes les frames
    all_frames = sorted(set(f for tid_frames in frame_index.values() for f in tid_frames))

    print(f"Nombre de tracks : {len(track_ids)}")
    print(f"Nombre de frames : {len(all_frames)}")
    print(f"Track IDs : {track_ids}\n")

    # Chaque frame est traitée de manière totalement indépendante
    rows = [compute_single_frame(frame, frame_index, track_ids) for frame in all_frames]

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    print(f"CSV exporté → {output_path}")
    print(f"Shape : {df.shape}")
    # print(f"\nAperçu (5 premières lignes) :\n{df.head().to_string(index=False)}")
    return df


# ──────────────────────────────────────────────
#  UTILISATION
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # Adapter le chemin vers le dossier contenant les JSONs
    JSON_DIR = "projects/hammer/exports/2021_114/postp_tracks"
    OUTPUT_CSV = "projects/hammer/exports/2021_114/cohesion_results.csv"
    tracks = load_tracks(JSON_DIR)
    print(f"Chargé {len(tracks)} tracks depuis {JSON_DIR}\n")

    df = compute_cohesion_per_frame(tracks, OUTPUT_CSV)

