import pandas as pd
import os

for f in os.listdir("projects/hammer/exports/2021_114/with_keypoints"):
    # === CONFIGURATION ===
    INPUT_CSV = "projects/hammer/exports/2021_114/with_keypoints/"+f
    OUTPUT_CSV = "projects/hammer/exports/2021_114/with_keypoints/"+f
    FPS = 30

    # Plages de temps (en secondes) où human_interaction = 1
    # Format : liste de tuples (début, fin)
    TIME_RANGES = [
        (55,60+15)
    ]
    # =====================

    df = pd.read_csv(INPUT_CSV)

    # Convertir les plages de temps en plages de frames
    frame_ranges = [(int(start * FPS), int(end * FPS)) for start, end in TIME_RANGES]

    # Colonne temps en secondes
    # df["time_seconds"] = df["frame"] / FPS

    # Créer la colonne, 0 par défaut
    df["human_interaction"] = 0
    

    # Marquer 1 pour chaque plage
    for f_start, f_end in frame_ranges:
        mask = (df["frame"] >= f_start) & (df["frame"] <= f_end)
        df.loc[mask, "human_interaction"] = 1

    df.to_csv(OUTPUT_CSV, index=False)

    n_total = len(df)
    n_interaction = df["human_interaction"].sum()
    print(f"Terminé ! {n_interaction}/{n_total} frames marquées comme interaction.")
    print(f"Fichier sauvegardé : {OUTPUT_CSV}")