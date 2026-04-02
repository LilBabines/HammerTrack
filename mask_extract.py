import os
import subprocess
import json
import csv
import shutil
import numpy as np
import torch
import cv2
from sam2.build_sam import build_sam2_video_predictor

# --- Config ---
VIDEO_PATH = "clips/selected/SIMP2021_057-13_10-16_25.mp4"
TRACK_PATH = "projects/hammer/exports/2021_057/postp_tracks/merged_2_34_40.json"
OUTPUT_PATH = "projects/hammer/exports/2021_057/output_masked.mp4"
FRAMES_DIR = "clips/by_frames/SIMP2021_057"

SAM2_CHECKPOINT = "sam2.1_hiera_base_plus.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"

CHUNK_SIZE = 200
MASK_CONFIDENCE_THRESHOLD = 0.85
MASK_COLOR = (0, 200, 255)
MASK_ALPHA = 0.45
CONTOUR_COLOR = (0, 255, 0)
CONTOUR_THICKNESS = 4
CSV_OUTPUT = "projects/hammer/exports/2021_057/shark_keypoints.csv"

CHUNK_FRAMES_DIR = "/tmp/sam2_chunk"
GRAPH_SECONDS = 5
GRAPH_W = 600  # largeur du bandeau droit
GRAPH_H = 400  # hauteur du graphique
GRAPH_MARGIN = 40


def compute_angles(kp):
    """Calcule les 3 angles des vecteurs vers la queue."""
    tail = np.array([kp["tail_x"], kp["tail_y"]])
    com = np.array([kp["com_x"], kp["com_y"]])
    a1 = np.array([kp["art1_x"], kp["art1_y"]])
    a2 = np.array([kp["art2_x"], kp["art2_y"]])

    def vec_angle(origin, target):
        v = target - origin
        return np.arctan2(v[1], v[0])

    return {
        "com_tail": vec_angle(com, tail),
        "art1_tail": vec_angle(a1, tail),
        "art2_tail": vec_angle(a2, tail),
    }


def draw_graph(angle_buffer, fps):
    """Dessine le graphique des angles sur une image."""
    graph = np.zeros((GRAPH_H, GRAPH_W, 3), dtype=np.uint8)
    graph[:] = (30, 30, 30)  # fond gris foncé

    n = len(angle_buffer)
    if n < 2:
        return graph

    # Axes
    gx0 = GRAPH_MARGIN
    gx1 = GRAPH_W - 10
    gy0 = 10
    gy1 = GRAPH_H - GRAPH_MARGIN
    gw = gx1 - gx0
    gh = gy1 - gy0

    # Range Y : min/max de toutes les courbes
    all_vals = []
    for entry in angle_buffer:
        all_vals.extend([entry["com_tail"], entry["art1_tail"], entry["art2_tail"]])
    y_min = min(all_vals) - 0.1
    y_max = max(all_vals) + 0.1
    if y_max - y_min < 0.2:
        y_min -= 0.1
        y_max += 0.1

    # Ligne zéro
    zero_y = int(gy0 + gh * (1 - (0 - y_min) / (y_max - y_min)))
    if gy0 < zero_y < gy1:
        cv2.line(graph, (gx0, zero_y), (gx1, zero_y), (80, 80, 80), 1)

    # Axe Y labels
    for val in np.linspace(y_min, y_max, 5):
        py = int(gy0 + gh * (1 - (val - y_min) / (y_max - y_min)))
        cv2.putText(graph, f"{val:.1f}", (2, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

    # Axe X (temps)
    max_time = n / fps
    for t in range(0, int(max_time) + 1):
        px = int(gx0 + gw * t / max(max_time, 0.1))
        if px <= gx1:
            cv2.putText(graph, f"{t}s", (px - 5, GRAPH_H - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

    # Courbes
    colors = {
        "com_tail": (0, 255, 255),   # jaune = COM->tail
        "art1_tail": (0, 200, 0),    # vert = art1->tail
        "art2_tail": (0, 100, 255),  # orange = art2->tail
    }

    for key, color in colors.items():
        points = []
        for i, entry in enumerate(angle_buffer):
            px = int(gx0 + gw * i / (n - 1))
            val = entry[key]
            py = int(gy0 + gh * (1 - (val - y_min) / (y_max - y_min)))
            points.append((px, py))
        for i in range(len(points) - 1):
            cv2.line(graph, points[i], points[i + 1], color, 2)

    # Légende
    lx, ly = gx0 + 5, gy0 + 15
    for key, color in colors.items():
        cv2.line(graph, (lx, ly), (lx + 20, ly), color, 2)
        cv2.putText(graph, key, (lx + 25, ly + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        ly += 18

    return graph


def extract_keypoints(binary_mask):
    """Extrait head (milieu du cephalofoil), COM, tail depuis un masque binaire."""
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 10:
        return None
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None

    com = np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]])
    pts = contour.reshape(-1, 2).astype(np.float64)

    # Tail = point le plus loin du COM (stable)
    dists_to_com = np.linalg.norm(pts - com, axis=1)
    tail = pts[dists_to_com.argmax()]

    # Axe du corps : COM -> tail
    body_vec = tail - com
    body_len = np.linalg.norm(body_vec)
    if body_len == 0:
        return None
    body_axis = body_vec / body_len
    perp_axis = np.array([-body_axis[1], body_axis[0]])

    # Projections de tous les points du contour sur l'axe du corps
    # Positif = côté queue, négatif = côté tête
    proj_along = (pts - com) @ body_axis

    # Scanner des coupes perpendiculaires côté tête (proj < 0)
    min_proj = proj_along.min()
    n_steps = 80
    step_size = abs(min_proj) / max(n_steps, 1)

    best_score = 0
    best_p1 = None
    best_p2 = None

    for t in np.linspace(min_proj, 0, n_steps):
        # Points du contour proches de cette coupe
        band = np.abs(proj_along - t) < max(step_size, 2.0)
        if band.sum() < 2:
            continue

        nearby = pts[band]
        perp_proj = (nearby - com) @ perp_axis

        width = perp_proj.max() - perp_proj.min()
        dist_from_com = abs(t) / max(abs(min_proj), 1e-6)  # 0 au COM, 1 au bout

        # Score = largeur × distance au COM (favorise le cephalofoil loin du corps)
        score = width * dist_from_com

        if score > best_score:
            best_score = score
            best_p1 = nearby[perp_proj.argmin()]
            best_p2 = nearby[perp_proj.argmax()]

    if best_p1 is None or best_p2 is None:
        # Fallback : point le plus loin de la queue
        dists_to_tail = np.linalg.norm(pts - tail, axis=1)
        head = pts[dists_to_tail.argmax()]
        head_p1, head_p2 = None, None
    else:
        head = (best_p1 + best_p2) / 2
        head_p1, head_p2 = best_p1, best_p2

    # Points d'articulation : 1/3, 2/3 entre COM et tail, centrés en largeur
    articulations = []
    for frac in [1/3, 2/3]:
        target_t = body_len * frac
        band_art = np.abs(proj_along - target_t) < max(step_size * 2, 3.0)
        if band_art.sum() >= 2:
            nearby_art = pts[band_art]
            perp_art = (nearby_art - com) @ perp_axis
            art_p1 = nearby_art[perp_art.argmin()]
            art_p2 = nearby_art[perp_art.argmax()]
            articulations.append((art_p1 + art_p2) / 2)
        else:
            articulations.append(com + body_vec * frac)

    # Angle head->COM->tail
    v1 = com - head
    v2 = tail - com
    angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
    angle = (angle + np.pi) % (2 * np.pi) - np.pi

    result = {
        "head_x": head[0], "head_y": head[1],
        "com_x": com[0], "com_y": com[1],
        "art1_x": articulations[0][0], "art1_y": articulations[0][1],
        "art2_x": articulations[1][0], "art2_y": articulations[1][1],
        "tail_x": tail[0], "tail_y": tail[1],
        "angle_rad": angle,
    }
    # Endpoints du segment tête pour visualisation
    if head_p1 is not None:
        result["head_p1"] = (int(head_p1[0]), int(head_p1[1]))
        result["head_p2"] = (int(head_p2[0]), int(head_p2[1]))

    return result


# --- Video info ---
cap = cv2.VideoCapture(VIDEO_PATH)
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

sample_frame = cv2.imread(os.path.join(FRAMES_DIR, f"{0:06d}.jpg"))
h, w = sample_frame.shape[:2]
print(f"Frame size: {w}x{h}, {total_frames} frames, {fps} fps")

cap_orig = cv2.VideoCapture(VIDEO_PATH)
orig_w = int(cap_orig.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap_orig.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap_orig.release()

scale_x = w / orig_w
scale_y = h / orig_h
print(f"Original: {orig_w}x{orig_h} -> Scale: {scale_x:.3f}, {scale_y:.3f}")

# --- Load tracking data ---
with open(TRACK_PATH) as f:
    track = json.load(f)
detections = track["detections"]
det_by_frame = {d["frame"]: d for d in detections}

# --- Video writer + CSV ---
canvas_w = w + GRAPH_W
writer = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (canvas_w, h))

csv_file = open(CSV_OUTPUT, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["frame", "head_x", "head_y", "com_x", "com_y", "art1_x", "art1_y", "art2_x", "art2_y", "tail_x", "tail_y", "angle_rad"])

# --- Live display via ffplay ---
disp_w, disp_h = int(canvas_w * 0.35), int(h * 0.35)
ffplay_proc = subprocess.Popen(
    ["ffplay", "-f", "rawvideo", "-pixel_format", "bgr24",
     "-video_size", f"{disp_w}x{disp_h}", "-framerate", str(fps),
     "-window_title", "SAM2 Shark", "-"],
    stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
)

# --- Process par chunks ---
reprompt_count = 0
angle_buffer = []  # rolling buffer des angles
max_buffer = int(fps * GRAPH_SECONDS)

for chunk_start in range(0, total_frames, CHUNK_SIZE):
    chunk_end = min(chunk_start + CHUNK_SIZE, total_frames)
    print(f"\n=== Chunk frames {chunk_start}-{chunk_end - 1} ===")

    if os.path.exists(CHUNK_FRAMES_DIR):
        shutil.rmtree(CHUNK_FRAMES_DIR)
    os.makedirs(CHUNK_FRAMES_DIR)

    for i, global_idx in enumerate(range(chunk_start, chunk_end)):
        src = os.path.join(FRAMES_DIR, f"{global_idx:06d}.jpg")
        dst = os.path.join(CHUNK_FRAMES_DIR, f"{i:06d}.jpg")
        os.symlink(os.path.abspath(src), dst)

    predictor = build_sam2_video_predictor(SAM2_CONFIG, SAM2_CHECKPOINT, device="cuda")
    inference_state = predictor.init_state(
        video_path=CHUNK_FRAMES_DIR,
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
    )

    prompt_det = None
    for offset in range(CHUNK_SIZE):
        global_idx = chunk_start + offset
        det = det_by_frame.get(global_idx)
        if det and not det.get("interpolated", False):
            prompt_det = (offset, det)
            break

    if prompt_det:
        local_idx, det = prompt_det
        x1, y1, x2, y2 = det["bbox"]
        cx, cy = det["centroid"]
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=local_idx,
            obj_id=1,
            box=np.array([x1*scale_x, y1*scale_y, x2*scale_x, y2*scale_y], dtype=np.float32),
            points=np.array([[cx*scale_x, cy*scale_y]], dtype=np.float32),
            labels=np.array([1], dtype=np.int32),
        )
        print(f"  Prompt at local frame {local_idx} (global {chunk_start + local_idx})")
    else:
        print(f"  No valid detection in chunk, skipping masks")

    for local_idx, obj_ids, mask_logits in predictor.propagate_in_video(inference_state):
        global_idx = chunk_start + local_idx
        frame_path = os.path.join(FRAMES_DIR, f"{global_idx:06d}.jpg")
        frame = cv2.imread(frame_path)

        if prompt_det is None:
            canvas = np.zeros((h, canvas_w, 3), dtype=np.uint8)
            canvas[:, :w] = frame
            writer.write(canvas)
            try:
                display = cv2.resize(canvas, (disp_w, disp_h))
                ffplay_proc.stdin.write(display.tobytes())
            except BrokenPipeError:
                pass
            continue

        # --- Mask ---
        mask_probs = torch.sigmoid(mask_logits)
        mask_score = mask_probs.max().item()
        binary_mask = (mask_probs[0, 0] > 0.5).cpu().numpy().astype(np.uint8)
        mask_pixels = binary_mask.sum()

        # Re-prompt si le score chute
        if mask_score < MASK_CONFIDENCE_THRESHOLD:
            det = det_by_frame.get(global_idx)
            if det and not det.get("interpolated", False):
                bx1, by1, bx2, by2 = det["bbox"]
                bcx, bcy = det["centroid"]
                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=local_idx,
                    obj_id=1,
                    box=np.array([bx1*scale_x, by1*scale_y, bx2*scale_x, by2*scale_y], dtype=np.float32),
                    points=np.array([[bcx*scale_x, bcy*scale_y]], dtype=np.float32),
                    labels=np.array([1], dtype=np.int32),
                )
                reprompt_count += 1
                print(f"  Re-prompt frame {global_idx} (mask score: {mask_score:.3f})")

        # Resize mask si nécessaire
        if binary_mask.shape[:2] != (h, w):
            binary_mask = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # --- Annotations sur la frame ---
        if mask_pixels > 0:
            overlay = frame.copy()
            overlay[binary_mask == 1] = MASK_COLOR
            frame = cv2.addWeighted(overlay, MASK_ALPHA, frame, 1 - MASK_ALPHA, 0)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, CONTOUR_COLOR, CONTOUR_THICKNESS)
        else:
            cv2.putText(
                frame, "NO MASK", (w // 3, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, max(3, w / 500), (0, 0, 255), 6,
            )

        # Keypoints
        kp = extract_keypoints(binary_mask) if mask_pixels > 0 else None
        if kp:
            csv_writer.writerow([
                global_idx,
                f"{kp['head_x']:.2f}", f"{kp['head_y']:.2f}",
                f"{kp['com_x']:.2f}", f"{kp['com_y']:.2f}",
                f"{kp['art1_x']:.2f}", f"{kp['art1_y']:.2f}",
                f"{kp['art2_x']:.2f}", f"{kp['art2_y']:.2f}",
                f"{kp['tail_x']:.2f}", f"{kp['tail_y']:.2f}",
                f"{kp['angle_rad']:.4f}",
            ])

            radius = max(6, int(w / 400))
            thickness = max(2, int(w / 800))
            font_kp = max(0.8, w / 2500)
            hpt = (int(kp["head_x"]), int(kp["head_y"]))
            cpt = (int(kp["com_x"]), int(kp["com_y"]))
            a1pt = (int(kp["art1_x"]), int(kp["art1_y"]))
            a2pt = (int(kp["art2_x"]), int(kp["art2_y"]))
            tpt = (int(kp["tail_x"]), int(kp["tail_y"]))

            cv2.drawMarker(frame, hpt, (255, 0, 0), cv2.MARKER_CROSS, radius * 2, thickness)
            cv2.putText(frame, "HEAD", (hpt[0] + 10, hpt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, font_kp, (255, 0, 0), thickness)

            # Segment tête (cephalofoil)
            if "head_p1" in kp:
                cv2.line(frame, kp["head_p1"], kp["head_p2"], (255, 0, 255), thickness)
                cv2.circle(frame, kp["head_p1"], radius, (255, 0, 255), -1)
                cv2.circle(frame, kp["head_p2"], radius, (255, 0, 255), -1)

            cv2.circle(frame, cpt, radius, (255, 255, 255), thickness)
            cv2.putText(frame, "COM", (cpt[0] + 10, cpt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, font_kp, (255, 255, 255), thickness)

            cv2.drawMarker(frame, tpt, (0, 0, 255), cv2.MARKER_TRIANGLE_UP, radius * 2, thickness)
            cv2.putText(frame, "TAIL", (tpt[0] + 10, tpt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, font_kp, (0, 0, 255), thickness)

            # Articulations = losanges verts
            cv2.drawMarker(frame, a1pt, (0, 255, 0), cv2.MARKER_DIAMOND, radius * 2, thickness)
            cv2.drawMarker(frame, a2pt, (0, 255, 0), cv2.MARKER_DIAMOND, radius * 2, thickness)

            cv2.line(frame, hpt, cpt, (255, 255, 0), thickness)
            cv2.line(frame, cpt, a1pt, (0, 255, 255), thickness)
            cv2.line(frame, a1pt, a2pt, (0, 255, 255), thickness)
            cv2.line(frame, a2pt, tpt, (0, 255, 255), thickness)

            # Angles pour le graphique
            angles = compute_angles(kp)
            angle_buffer.append(angles)
            if len(angle_buffer) > max_buffer:
                angle_buffer.pop(0)

        # Graphique sur le bandeau droit
        if len(angle_buffer) >= 2:
            graph = draw_graph(angle_buffer, fps)
        else:
            graph = np.zeros((GRAPH_H, GRAPH_W, 3), dtype=np.uint8)
            graph[:] = (30, 30, 30)

        # Composer le canvas : frame + bandeau droit
        canvas = np.zeros((h, canvas_w, 3), dtype=np.uint8)
        canvas[:, :w] = frame
        canvas[h - GRAPH_H:, w:] = graph

        # Infos texte
        font_scale = max(1, w / 1500)
        cv2.putText(
            frame, f"Frame {global_idx}",
            (20, 60), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 3,
        )

        # --- Écriture + affichage de la MÊME frame annotée ---
        writer.write(canvas)
        try:
            display = cv2.resize(canvas, (disp_w, disp_h))
            ffplay_proc.stdin.write(display.tobytes())
        except BrokenPipeError:
            print("Display window closed")
            break

    del predictor, inference_state
    torch.cuda.empty_cache()

writer.release()
csv_file.close()
ffplay_proc.stdin.close()
ffplay_proc.wait()
shutil.rmtree(CHUNK_FRAMES_DIR, ignore_errors=True)
print(f"\nDone! {OUTPUT_PATH} ({total_frames} frames, {reprompt_count} re-prompts)")
print(f"Keypoints CSV: {CSV_OUTPUT}")