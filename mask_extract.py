import os
import subprocess
import json
import csv
import glob
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import cv2
from sam2.build_sam import build_sam2_video_predictor

# --- Config ---
VIDEO_PATH = "clips/selected/SIMP2021_057-13_10-16_25.mp4"
OUTPUT_PATH = "projects/hammer/exports/2021_057/output_masked.mp4"
FRAMES_DIR = "clips/by_frames/SIMP2021_057"
CSV_OUTPUT_DIR = "projects/hammer/exports/2021_057/keypoints/"

TRACK_PATTERN = "projects/hammer/exports/2021_057/postp_tracks/*.json"
TRACK_FILES = sorted(glob.glob(TRACK_PATTERN))
print(f"Found {len(TRACK_FILES)} tracks: {[os.path.basename(f) for f in TRACK_FILES]}")

SAM2_CHECKPOINT = "sam2.1_hiera_base_plus.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"

CHUNK_SIZE = 200
MASK_CONFIDENCE_THRESHOLD = 0.85
MAX_TRACKS_PER_BATCH = 6
MASK_ALPHA = 0.45
CONTOUR_THICKNESS = 4

GRAPH_SECONDS = 5
GRAPH_W = 600
GRAPH_H = 400
GRAPH_MARGIN = 40

CHUNK_FRAMES_DIR = "/tmp/sam2_chunk"
NUM_WORKERS = min(8, os.cpu_count() or 4)

TRACK_COLORS = [
    (0, 200, 255), (255, 100, 0), (0, 255, 100), (200, 0, 255),
    (255, 200, 0), (0, 100, 255), (255, 0, 200), (100, 255, 0),
    (128, 200, 100), (200, 128, 255), (100, 255, 200), (255, 128, 50),
    (50, 200, 200), (200, 50, 128), (128, 255, 50), (50, 128, 255),
    (255, 50, 128), (128, 50, 200), (50, 255, 128), (200, 255, 50),
]


# =============================================================================
# Fonctions de calcul
# =============================================================================

def extract_keypoints(binary_mask):
    """Extrait head (milieu du cephalofoil), COM, art1, art2, tail."""
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

    # Tail = point le plus loin du COM
    dists_to_com = np.linalg.norm(pts - com, axis=1)
    tail = pts[dists_to_com.argmax()]

    body_vec = tail - com
    body_len = np.linalg.norm(body_vec)
    if body_len == 0:
        return None
    body_axis = body_vec / body_len
    perp_axis = np.array([-body_axis[1], body_axis[0]])

    proj_along = (pts - com) @ body_axis
    min_proj = proj_along.min()
    n_steps = 80
    step_size = abs(min_proj) / max(n_steps, 1)

    # Head : coupe la plus large côté tête × distance au COM
    best_score = 0
    best_p1 = best_p2 = None
    for t in np.linspace(min_proj, 0, n_steps):
        band = np.abs(proj_along - t) < max(step_size, 2.0)
        if band.sum() < 2:
            continue
        nearby = pts[band]
        perp_proj = (nearby - com) @ perp_axis
        width = perp_proj.max() - perp_proj.min()
        dist_from_com = abs(t) / max(abs(min_proj), 1e-6)
        score = width * dist_from_com
        if score > best_score:
            best_score = score
            best_p1 = nearby[perp_proj.argmin()]
            best_p2 = nearby[perp_proj.argmax()]

    if best_p1 is None:
        dists_to_tail = np.linalg.norm(pts - tail, axis=1)
        head = pts[dists_to_tail.argmax()]
        head_p1, head_p2 = None, None
    else:
        head = (best_p1 + best_p2) / 2
        head_p1, head_p2 = best_p1, best_p2

    # Articulations à 1/3 et 2/3 entre COM et tail
    articulations = []
    for frac in [1 / 3, 2 / 3]:
        target_t = body_len * frac
        band_art = np.abs(proj_along - target_t) < max(step_size * 2, 3.0)
        if band_art.sum() >= 2:
            nearby_art = pts[band_art]
            perp_art = (nearby_art - com) @ perp_axis
            articulations.append((nearby_art[perp_art.argmin()] + nearby_art[perp_art.argmax()]) / 2)
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
    if head_p1 is not None:
        result["head_p1"] = (int(head_p1[0]), int(head_p1[1]))
        result["head_p2"] = (int(head_p2[0]), int(head_p2[1]))
    return result


def compute_angles(kp):
    """Calcule les 3 angles des vecteurs vers la queue."""
    tail = np.array([kp["tail_x"], kp["tail_y"]])
    com = np.array([kp["com_x"], kp["com_y"]])
    a1 = np.array([kp["art1_x"], kp["art1_y"]])
    a2 = np.array([kp["art2_x"], kp["art2_y"]])
    def vec_angle(o, t):
        v = t - o
        return np.arctan2(v[1], v[0])
    return {
        "com_tail": vec_angle(com, tail),
        "art1_tail": vec_angle(a1, tail),
        "art2_tail": vec_angle(a2, tail),
    }


def process_track_mask(args):
    """Calcule keypoints + angles pour un track (exécuté en thread)."""
    tid, binary_mask = args
    mask_pixels = binary_mask.sum()
    if mask_pixels == 0:
        return tid, 0, None, None
    kp = extract_keypoints(binary_mask)
    angles = compute_angles(kp) if kp else None
    return tid, mask_pixels, kp, angles


def draw_graph(angle_buffers, track_colors, fps):
    """Dessine le graphique art2_tail pour chaque track."""
    graph = np.zeros((GRAPH_H, GRAPH_W, 3), dtype=np.uint8)
    graph[:] = (30, 30, 30)
    gx0, gx1 = GRAPH_MARGIN, GRAPH_W - 10
    gy0, gy1 = 10, GRAPH_H - GRAPH_MARGIN
    gw, gh = gx1 - gx0, gy1 - gy0

    all_vals = []
    for buf in angle_buffers.values():
        all_vals.extend([e["art2_tail"] for e in buf])
    if len(all_vals) < 2:
        return graph

    y_min, y_max = min(all_vals) - 0.1, max(all_vals) + 0.1
    if y_max - y_min < 0.2:
        y_min -= 0.1
        y_max += 0.1

    zero_y = int(gy0 + gh * (1 - (0 - y_min) / (y_max - y_min)))
    if gy0 < zero_y < gy1:
        cv2.line(graph, (gx0, zero_y), (gx1, zero_y), (80, 80, 80), 1)

    for val in np.linspace(y_min, y_max, 5):
        py = int(gy0 + gh * (1 - (val - y_min) / (y_max - y_min)))
        cv2.putText(graph, f"{val:.1f}", (2, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

    max_n = max((len(b) for b in angle_buffers.values()), default=1)
    max_time = max_n / fps
    for t in range(0, int(max_time) + 1):
        px = int(gx0 + gw * t / max(max_time, 0.1))
        if px <= gx1:
            cv2.putText(graph, f"{t}s", (px - 5, GRAPH_H - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

    lx, ly = gx0 + 5, gy0 + 15
    for tid, buf in angle_buffers.items():
        if len(buf) < 2:
            continue
        color = track_colors[tid]
        n = len(buf)
        points = []
        for i, entry in enumerate(buf):
            px = int(gx0 + gw * i / (n - 1))
            val = entry["art2_tail"]
            py = int(gy0 + gh * (1 - (val - y_min) / (y_max - y_min)))
            points.append((px, py))
        for i in range(len(points) - 1):
            cv2.line(graph, points[i], points[i + 1], color, 2)
        cv2.line(graph, (lx, ly), (lx + 20, ly), color, 2)
        cv2.putText(graph, f"T{tid}", (lx + 25, ly + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        ly += 18

    return graph


def send_to_display(canvas):
    """Envoie une frame au processus ffplay."""
    global display_alive
    if not display_alive:
        return
    if ffplay_proc.poll() is not None:
        display_alive = False
        return
    ffplay_proc.stdin.write(cv2.resize(canvas, (disp_w, disp_h)).tobytes())


# =============================================================================
# Setup
# =============================================================================

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

# --- Load all tracks ---
tracks = {}
for tid, path in enumerate(TRACK_FILES):
    with open(path) as f:
        data = json.load(f)
    dets = data["detections"]
    color = TRACK_COLORS[tid % len(TRACK_COLORS)]
    tracks[tid] = {
        "detections": dets,
        "det_by_frame": {d["frame"]: d for d in dets},
        "first_frame": data.get("first_frame", dets[0]["frame"] if dets else 0),
        "last_frame": data.get("last_frame", dets[-1]["frame"] if dets else 0),
        "color": color,
    }
    print(f"Track {tid}: frames {tracks[tid]['first_frame']}-{tracks[tid]['last_frame']} ({len(dets)} dets)")

track_colors = {tid: t["color"] for tid, t in tracks.items()}

# --- CSV : un fichier par track ---
os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)
csv_header = [
    "frame", "head_x", "head_y", "com_x", "com_y",
    "art1_x", "art1_y", "art2_x", "art2_y",
    "tail_x", "tail_y",
    "angle_rad", "angle_com_tail", "angle_art1_tail", "angle_art2_tail",
]
csv_files = {}
csv_writers = {}
for tid, path in enumerate(TRACK_FILES):
    basename = os.path.splitext(os.path.basename(path))[0]
    csv_path = os.path.join(CSV_OUTPUT_DIR, f"{basename}.csv")
    f = open(csv_path, "w", newline="")
    w_csv = csv.writer(f)
    w_csv.writerow(csv_header)
    csv_files[tid] = f
    csv_writers[tid] = w_csv

# --- Video writer ---
canvas_w = w + GRAPH_W
writer = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (canvas_w, h))

# --- Live display ---
disp_w, disp_h = int(canvas_w * 0.35), int(h * 0.35)
ffplay_proc = subprocess.Popen(
    ["ffplay", "-f", "rawvideo", "-pixel_format", "bgr24",
     "-video_size", f"{disp_w}x{disp_h}", "-framerate", str(fps),
     "-window_title", "SAM2 Sharks", "-"],
    stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
)
display_alive = True

# --- Thread pool ---
thread_pool = ThreadPoolExecutor(max_workers=NUM_WORKERS)


# =============================================================================
# Processing
# =============================================================================

reprompt_count = 0
max_buffer = int(fps * GRAPH_SECONDS)
angle_buffers = {tid: [] for tid in tracks}

for chunk_start in range(0, total_frames, CHUNK_SIZE):
    chunk_end = min(chunk_start + CHUNK_SIZE, total_frames)
    chunk_len = chunk_end - chunk_start
    print(f"\n=== Chunk frames {chunk_start}-{chunk_end - 1} ===")

    # Symlinks
    if os.path.exists(CHUNK_FRAMES_DIR):
        shutil.rmtree(CHUNK_FRAMES_DIR)
    os.makedirs(CHUNK_FRAMES_DIR)
    for i, gi in enumerate(range(chunk_start, chunk_end)):
        src = os.path.join(FRAMES_DIR, f"{gi:06d}.jpg")
        dst = os.path.join(CHUNK_FRAMES_DIR, f"{i:06d}.jpg")
        os.symlink(os.path.abspath(src), dst)

    # Trouver les tracks actifs dans ce chunk (au moins 1 détection réelle)
    active_tids = []
    for tid, t in tracks.items():
        has_real = any(
            chunk_start + offset in t["det_by_frame"]
            and not t["det_by_frame"][chunk_start + offset].get("interpolated", False)
            for offset in range(chunk_len)
        )
        if has_real:
            active_tids.append(tid)

    print(f"  {len(active_tids)} active tracks")

    # Chunk sans tracks actifs
    if not active_tids:
        for i in range(chunk_len):
            gi = chunk_start + i
            frame = cv2.imread(os.path.join(FRAMES_DIR, f"{gi:06d}.jpg"))
            canvas = np.zeros((h, canvas_w, 3), dtype=np.uint8)
            canvas[:, :w] = frame
            canvas[h - GRAPH_H:, w:] = draw_graph(angle_buffers, track_colors, fps)
            writer.write(canvas)
            send_to_display(canvas)
        continue

    # Traiter par sous-groupes
    batches = [active_tids[i:i + MAX_TRACKS_PER_BATCH]
               for i in range(0, len(active_tids), MAX_TRACKS_PER_BATCH)]

    chunk_results = defaultdict(dict)

    for batch_idx, batch_tids in enumerate(batches):
        print(f"  Batch {batch_idx + 1}/{len(batches)}: tracks {batch_tids}")

        predictor = build_sam2_video_predictor(SAM2_CONFIG, SAM2_CHECKPOINT, device="cuda")
        inference_state = predictor.init_state(
            video_path=CHUNK_FRAMES_DIR,
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
        )

        with torch.autocast("cuda", dtype=torch.bfloat16):

            # ═══════════════════════════════════════════════════════════
            # CHANGEMENT PRINCIPAL : prompter TOUTES les détections
            # réelles du chunk AVANT la propagation.
            # ═══════════════════════════════════════════════════════════
            prompt_count = 0
            for local_idx in range(chunk_len):
                gi = chunk_start + local_idx
                for tid in batch_tids:
                    det = tracks[tid]["det_by_frame"].get(gi)
                    if det and not det.get("interpolated", False):
                        sam2_id = batch_tids.index(tid) + 1
                        x1, y1, x2, y2 = det["bbox"]
                        cx, cy = det["centroid"]
                        predictor.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=local_idx,
                            obj_id=sam2_id,
                            box=np.array([x1*scale_x, y1*scale_y,
                                          x2*scale_x, y2*scale_y], dtype=np.float32),
                            points=np.array([[cx*scale_x, cy*scale_y]], dtype=np.float32),
                            labels=np.array([1], dtype=np.int32),
                        )
                        prompt_count += 1

            print(f"    {prompt_count} prompts added before propagation")

            # ═══════════════════════════════════════════════════════════
            # Propagation — le re-prompt réactif reste en filet de
            # sécurité pour les frames sans détection tracker.
            # ═══════════════════════════════════════════════════════════
            for local_idx, obj_ids, mask_logits in predictor.propagate_in_video(inference_state):
                gi = chunk_start + local_idx
                for i, sam2_id in enumerate(obj_ids):
                    tid = batch_tids[sam2_id - 1]
                    mask_probs = torch.sigmoid(mask_logits[i])
                    mask_score = mask_probs.max().item()
                    binary_mask = (mask_probs[0] > 0.5).cpu().numpy().astype(np.uint8)

                    # Re-prompt réactif : uniquement si score bas ET
                    # pas déjà prompté (= frame interpolée ou sans det)
                    if mask_score < MASK_CONFIDENCE_THRESHOLD:
                        det = tracks[tid]["det_by_frame"].get(gi)
                        already_prompted = (
                            det is not None
                            and not det.get("interpolated", False)
                        )
                        if det and not already_prompted:
                            # Frame interpolée mais on a quand même une bbox
                            bx1, by1, bx2, by2 = det["bbox"]
                            bcx, bcy = det["centroid"]
                            predictor.add_new_points_or_box(
                                inference_state=inference_state,
                                frame_idx=local_idx,
                                obj_id=sam2_id,
                                box=np.array([bx1*scale_x, by1*scale_y,
                                              bx2*scale_x, by2*scale_y], dtype=np.float32),
                                points=np.array([[bcx*scale_x, bcy*scale_y]], dtype=np.float32),
                                labels=np.array([1], dtype=np.int32),
                            )
                            reprompt_count += 1

                    if binary_mask.shape[:2] != (h, w):
                        binary_mask = cv2.resize(binary_mask, (w, h),
                                                 interpolation=cv2.INTER_NEAREST)

                    chunk_results[local_idx][tid] = (binary_mask, mask_score)

        del predictor, inference_state
        torch.cuda.empty_cache()

    # --- Composer les frames ---
    for local_idx in range(chunk_len):
        gi = chunk_start + local_idx
        frame = cv2.imread(os.path.join(FRAMES_DIR, f"{gi:06d}.jpg"))
        frame_tracks = chunk_results.get(local_idx, {})

        # 1) Appliquer tous les masques
        for tid, (binary_mask, mask_score) in frame_tracks.items():
            if binary_mask.sum() > 0:
                overlay = frame.copy()
                overlay[binary_mask == 1] = tracks[tid]["color"]
                frame = cv2.addWeighted(overlay, MASK_ALPHA, frame, 1 - MASK_ALPHA, 0)
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(frame, contours, -1, tracks[tid]["color"], CONTOUR_THICKNESS)

        # 2) Calculer keypoints en parallèle
        tasks = [(tid, binary_mask) for tid, (binary_mask, _) in frame_tracks.items()]
        results = list(thread_pool.map(process_track_mask, tasks))

        # 3) Dessiner + CSV
        for tid, mask_pixels, kp, angles in results:
            if not kp or not angles:
                continue
            color = tracks[tid]["color"]

            csv_writers[tid].writerow([
                gi,
                f"{kp['head_x']:.2f}", f"{kp['head_y']:.2f}",
                f"{kp['com_x']:.2f}", f"{kp['com_y']:.2f}",
                f"{kp['art1_x']:.2f}", f"{kp['art1_y']:.2f}",
                f"{kp['art2_x']:.2f}", f"{kp['art2_y']:.2f}",
                f"{kp['tail_x']:.2f}", f"{kp['tail_y']:.2f}",
                f"{kp['angle_rad']:.4f}",
                f"{angles['com_tail']:.4f}",
                f"{angles['art1_tail']:.4f}",
                f"{angles['art2_tail']:.4f}",
            ])

            radius = max(6, int(w / 400))
            thickness = max(2, int(w / 800))
            font_kp = max(0.8, w / 2500)
            hpt = (int(kp["head_x"]), int(kp["head_y"]))
            cpt = (int(kp["com_x"]), int(kp["com_y"]))
            a1pt = (int(kp["art1_x"]), int(kp["art1_y"]))
            a2pt = (int(kp["art2_x"]), int(kp["art2_y"]))
            tpt = (int(kp["tail_x"]), int(kp["tail_y"]))

            cv2.drawMarker(frame, hpt, color, cv2.MARKER_CROSS, radius * 2, thickness)
            if "head_p1" in kp:
                cv2.line(frame, kp["head_p1"], kp["head_p2"], color, thickness)
                cv2.circle(frame, kp["head_p1"], radius, color, -1)
                cv2.circle(frame, kp["head_p2"], radius, color, -1)
            cv2.circle(frame, cpt, radius, (255, 255, 255), thickness)
            cv2.drawMarker(frame, a1pt, color, cv2.MARKER_DIAMOND, radius * 2, thickness)
            cv2.drawMarker(frame, a2pt, color, cv2.MARKER_DIAMOND, radius * 2, thickness)
            cv2.drawMarker(frame, tpt, color, cv2.MARKER_TRIANGLE_UP, radius * 2, thickness)
            cv2.line(frame, hpt, cpt, color, thickness)
            cv2.line(frame, cpt, a1pt, color, thickness)
            cv2.line(frame, a1pt, a2pt, color, thickness)
            cv2.line(frame, a2pt, tpt, color, thickness)

            angle_buffers[tid].append(angles)
            if len(angle_buffers[tid]) > max_buffer:
                angle_buffers[tid].pop(0)

        font_scale = max(1, w / 1500)
        cv2.putText(frame, f"Frame {gi}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 3)

        canvas = np.zeros((h, canvas_w, 3), dtype=np.uint8)
        canvas[:, :w] = frame
        canvas[h - GRAPH_H:, w:] = draw_graph(angle_buffers, track_colors, fps)

        writer.write(canvas)
        send_to_display(canvas)


# =============================================================================
# Cleanup
# =============================================================================

writer.release()
thread_pool.shutdown()
for f in csv_files.values():
    f.close()
if display_alive:
    ffplay_proc.stdin.close()
    ffplay_proc.wait()
shutil.rmtree(CHUNK_FRAMES_DIR, ignore_errors=True)
print(f"\nDone! {OUTPUT_PATH} ({total_frames} frames, {reprompt_count} re-prompts)")
print(f"Keypoints CSVs in: {CSV_OUTPUT_DIR}")