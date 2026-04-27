import os
import sys
import argparse
import subprocess
import json
import csv
import glob
import shutil
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import cv2
from sam2.build_sam import build_sam2_video_predictor
from skimage.morphology import skeletonize

# --- Config ---
VIDEO_PATH = "clips/selected/SIMP2021_114-00-08_06.mp4"
OUTPUT_PATH = "projects/hammer/exports/2021_114/display_skeleton_5_points.mp4"
FRAMES_DIR = "clips/by_frames/SIMP2021_114"
CSV_OUTPUT_DIR = "projects/hammer/exports/2021_114/keypoints/"

TRACK_PATTERN = "projects/hammer/exports/2021_114/postp_tracks/*.json"
TRACK_FILES = sorted(glob.glob(TRACK_PATTERN))
print(f"Found {len(TRACK_FILES)} tracks: {[os.path.basename(f) for f in TRACK_FILES]}")

SAM2_CHECKPOINT = "sam2.1_hiera_base_plus.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"

PLOT_MARGIN = True       # False = pas de marge graphique (plus rapide)
SHOW_VIDEO = True        # False = pas de fenêtre ffplay (video enregistrée quand même)

CHUNK_SIZE = 100
MASK_CONFIDENCE_THRESHOLD = 0.85
MAX_TRACKS_PER_BATCH = 12
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

ANGLE_NAMES = [
    "angle_head_com_tail", "angle_head_com_art1", "angle_head_com_art2",
    "angle_com_art1", "angle_com_art2",
    "angle_art1_art2", "angle_art1_tail", "angle_art2_tail",
]
csv_header = [
    "frame", "head_x", "head_y", "com_x", "com_y",
    "art1_x", "art1_y", "art2_x", "art2_y",
    "tail_x", "tail_y",
] + ANGLE_NAMES


def extract_keypoints_basic(binary_mask):
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

    skeleton = skeletonize(binary_mask.astype(bool))
    skel_pts = np.argwhere(skeleton)  # (y, x)
    if len(skel_pts) > 0:
        dists = np.linalg.norm(skel_pts - np.array([com[1], com[0]]), axis=1)
        nearest = skel_pts[dists.argmin()]
        com = np.array([nearest[1], nearest[0]])

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


def compute_angles_basic(kp):
    """Calcule 6 angles : 1 articulaire (head_com_tail) + 5 directionnels."""
    head = np.array([kp["head_x"], kp["head_y"]])
    com  = np.array([kp["com_x"],  kp["com_y"]])
    a1   = np.array([kp["art1_x"], kp["art1_y"]])
    a2   = np.array([kp["art2_x"], kp["art2_y"]])
    tail = np.array([kp["tail_x"], kp["tail_y"]])

    def vec_angle(origin, target):
        v = target - origin
        return np.arctan2(v[1], v[0])

    v1 = head - com

    v2 = tail - com
    a = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
    angle_head_com_tail = (a + np.pi) % (2 * np.pi) - np.pi

    v2_2 = a1 - com
    a = np.arctan2(v2_2[1], v2_2[0]) - np.arctan2(v1[1], v1[0])
    angle_head_com_art1 = (a + np.pi) % (2 * np.pi) - np.pi



    v2_3 = a2 - com
    a = np.arctan2(v2_3[1], v2_3[0]) - np.arctan2(v1[1], v1[0])
    angle_head_com_art2 = (a + np.pi) % (2 * np.pi) - np.pi
    
    
    return {
        "angle_head_com_tail": angle_head_com_tail,
        "angle_head_com_art1":angle_head_com_art1,
        "angle_head_com_art2": angle_head_com_art2,
        "angle_com_art1":      vec_angle(com, a1),
        "angle_com_art2":      vec_angle(com, a2),
        "angle_art1_art2":     vec_angle(a1, a2),
        "angle_art1_tail":     vec_angle(a1, tail),
        "angle_art2_tail":     vec_angle(a2, tail),
    }





def build_csv_row(frame_idx, kp, angles):
    """Construit la ligne CSV selon le mode."""
    
    return [
        frame_idx,
        f"{kp['head_x']:.2f}", f"{kp['head_y']:.2f}",
        f"{kp['com_x']:.2f}", f"{kp['com_y']:.2f}",
        f"{kp['art1_x']:.2f}", f"{kp['art1_y']:.2f}",
        f"{kp['art2_x']:.2f}", f"{kp['art2_y']:.2f}",
        f"{kp['tail_x']:.2f}", f"{kp['tail_y']:.2f}",
        f"{angles['angle_head_com_tail']:.4f}",
        f"{angles['angle_head_com_art1']:.4f}",
        f"{angles['angle_head_com_art2']:.4f}",
        f"{angles['angle_com_art1']:.4f}",
        f"{angles['angle_com_art2']:.4f}",
        f"{angles['angle_art1_art2']:.4f}",
        f"{angles['angle_art1_tail']:.4f}",
        f"{angles['angle_art2_tail']:.4f}",
    ]
# ==========
# ===================================================================
# Fonctions communes
# =============================================================================

def process_track_mask(args):
    """Calcule keypoints + angles pour un track (exécuté en thread)."""
    tid, binary_mask = args
    mask_pixels = binary_mask.sum()
    if mask_pixels == 0:
        return tid, 0, None, None
    kp = extract_keypoints_basic(binary_mask)
    angles = compute_angles_basic(kp) if kp else None
    return tid, mask_pixels, kp, angles


def draw_skeleton_on_frame(frame, kp, color, w):
    """Dessine le squelette sur la frame selon le mode."""
    radius = max(6, int(w / 400))
    thickness = max(2, int(w / 800))


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

    
            
def draw_graph(angle_buffers, track_colors, fps, graph_h):
    """Dessine un graphe unique par domaine, courbes superposées."""
    graph = np.zeros((graph_h, GRAPH_W, 3), dtype=np.uint8)
    graph[:] = (30, 30, 30)

    domains = _build_basic_domains()

    _draw_domain_plots(graph, domains, angle_buffers, track_colors, fps, graph_h)
    return graph


def _build_basic_domains():

    # Exemple : 2 domaines de 3 angles chacun
    group1 = [
        ("angle_head_com_tail", "head-com-tail", (255, 200, 50)),
        ("angle_head_com_art1", "head-com-art1", (100, 200, 255)),
        ("angle_head_com_art2", "head-com-art2", (255, 100, 150)),
    ]
    group2 =  [
        ("angle_art1_tail", "head-art1-tail", (100, 255, 150)),
        ("angle_com_art1", "head-com-art1", (255, 130, 80)),
        ("angle_com_art2", "head-com-art2", (180, 130, 255)),
    ]
    return [
        {"title": "Angles (group 1)", "curves": group1},
        {"title": "Angles (group 2)", "curves": group2},
    ]




def _draw_subplot(graph, title, curves, angle_buffers, track_colors, fps,
                  gx0, gy0, gx1, gy1, cx0):
    """Dessine un subplot avec N courbes superposées (tous tracks)."""
    gw, gh = gx1 - gx0, gy1 - gy0
    if gw < 10 or gh < 10:
        return

    # Titre
    cv2.putText(graph, title, (gx0, gy0 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)

    # Collecter toutes les valeurs pour l'échelle Y commune
    all_vals = []
    for key, label, color in curves:
        for buf in angle_buffers.values():
            all_vals.extend([e[key] for e in buf if key in e])
    if len(all_vals) < 2:
        return

    y_min, y_max = min(all_vals) - 0.1, max(all_vals) + 0.1
    if y_max - y_min < 0.2:
        y_min -= 0.1
        y_max += 0.1

    # Cadre
    cv2.rectangle(graph, (gx0, gy0), (gx1, gy1), (60, 60, 60), 1)

    # Ligne du zéro
    if y_min < 0 < y_max:
        zy = int(gy0 + gh * (1 - (0 - y_min) / (y_max - y_min)))
        cv2.line(graph, (gx0, zy), (gx1, zy), (80, 80, 80), 1)

    # Labels Y
    for val in np.linspace(y_min, y_max, 4):
        py = int(gy0 + gh * (1 - (val - y_min) / (y_max - y_min)))
        cv2.putText(graph, f"{val:.1f}", (cx0 + 2, py + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (120, 120, 120), 1)

    # Labels X
    max_n = max((len(b) for b in angle_buffers.values()), default=1)
    max_time = max_n / fps
    for t in range(0, int(max_time) + 1, max(1, int(max_time) // 4)):
        px = int(gx0 + gw * t / max(max_time, 0.1))
        if px <= gx1:
            cv2.putText(graph, f"{t}s", (px - 5, gy1 + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (120, 120, 120), 1)

    # Courbes
    for key, label, curve_color in curves:
        for tid, buf in angle_buffers.items():
            if len(buf) < 2:
                continue
            # Mélanger couleur courbe + couleur track
            tc = track_colors[tid]
            mix = (
                int(curve_color[0] * 0.6 + tc[0] * 0.4),
                int(curve_color[1] * 0.6 + tc[1] * 0.4),
                int(curve_color[2] * 0.6 + tc[2] * 0.4),
            )
            n = len(buf)
            points = []
            for i, entry in enumerate(buf):
                if key not in entry:
                    continue
                px = int(gx0 + gw * i / (n - 1))
                val = entry[key]
                py = int(gy0 + gh * (1 - (val - y_min) / (y_max - y_min)))
                points.append((px, py))
            for i in range(len(points) - 1):
                cv2.line(graph, points[i], points[i + 1], mix, 2)

    # Légende des courbes dans le subplot
    lx = gx1 - 80
    ly = gy0 + 12
    for key, label, curve_color in curves:
        cv2.line(graph, (lx, ly), (lx + 15, ly), curve_color, 2)
        cv2.putText(graph, label, (lx + 18, ly + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, curve_color, 1)
        ly += 12



def _draw_domain_plots(graph, domains, angle_buffers, track_colors, fps, graph_h):
    """Un subplot par domaine, empilés verticalement."""
    pad_x, pad_y = 5, 5
    # n_plots = len(domains)
    n_plots = 3
    cell_h = (graph_h - 2 * pad_y) // n_plots
    margin_l, margin_b, margin_t = 40, 18, 18

    for idx, domain in enumerate(domains):
        cy0 = pad_y + idx * cell_h
        gx0 = pad_x + margin_l
        gy0 = cy0 + margin_t
        gx1 = GRAPH_W - 8
        gy1 = cy0 + cell_h - margin_b
        _draw_subplot(graph, domain["title"], domain["curves"],
                      angle_buffers, track_colors, fps,
                      gx0, gy0, gx1, gy1, pad_x)


def send_to_display(canvas):
    """Envoie une frame au processus ffplay."""
    global display_alive
    if not display_alive or ffplay_proc is None:
        return
    if ffplay_proc.poll() is not None:
        display_alive = False
        return
    ffplay_proc.stdin.write(cv2.resize(canvas, (disp_w, disp_h)).tobytes())


# =============================================================================
# Setup
# =============================================================================

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

# Track avec le plus de détections (pour PLOT_MARGIN: 1 seule track affichée)
best_track_tid = max(tracks, key=lambda tid: sum(
    1 for d in tracks[tid]["detections"] if not d.get("interpolated", False)
))
print(f"Best track for graph: {best_track_tid} ({sum(1 for d in tracks[best_track_tid]['detections'] if not d.get('interpolated', False))} real dets)")

# --- CSV ---
os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)
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
canvas_w = w + GRAPH_W if PLOT_MARGIN else w
writer = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (canvas_w, h))

# --- Live display ---
disp_w, disp_h = int(canvas_w * 0.35), int(h * 0.35)
ffplay_proc = None
display_alive = False
if SHOW_VIDEO:
    ffplay_proc = subprocess.Popen(
        ["ffplay", "-f", "rawvideo", "-pixel_format", "bgr24",
         "-video_size", f"{disp_w}x{disp_h}", "-framerate", str(fps),
         "-window_title", "SAM2 Sharks", "-"],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    display_alive = True

thread_pool = ThreadPoolExecutor(max_workers=NUM_WORKERS)


# =============================================================================
# Processing
# =============================================================================

reprompt_count = 0
max_buffer = int(fps * GRAPH_SECONDS)
angle_buffers = {tid: [] for tid in tracks}
latest_kp = {tid: None for tid in tracks}  # pour la vue squelette normalisée

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

    if not active_tids:
        for i in range(chunk_len):
            gi = chunk_start + i
            frame = cv2.imread(os.path.join(FRAMES_DIR, f"{gi:06d}.jpg"))
            canvas = np.zeros((h, canvas_w, 3), dtype=np.uint8)
            canvas[:, :w] = frame
            if PLOT_MARGIN:
                graph_buffers = {best_track_tid: angle_buffers[best_track_tid]}
                graph_colors = {best_track_tid: track_colors[best_track_tid]}
                canvas[:, w:] = draw_graph(graph_buffers, graph_colors, fps, h)
            writer.write(canvas)
            send_to_display(canvas)
        continue

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

            for local_idx, obj_ids, mask_logits in predictor.propagate_in_video(inference_state):
                gi = chunk_start + local_idx
                for i, sam2_id in enumerate(obj_ids):
                    tid = batch_tids[sam2_id - 1]
                    mask_probs = torch.sigmoid(mask_logits[i])
                    mask_score = mask_probs.max().item()
                    binary_mask = (mask_probs[0] > 0.5).cpu().numpy().astype(np.uint8)

                    if mask_score < MASK_CONFIDENCE_THRESHOLD:
                        det = tracks[tid]["det_by_frame"].get(gi)
                        already_prompted = (
                            det is not None
                            and not det.get("interpolated", False)
                        )
                        if det and not already_prompted:
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

        # 1) Masques
        for tid, (binary_mask, mask_score) in frame_tracks.items():
            if binary_mask.sum() > 0:
                overlay = frame.copy()
                overlay[binary_mask == 1] = tracks[tid]["color"]
                frame = cv2.addWeighted(overlay, MASK_ALPHA, frame, 1 - MASK_ALPHA, 0)
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(frame, contours, -1, tracks[tid]["color"], CONTOUR_THICKNESS)

        # 2) Keypoints en parallèle
        tasks = [(tid, binary_mask) for tid, (binary_mask, _) in frame_tracks.items()]
        results = list(thread_pool.map(process_track_mask, tasks))

        # 3) Dessiner + CSV
        for tid, mask_pixels, kp, angles in results:
            if not kp or not angles:
                continue
            color = tracks[tid]["color"]

            csv_writers[tid].writerow(build_csv_row(gi, kp, angles))

            draw_skeleton_on_frame(frame, kp, color, w)

            angle_buffers[tid].append(angles)
            if len(angle_buffers[tid]) > max_buffer:
                angle_buffers[tid].pop(0)

            # Stocker les keypoints les plus récents pour la vue squelette
            latest_kp[tid] = kp

        font_scale = max(1, w / 1500)
        cv2.putText(frame, f"Frame {gi}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 3)

        canvas = np.zeros((h, canvas_w, 3), dtype=np.uint8)
        canvas[:, :w] = frame

        if PLOT_MARGIN:
            graph_buffers = {best_track_tid: angle_buffers[best_track_tid]}
            graph_colors = {best_track_tid: track_colors[best_track_tid]}
            canvas[:, w:] = draw_graph(graph_buffers, graph_colors, fps, h)

        writer.write(canvas)
        send_to_display(canvas)


# =============================================================================
# Cleanup
# =============================================================================

writer.release()
thread_pool.shutdown()
for f in csv_files.values():
    f.close()
if display_alive and ffplay_proc is not None:
    ffplay_proc.stdin.close()
    ffplay_proc.wait()
shutil.rmtree(CHUNK_FRAMES_DIR, ignore_errors=True)
print(f"\nDone! {OUTPUT_PATH} ({total_frames} frames, {reprompt_count} re-prompts)")
print(f"Keypoints CSVs in: {CSV_OUTPUT_DIR}")