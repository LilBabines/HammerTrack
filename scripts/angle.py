"""
Render OBB + cohesion + compass on video.

Track identity
  Every track is identified by the stem of its JSON filename (e.g. "shark_3").
  This ID is used consistently for: video labels, CSV columns, cohesion lookup,
  and panel display.  There is NO dependency on merged_track_ids.

Angle computation (image space):
  1. CMC-warped trail → trajectory direction
  2. OBB long axis smoothed (×2 trick, π-periodic)
  3. Disambiguate axis with trail direction → directed angle
  4. Local ±π fix: EMA + trajectory blend (no cumulative offset)
  5. Arrow on video = angle_img
  6. Compass = angle_img - cum_rot - ref

Cohesion (per frame, independent):
  T(t) = median bbox diagonal at frame t
  cohesion_i(t) = median(dist(i,j) for j present at t) / T(t)
  cohesion_global(t) = mean(cohesion_i(t))
"""

import os
import csv
import cv2
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from scipy.signal import savgol_filter


# ── JSON loaders ─────────────────────────────────────────────────────

def load_track_json(p):
    """Load a post-processed track JSON.  ID = filename stem (e.g. 'shark_3')."""
    with open(p) as f:
        track = json.load(f)
    track["id"] = Path(p).stem          # "shark_3"
    return track


def load_cmc_json(p):
    """Load a CMC transforms JSON (plain dict, no extra keys)."""
    with open(p) as f:
        return json.load(f)


# ── CMC ──────────────────────────────────────────────────────────────

def cmc_rotation_angle(affine):
    a, b = affine[0][0], affine[0][1]
    c, d = affine[1][0], affine[1][1]
    return (np.arctan2(c, a) + np.arctan2(-b, d)) / 2.0


def build_cum_rot(cmc, start, end):
    cum, total = {}, 0.0
    for f in range(start, end):
        fk = str(f)
        if fk in cmc:
            total += cmc_rotation_angle(cmc[fk])
        cum[f] = total
    return cum


# ── Angle math ───────────────────────────────────────────────────────

def obb_long_edge_angle(obb):
    pts = np.array(obb)
    e1, e2 = pts[1] - pts[0], pts[2] - pts[1]
    return np.arctan2(e1[1], e1[0]) if np.linalg.norm(e1) >= np.linalg.norm(e2) \
        else np.arctan2(e2[1], e2[0])


def angle_diff(a, b):
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def disambiguate(axis_angle, direction):
    c = [axis_angle, axis_angle + np.pi]
    best = min(c, key=lambda x: abs(angle_diff(x, direction)))
    return (best + np.pi) % (2 * np.pi) - np.pi


def circular_mean(angles):
    return np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))


def circular_blend(a, b, alpha):
    return np.arctan2(
        alpha * np.sin(a) + (1 - alpha) * np.sin(b),
        alpha * np.cos(a) + (1 - alpha) * np.cos(b),
    )


# ── Savgol ───────────────────────────────────────────────────────────

def _sw(n, w, p):
    w = min(w, n)
    if w % 2 == 0: w -= 1
    w = max(w, p + 2)
    if w % 2 == 0: w += 1
    return w


def smooth(v, w=51, p=2):
    if len(v) <= p: return v
    return savgol_filter(v, _sw(len(v), w, p), p)


def smooth_axis(a, w=51, p=2):
    if len(a) <= p: return a
    return savgol_filter(np.unwrap(a * 2.0), _sw(len(a), w, p), p) / 2.0


def smooth_dir(a, w=51, p=2):
    if len(a) <= p: return a
    s = savgol_filter(np.unwrap(a), _sw(len(a), w, p), p)
    return (s + np.pi) % (2 * np.pi) - np.pi


# ── Fix ±180° — local EMA, no cumulative offset ─────────────────────

def fix_pi_jumps_local(angle_abs, traj_dir_abs, ema_alpha=0.85):
    valid_idx = np.where(~np.isnan(angle_abs))[0]
    if len(valid_idx) < 2:
        return angle_abs

    def _pass(indices, angles):
        out = angles.copy()
        ema = out[indices[0]]
        for k in range(1, len(indices)):
            i = indices[k]
            raw = out[i]
            candidates = [raw, raw + np.pi, raw - np.pi]

            traj = traj_dir_abs[i] if not np.isnan(traj_dir_abs[i]) else np.nan

            if not np.isnan(traj):
                best = min(candidates, key=lambda c: abs(angle_diff(c, traj)))
            else:
                best = min(candidates, key=lambda c: abs(angle_diff(c, ema)))

            out[i] = best
            ema = circular_blend(ema, best, ema_alpha)
        return out

    fixed_fwd = _pass(valid_idx, angle_abs.copy())
    fixed_bwd = _pass(valid_idx[::-1], angle_abs.copy())

    merged = fixed_fwd.copy()
    for k in range(len(valid_idx)):
        i = valid_idx[k]
        fwd_val = fixed_fwd[i]
        bwd_val = fixed_bwd[i]

        if abs(angle_diff(fwd_val, bwd_val)) < 0.1:
            merged[i] = fwd_val
        else:
            traj = traj_dir_abs[i] if not np.isnan(traj_dir_abs[i]) else np.nan
            if not np.isnan(traj):
                merged[i] = min([fwd_val, bwd_val],
                                key=lambda c: abs(angle_diff(c, traj)))
            else:
                merged[i] = fwd_val

    return merged


# ── Preprocess: smooth centroids ─────────────────────────────────────

def preprocess_track(track, sw, sp):
    dets = sorted(track["detections"], key=lambda d: d["frame"])
    if len(dets) < 2:
        return track

    cx = np.array([d["centroid"][0] for d in dets])
    cy = np.array([d["centroid"][1] for d in dets])
    cx_s, cy_s = smooth(cx, sw, sp), smooth(cy, sw, sp)

    out = []
    for i, d in enumerate(dets):
        dd = d.copy()
        dd["centroid"] = [float(cx_s[i]), float(cy_s[i])]
        out.append(dd)

    r = track.copy()
    r["detections"] = out
    return r


# ── Compute angles ───────────────────────────────────────────────────

def compute_angles(track, cmc, cum_rot, trail_length, smooth_window, sw, sp, n_ref):
    dets = track["detections"]
    n = len(dets)
    frames = [d["frame"] for d in dets]

    # ── Trajectory direction in image space (from CMC-warped trail)
    trail = []
    traj_dir_img = np.full(n, np.nan)

    for i, det in enumerate(dets):
        fk = str(det["frame"])
        if fk in cmc:
            M = np.array(cmc[fk])
            for j in range(len(trail)):
                pt = trail[j]
                trail[j] = M @ np.array([pt[0], pt[1], 1.0])

        c = np.array(det["centroid"], dtype=np.float64)
        trail.append(c.copy())
        if len(trail) > trail_length:
            trail = trail[-trail_length:]

        if len(trail) >= 2:
            half = min(smooth_window // 2, len(trail) - 1)
            delta = trail[-1] - trail[-1 - half]
            if np.linalg.norm(delta) > 1e-3:
                traj_dir_img[i] = np.arctan2(delta[1], delta[0])

    # ── OBB axis (π-periodic, smoothed)
    axis_img = np.full(n, np.nan)
    obb_mask = np.zeros(n, dtype=bool)
    for i, det in enumerate(dets):
        obb = det.get("obb")
        if obb and len(obb) == 4:
            axis_img[i] = obb_long_edge_angle(obb)
            obb_mask[i] = True

    obb_idx = np.where(obb_mask)[0]
    sm_axis = smooth_axis(axis_img[obb_idx], sw, sp) if len(obb_idx) > 2 else axis_img[obb_idx]

    traj_idx = np.where(~np.isnan(traj_dir_img))[0]
    sm_traj = smooth_dir(traj_dir_img[traj_idx], sw, sp) if len(traj_idx) > 2 else traj_dir_img[traj_idx]

    axis_lut = {int(obb_idx[k]): float(sm_axis[k]) for k in range(len(obb_idx))}
    traj_lut = {int(traj_idx[k]): float(sm_traj[k]) for k in range(len(traj_idx))}

    # ── Initial disambiguation: axis + trajectory
    angle_img_result = np.full(n, np.nan)
    prev_dir = None

    for i in range(n):
        ax = axis_lut.get(i)
        tr = traj_lut.get(i)

        if ax is not None:
            if tr is not None:
                d = disambiguate(ax, tr)
                prev_dir = d
            elif prev_dir is not None:
                d = disambiguate(ax, prev_dir)
            else:
                d = ax
                prev_dir = d
            angle_img_result[i] = d
        elif prev_dir is not None:
            angle_img_result[i] = prev_dir

    # ── angle_abs = directed angle in stabilized space
    angle_abs = np.full(n, np.nan)
    traj_dir_abs = np.full(n, np.nan)
    for i in range(n):
        cr = cum_rot.get(frames[i], 0.0)
        if not np.isnan(angle_img_result[i]):
            angle_abs[i] = angle_img_result[i] - cr
        if not np.isnan(traj_dir_img[i]):
            traj_dir_abs[i] = traj_dir_img[i] - cr

    # ── Local ±π fix (forward+backward EMA, no cumulative offset)
    angle_abs = fix_pi_jumps_local(angle_abs, traj_dir_abs)

    # ── Recompute angle_img from fixed angle_abs
    for i in range(n):
        if not np.isnan(angle_abs[i]):
            angle_img_result[i] = angle_abs[i] + cum_rot.get(frames[i], 0.0)

    return {
        "angle_img": angle_img_result,
        "angle_abs": angle_abs,
        "delta_abs": np.full(n, np.nan),
        "ref_angle": 0.0,
        "obb_mask": obb_mask,
        "frames": frames,
    }


def compute_group_ref_and_deltas(angles: dict, n_ref_frames: int):
    all_first_frames = []
    for ti, ad in angles.items():
        valid = np.where(~np.isnan(ad["angle_abs"]))[0]
        if len(valid) > 0:
            all_first_frames.append(ad["frames"][valid[0]])
    if not all_first_frames:
        return 0.0

    global_start = min(all_first_frames)
    cutoff_frame = global_start + n_ref_frames

    ref_values = []
    for ti, ad in angles.items():
        for i, f in enumerate(ad["frames"]):
            if f >= cutoff_frame:
                break
            if not np.isnan(ad["angle_abs"][i]):
                ref_values.append(ad["angle_abs"][i])

    group_ref = circular_mean(ref_values) if ref_values else 0.0

    for ti, ad in angles.items():
        n = len(ad["angle_abs"])
        delta = np.full(n, np.nan)
        for i in range(n):
            if not np.isnan(ad["angle_abs"][i]):
                delta[i] = angle_diff(ad["angle_abs"][i], group_ref)
        ad["delta_abs"] = delta
        ad["ref_angle"] = group_ref

    return group_ref


# ── Cohesion (loaded from precomputed CSV) ───────────────────────────

def load_cohesion_csv(csv_path):
    """
    Load cohesion CSV (one row per frame).
    Columns expected: frame, T, shark_0, shark_1, …, cohesion_globale
    The shark_* column names must match the track file stems exactly.
    Returns dict: {frame: (T, {track_id_str: cohesion_i}, cohesion_global)}
    """
    df = pd.read_csv(csv_path)
    cohesion_lut = {}

    # Detect shark_* columns (these are the per-track cohesion values)
    coh_cols = [c for c in df.columns
                if c.startswith("shark_") and c != "cohesion_globale"]

    for _, row in df.iterrows():
        f = int(row["frame"])
        T_val = row.get("T", None)
        T_val = float(T_val) if pd.notna(T_val) else None

        # Keys are kept as column names ("shark_0", "shark_3", …)
        # so they match track["id"] directly.
        coh_per = {}
        for c in coh_cols:
            v = row.get(c, None)
            if pd.notna(v):
                coh_per[c] = float(v)      # key = "shark_0" etc.

        coh_global = row.get("cohesion_globale", None)
        coh_global = float(coh_global) if pd.notna(coh_global) else None

        cohesion_lut[f] = (T_val, coh_per, coh_global)

    return cohesion_lut


# ── Drawing helpers ──────────────────────────────────────────────────

PALETTE = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),
    (0, 128, 255), (128, 255, 0),
]


def get_color(i):
    return PALETTE[i % len(PALETTE)]


def draw_obb(fr, obb, col, th=2):
    cv2.polylines(fr, [np.array(obb, dtype=np.int32).reshape((-1, 1, 2))], True, col, th)


def draw_centroid(fr, c, col, r=4):
    cv2.circle(fr, (int(c[0]), int(c[1])), r, col, -1)


def draw_trail(fr, pts, col, th=2, mx=30):
    pts = pts[-mx:]
    for i in range(1, len(pts)):
        t = max(1, int(th * i / len(pts)))
        cv2.line(fr, (int(pts[i - 1][0]), int(pts[i - 1][1])),
                 (int(pts[i][0]), int(pts[i][1])), col, t)


def draw_arrow(fr, c, angle, col, length=50, th=2):
    cx, cy = int(c[0]), int(c[1])
    cv2.arrowedLine(fr, (cx, cy),
                    (cx + int(length * np.cos(angle)),
                     cy + int(length * np.sin(angle))),
                    col, th, tipLength=0.3)


# ── Panel ────────────────────────────────────────────────────────────

def draw_panel(ph, pw, infos, full_trails, cohesion_data):
    """
    Panel layout (top to bottom):
      - Trajectories     (~40%)
      - Cohesion         (~20%)
      - Compass          (~40%)
    """
    panel = np.zeros((ph, pw, 3), dtype=np.uint8)
    n = len(infos)

    traj_h = int(ph * 0.40)
    cohe_h = int(ph * 0.20)
    comp_h = ph - traj_h - cohe_h

    # ══════════════════════════════════════════════════════════════════
    # TOP: Stabilized trajectories
    # ══════════════════════════════════════════════════════════════════
    all_pts = []
    track_pts = {}
    active_tracks = {info["track_idx"] for info in infos}
    for t, pts in full_trails.items():
        if len(pts) >= 2:
            track_pts[t] = pts
            all_pts.extend(pts)

    if len(all_pts) >= 2:
        anp = np.array(all_pts)
        mn, mx_pt = anp.min(axis=0), anp.max(axis=0)
        span = mx_pt - mn
        m = 12
        draw_w, draw_h = pw - 2 * m, traj_h - 2 * m - 16
        sx = draw_w / max(span[0], 1e-3)
        sy = draw_h / max(span[1], 1e-3)
        sc = min(sx, sy)
        used_w, used_h = span[0] * sc, span[1] * sc
        off_x = m + (draw_w - used_w) / 2.0
        off_y = m + 16 + (draw_h - used_h) / 2.0

        def tp(pt):
            return (int(off_x + (pt[0] - mn[0]) * sc),
                    int(off_y + (pt[1] - mn[1]) * sc))

        cv2.putText(panel, "Stabilized trajectories", (4, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)

        for t, pts in track_pts.items():
            col = get_color(t)
            npts = len(pts)
            is_active = t in active_tracks
            for j in range(1, npts):
                alpha = 0.3 + 0.7 * (j / npts)
                if not is_active:
                    alpha *= 0.5
                c_fade = tuple(int(ch * alpha) for ch in col)
                cv2.line(panel, tp(pts[j - 1]), tp(pts[j]), c_fade, 1)
            if is_active:
                cv2.circle(panel, tp(pts[-1]), 6, col, -1)
            else:
                cv2.circle(panel, tp(pts[-1]), 5, col, 1)

    cv2.line(panel, (0, traj_h), (pw, traj_h), (50, 50, 50), 2)

    # ══════════════════════════════════════════════════════════════════
    # MIDDLE: Cohesion
    # ══════════════════════════════════════════════════════════════════
    y0 = traj_h
    T_val, coh_per, coh_global = cohesion_data
    # coh_per keys are render indices (int) after remapping in render()

    cv2.putText(panel, "Cohesion", (4, y0 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

    if coh_global is not None:
        gtxt = f"{coh_global:.2f}"
        cv2.putText(panel, gtxt, (pw // 2 - 30, y0 + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        if T_val is not None:
            cv2.putText(panel, f"T={T_val:.0f}px  n={len(coh_per)}",
                        (4, y0 + 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 80, 80), 1)

        bar_y = y0 + 56
        bar_h = 8
        max_bar_w = pw - 20
        max_display = max(coh_per.values()) * 1.2 if coh_per else 1.0
        max_display = max(max_display, 0.01)

        sorted_sharks = sorted(coh_per.items(), key=lambda x: x[0])
        for ti, ci in sorted_sharks:
            if bar_y + bar_h + 2 > y0 + cohe_h - 4:
                break
            col = get_color(ti)
            bw = int((ci / max_display) * max_bar_w)
            bw = max(bw, 2)
            cv2.rectangle(panel, (10, bar_y), (10 + bw, bar_y + bar_h), col, -1)
            cv2.putText(panel, f"{ci:.1f}", (14 + bw, bar_y + bar_h - 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, col, 1)
            bar_y += bar_h + 3
    else:
        cv2.putText(panel, "N/A (<2 sharks)", (10, y0 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)

    cv2.line(panel, (0, y0 + cohe_h), (pw, y0 + cohe_h), (50, 50, 50), 2)

    # ══════════════════════════════════════════════════════════════════
    # BOTTOM: Compass
    # ══════════════════════════════════════════════════════════════════
    comp_y0 = y0 + cohe_h
    cx = pw // 2
    cy = comp_y0 + comp_h // 2
    margin = 30
    r = min(pw // 2 - margin, comp_h // 2 - margin)
    r = max(r, 40)

    cv2.putText(panel, "Group orientation", (4, comp_y0 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

    cv2.circle(panel, (cx, cy), r, (60, 60, 60), 1, cv2.LINE_AA)
    for tick_angle, label in zip(
            [-np.pi / 2, 0, np.pi / 2, np.pi], ["0", "90", "180", "-90"]):
        t1 = (int(cx + (r - 6) * np.cos(tick_angle)),
              int(cy + (r - 6) * np.sin(tick_angle)))
        t2 = (int(cx + r * np.cos(tick_angle)),
              int(cy + r * np.sin(tick_angle)))
        cv2.line(panel, t1, t2, (80, 80, 80), 1, cv2.LINE_AA)
        lx = int(cx + (r + 16) * np.cos(tick_angle))
        ly = int(cy + (r + 16) * np.sin(tick_angle))
        cv2.putText(panel, label, (lx - 8, ly + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)

    cv2.circle(panel, (cx, cy - r), 3, (150, 150, 150), -1)

    valid_deltas = []
    overlay = panel.copy()
    for info in infos:
        delta = info.get("delta")
        if delta is None:
            continue
        valid_deltas.append(delta)
        col = info["color"]
        da = -np.pi / 2 + delta
        ax = int(cx + (r - 12) * np.cos(da))
        ay = int(cy + (r - 12) * np.sin(da))
        cv2.arrowedLine(overlay, (cx, cy), (ax, ay), col, 2, tipLength=0.25)

    cv2.addWeighted(overlay, 0.5, panel, 0.5, 0, panel)
    cv2.circle(panel, (cx, cy), 3, (180, 180, 180), -1, cv2.LINE_AA)

    if valid_deltas:
        mean_delta = circular_mean(valid_deltas)
        da_mean = -np.pi / 2 + mean_delta
        mx_a = int(cx + (r - 8) * np.cos(da_mean))
        my_a = int(cy + (r - 8) * np.sin(da_mean))
        cv2.arrowedLine(panel, (cx, cy), (mx_a, my_a),
                        (255, 255, 255), 3, tipLength=0.2, line_type=cv2.LINE_AA)

        deg_txt = f"mean {np.degrees(mean_delta):+.1f} deg"
        (tw, _), _ = cv2.getTextSize(deg_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(panel, deg_txt, (cx - tw // 2, cy + r + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    cnt_txt = f"n={len(valid_deltas)}/{n}"
    cv2.putText(panel, cnt_txt, (cx - 20, cy + r + 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (90, 90, 90), 1)

    return panel


# ── CSV export ───────────────────────────────────────────────────────

def export_csvs(tracks, angles, output_prefix):
    """
    Export angle CSVs.  Column names = track["id"] (e.g. "shark_3").
    """
    track_ids = [t["id"] for t in tracks]
    all_frames = set()
    for t in tracks:
        for d in t["detections"]:
            all_frames.add(d["frame"])

    frames_sorted = sorted(all_frames)

    # Build frame → detection-index lookup per track
    frame_to_di = {}
    for ti, t in enumerate(tracks):
        lut = {}
        for di, d in enumerate(t["detections"]):
            lut[d["frame"]] = di
        frame_to_di[ti] = lut

    for suffix, key in [("_angle_image.csv", "angle_img"),
                        ("_angle_absolute.csv", "delta_abs")]:
        path = output_prefix + suffix
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["frame"] + track_ids)
            for f in frames_sorted:
                row = [f]
                for ti in range(len(tracks)):
                    di = frame_to_di[ti].get(f)
                    if di is not None and not np.isnan(angles[ti][key][di]):
                        row.append(f"{angles[ti][key][di]:.6f}")
                    else:
                        row.append("")
                w.writerow(row)
        print(f"  CSV → {path}")


# ── Main ─────────────────────────────────────────────────────────────

def build_frame_index(tracks):
    idx = defaultdict(list)
    for ti, t in enumerate(tracks):
        for d in t["detections"]:
            idx[d["frame"]].append((ti, d))
    return idx


def render(
    video_path, track_paths, cmc_path, cohesion_csv_path, output_path,
    trail_length=40, smooth_window=7,
    thickness=2, panel_width=300,
    codec="MJPG", start_frame=0, end_frame=None,
    savgol_win=51, savgol_poly=2, n_ref_frames=30,
):
    raw_tracks = [load_track_json(p) for p in track_paths]
    cmc = load_cmc_json(cmc_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    end_frame = min(end_frame or total, total)

    print("Smoothing tracks...")
    tracks = [preprocess_track(rt, savgol_win, savgol_poly) for rt in raw_tracks]
    for t in tracks:
        print(f"  {t['id']}: {len(t['detections'])} dets")

    fidx = build_frame_index(tracks)
    cum_rot = build_cum_rot(cmc, start_frame, end_frame)

    # ── Cohesion ─────────────────────────────────────────────────────
    print("Loading cohesion CSV...")
    cohesion_lut_raw = load_cohesion_csv(cohesion_csv_path)
    print(f"  {len(cohesion_lut_raw)} frames loaded from {cohesion_csv_path}")

    # Map track["id"] (e.g. "shark_3") → render index (0, 1, 2…)
    id_to_idx = {t["id"]: ti for ti, t in enumerate(tracks)}

    # Remap cohesion CSV keys ("shark_3" etc.) to render indices
    cohesion_lut = {}
    for f, (T_val, coh_per, coh_global) in cohesion_lut_raw.items():
        remapped = {}
        for csv_id, v in coh_per.items():          # csv_id = "shark_3"
            if csv_id in id_to_idx:
                remapped[id_to_idx[csv_id]] = v     # render index → value
        cohesion_lut[f] = (T_val, remapped, coh_global)

    n_mapped = sum(1 for _, (_, rp, _) in cohesion_lut.items() if rp)
    print(f"  Cohesion remapped: {n_mapped}/{len(cohesion_lut)} frames have per-shark data")

    # ── Angles ───────────────────────────────────────────────────────
    print("Computing angles...")
    angles = {}
    for ti, t in enumerate(tracks):
        angles[ti] = compute_angles(
            t, cmc, cum_rot, trail_length, smooth_window,
            savgol_win, savgol_poly, n_ref_frames
        )

    group_ref = compute_group_ref_and_deltas(angles, n_ref_frames)
    print(f"  Group reference angle: {np.degrees(group_ref):+.1f} deg "
          f"(from first {n_ref_frames} frames across all tracks)")

    csv_prefix = os.path.splitext(output_path)[0]
    print("Exporting CSVs...")
    export_csvs(tracks, angles, csv_prefix)

    det_idx = {}
    for ti, t in enumerate(tracks):
        det_idx[ti] = {d["frame"]: i for i, d in enumerate(t["detections"])}

    # ── Render loop ──────────────────────────────────────────────────
    print("Rendering...")
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w + panel_width, h))

    dtrails = defaultdict(list)
    full_trails = defaultdict(list)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for f in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        fk = str(f)
        pinfo = []

        # CMC warp display trails (accumulated — needed for stabilized view)
        if fk in cmc:
            M = np.array(cmc[fk])
            for t in dtrails:
                for i in range(len(dtrails[t])):
                    pt = dtrails[t][i]
                    dtrails[t][i] = M @ np.array([pt[0], pt[1], 1.0])
            for t in full_trails:
                for i in range(len(full_trails[t])):
                    pt = full_trails[t][i]
                    full_trails[t][i] = M @ np.array([pt[0], pt[1], 1.0])

        cohesion_data = cohesion_lut.get(f, (None, {}, None))
        frame_dets = fidx.get(f, [])

        for ti, det in frame_dets:
            col = get_color(ti)
            trk = tracks[ti]
            obb = det.get("obb")
            has_obb = bool(obb and len(obb) == 4)
            c = np.array(det["centroid"], dtype=np.float64)
            di = det_idx[ti].get(f)
            ad = angles[ti]

            if has_obb:
                draw_obb(frame, obb, col, thickness)

            draw_centroid(frame, c, col, thickness + 2)

            dtrails[ti].append(c.copy())
            if len(dtrails[ti]) > trail_length:
                dtrails[ti] = dtrails[ti][-trail_length:]
            draw_trail(frame, dtrails[ti], col, thickness, trail_length)

            full_trails[ti].append(c.copy())

            if di is not None and not np.isnan(ad["angle_img"][di]):
                acol = tuple(c_ // 2 for c_ in col) if not has_obb else col
                draw_arrow(frame, c, ad["angle_img"][di], acol, 60, thickness)

            # ── Label = track file stem (e.g. "shark_3") ──
            label = trk["id"]

            info = {"label": label, "color": col, "track_idx": ti, "delta": None}
            if di is not None and not np.isnan(ad["delta_abs"][di]):
                info["delta"] = float(ad["delta_abs"][di])
            pinfo.append(info)

            # Draw label on frame
            cx_i, cy_i = int(c[0]), int(c[1])
            cv2.putText(frame, label, (cx_i + 8, cy_i - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

        panel = draw_panel(h, panel_width, pinfo, full_trails, cohesion_data)
        cv2.line(panel, (0, 0), (0, h), (60, 60, 60), 1)
        combined = np.hstack([frame, panel])
        cv2.putText(combined, f"Frame {f}",
                    (w + 8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 100), 1)
        writer.write(combined)

        if f % 500 == 0:
            print(f"  frame {f}/{end_frame} "
                  f"({100 * (f - start_frame) / max(1, end_frame - start_frame):.0f}%)")

    cap.release()
    writer.release()
    print(f"Done → {output_path}")


if __name__ == "__main__":
    pa = argparse.ArgumentParser(description="Render OBB + cohesion + compass")
    pa.add_argument("video")
    pa.add_argument("tracks", nargs="+")
    pa.add_argument("--cmc", required=True)
    pa.add_argument("--cohesion", required=True, help="Path to cohesion_per_frame.csv")
    pa.add_argument("-o", "--output", default="output_angle.mp4")
    pa.add_argument("--trail", type=int, default=120)
    pa.add_argument("--smooth", type=int, default=120)
    pa.add_argument("--thickness", type=int, default=3)
    pa.add_argument("--savgol-win", type=int, default=51)
    pa.add_argument("--savgol-poly", type=int, default=2)
    pa.add_argument("--n-ref", type=int, default=30)
    pa.add_argument("--panel-width", type=int, default=800)
    pa.add_argument("--start", type=int, default=0)
    pa.add_argument("--end", type=int, default=None)
    pa.add_argument("--codec", default="mp4v")
    a = pa.parse_args()

    render(
        a.video, a.tracks, a.cmc, a.cohesion, a.output,
        a.trail, a.smooth, a.thickness, a.panel_width,
        a.codec, a.start, a.end,
        a.savgol_win, a.savgol_poly, a.n_ref,
    )