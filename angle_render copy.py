"""
Render OBB + compass on video.

All angle computation in IMAGE SPACE:
  1. CMC-warped trail → trajectory direction
  2. OBB long axis smoothed (×2 trick, π-periodic)
  3. Disambiguate axis with trail direction → directed angle
  4. Fix ±180° jumps from bad CMC frames
  5. Arrow on video = angle_img
  6. Compass = angle_img - cum_rot - ref
"""

import os
import csv
import cv2
import json
import argparse
import numpy as np
from collections import defaultdict
from scipy.signal import savgol_filter


# ── JSON ─────────────────────────────────────────────────────────────

def load_json(p):
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


# ── Fix ±180° jumps ─────────────────────────────────────────────────

def fix_pi_jumps(angle_abs, threshold_deg=135):
    """
    Remove spurious ±180° flips in angle_abs caused by bad CMC frames.

    A real shark doesn't turn >threshold_deg in a single frame, so any
    consecutive jump above that threshold is assumed to be a π-ambiguity
    error and is corrected by adding/subtracting π.

    Operates on the full array (with NaNs); only consecutive valid
    entries are compared.
    """
    valid_idx = np.where(~np.isnan(angle_abs))[0]
    if len(valid_idx) < 2:
        return angle_abs

    fixed = angle_abs.copy()
    threshold = np.radians(threshold_deg)
    offset = 0.0

    for k in range(1, len(valid_idx)):
        i_prev = valid_idx[k - 1]
        i_curr = valid_idx[k]
        diff = angle_diff(fixed[i_curr] + offset, fixed[i_prev])

        if abs(diff) > threshold:
            # Jump is near ±π → false flip, correct it
            if diff > 0:
                offset -= np.pi
            else:
                offset += np.pi

        fixed[i_curr] += offset

    return fixed


# ── Preprocess: smooth only ──────────────────────────────────────────

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

    # angle_abs = directed angle in stabilized (absolute) space
    angle_abs = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(angle_img_result[i]):
            angle_abs[i] = angle_img_result[i] - cum_rot.get(frames[i], 0.0)

    # Fix ±180° jumps from bad CMC frames
    angle_abs = fix_pi_jumps(angle_abs)

    # Also fix angle_img consistently: recompute from fixed angle_abs
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
    """
    Compute a single group reference angle from the first n_ref_frames
    of the video (across ALL tracks), then set delta_abs for each track
    relative to that group reference.
    """
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

    if not ref_values:
        group_ref = 0.0
    else:
        rv = np.array(ref_values)
        group_ref = np.arctan2(np.mean(np.sin(rv)), np.mean(np.cos(rv)))

    for ti, ad in angles.items():
        n = len(ad["angle_abs"])
        delta = np.full(n, np.nan)
        for i in range(n):
            if not np.isnan(ad["angle_abs"][i]):
                delta[i] = (ad["angle_abs"][i] - group_ref + np.pi) % (2 * np.pi) - np.pi
        ad["delta_abs"] = delta
        ad["ref_angle"] = group_ref

    return group_ref


# ── Drawing ──────────────────────────────────────────────────────────

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

def draw_panel(ph, pw, infos, full_trails, display_trails, tl=40):
    panel = np.zeros((ph, pw, 3), dtype=np.uint8)
    n = len(infos)
    if n == 0:
        return panel

    th = ph // 2
    bh = ph - th

    # ── Top section: full trajectory history (orthonormal) ───────────
    all_pts = []
    track_pts = {}
    active_tracks = {info["track_idx"] for info in infos}
    for t, pts in full_trails.items():
        if len(pts) >= 2:
            track_pts[t] = pts
            all_pts.extend(pts)

    if len(all_pts) >= 2:
        anp = np.array(all_pts)
        mn = anp.min(axis=0)
        mx = anp.max(axis=0)
        span = mx - mn

        m = 12
        draw_w = pw - 2 * m
        draw_h = th - 2 * m - 16

        sx = draw_w / max(span[0], 1e-3)
        sy = draw_h / max(span[1], 1e-3)
        sc = min(sx, sy)

        used_w = span[0] * sc
        used_h = span[1] * sc

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

    cv2.line(panel, (0, th), (pw, th), (50, 50, 50), 2)

    # ── Bottom section: single group compass ─────────────────────────
    cx = pw // 2
    cy = th + bh // 2
    margin = 30
    r = min(pw // 2 - margin, bh // 2 - margin)
    r = max(r, 40)

    cv2.putText(panel, "Group orientation", (4, th + 16),
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
        sin_sum = sum(np.sin(d) for d in valid_deltas)
        cos_sum = sum(np.cos(d) for d in valid_deltas)
        mean_delta = np.arctan2(sin_sum, cos_sum)

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
    track_labels = []
    all_frames = set()
    for ti, t in enumerate(tracks):
        ids = t.get("merged_track_ids", [ti])
        track_labels.append("_".join(str(x) for x in ids))
        for d in t["detections"]:
            all_frames.add(d["frame"])

    frames_sorted = sorted(all_frames)

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
            w.writerow(["frame"] + [f"track_{lb}" for lb in track_labels])
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
    video_path, track_paths, cmc_path, output_path,
    trail_length=40, smooth_window=7,
    thickness=2, panel_width=300,
    codec="mp4v", start_frame=0, end_frame=None,
    savgol_win=51, savgol_poly=2, n_ref_frames=30,
):
    raw_tracks = [load_json(p) for p in track_paths]
    cmc = load_json(cmc_path)

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
    for i, t in enumerate(tracks):
        print(f"  Track {t.get('merged_track_ids', [i])}: {len(t['detections'])} dets")

    fidx = build_frame_index(tracks)
    cum_rot = build_cum_rot(cmc, start_frame, end_frame)

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

        for ti, det in fidx.get(f, []):
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

            ids = trk.get("merged_track_ids", [])
            label = f"IDs {ids}" if len(ids) > 1 else f"ID {ids[0]}"

            info = {"label": label, "color": col, "track_idx": ti, "delta": None}
            if di is not None and not np.isnan(ad["delta_abs"][di]):
                info["delta"] = float(ad["delta_abs"][di])
            pinfo.append(info)

        panel = draw_panel(h, panel_width, pinfo, full_trails, dtrails, trail_length)
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
    pa = argparse.ArgumentParser(description="Render OBB + absolute orientation compass")
    pa.add_argument("video")
    pa.add_argument("tracks", nargs="+")
    pa.add_argument("--cmc", required=True)
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
        a.video, a.tracks, a.cmc, a.output,
        a.trail, a.smooth, a.thickness, a.panel_width,
        a.codec, a.start, a.end,
        a.savgol_win, a.savgol_poly, a.n_ref,
    )