"""
Convert track JSON(s) to CSV.

Output columns:
  frame, centroid_x, centroid_y, interpolated,
  obb_x0, obb_y0, obb_x1, obb_y1, obb_x2, obb_y2, obb_x3, obb_y3

All CSVs are padded from frame 0 to --end-frame (default 14579) with NA.

Usage:
  python json_track_to_csv.py track6.json track50.json -o output_dir/
  python json_track_to_csv.py *.json --end-frame 20000
"""

import os, json, argparse, csv


def load_json(p):
    with open(p) as f:
        return json.load(f)


def convert(json_path, output_dir, end_frame):
    track = load_json(json_path)

    ids = track.get("merged_track_ids", [0])
    label = "_".join(str(x) for x in ids)
    base = os.path.splitext(os.path.basename(json_path))[0]
    out_path = os.path.join(output_dir, f"{base}.csv")

    # Index detections by frame
    by_frame = {}
    for d in track["detections"]:
        by_frame[d["frame"]] = d

    header = [
        "frame", "centroid_x", "centroid_y", "interpolated",
        "obb_x0", "obb_y0", "obb_x1", "obb_y1",
        "obb_x2", "obb_y2", "obb_x3", "obb_y3",
    ]

    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)

        for f in range(0, end_frame + 1):
            det = by_frame.get(f)
            if det is None:
                w.writerow([f] + ["NA"] * 11)
            else:
                cx, cy = det["centroid"]
                interp = det.get("interpolated", "NA")
                obb = det.get("obb")
                if obb and len(obb) == 4:
                    obb_flat = [coord for pt in obb for coord in pt]
                else:
                    obb_flat = ["NA"] * 8
                w.writerow([f, cx, cy, interp] + obb_flat)

    print(f"  {out_path}  (track {label}, {len(by_frame)} dets, "
          f"frames 0..{end_frame})")


def main():
    pa = argparse.ArgumentParser(description="Convert track JSON to padded CSV")
    pa.add_argument("jsons", nargs="+", help="Track JSON files")
    pa.add_argument("-o", "--output-dir", default=".", help="Output directory")
    pa.add_argument("--end-frame", type=int, default=14579,
                    help="Last frame (inclusive), default 14579")
    a = pa.parse_args()

    os.makedirs(a.output_dir, exist_ok=True)
    for p in a.jsons:
        convert(p, a.output_dir, a.end_frame)
    print("Done.")


if __name__ == "__main__":
    main()
