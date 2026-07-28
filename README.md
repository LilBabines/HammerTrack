# HammerTrack 🦈

> ⚠️ **Warning** — For productivity reasons, most of the code in this repo is AI-generated. The code has not yet been fully reviewed, so please be careful when using or relying on it.

**[Sharks International 2026]** — Advancing remote monitoring of scalloped hammerhead sharks (*Sphyrna lewini*) using human-in-the-loop drone analytics.

## 🎯 Overview

HammerTrack supports ecological monitoring of scalloped hammerhead sharks by combining:
- **Detection** on aerial drone footage,
- **Segmentation** for precise outlines of individuals,
- **Multi-object tracking** across video frames,
- **Human-in-the-loop validation** through a graphical interface to correct and enrich detection and tracking.

## 🧰 Tech stack

- **Python 3.12**
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) — detection
- [SAM2](https://github.com/facebookresearch/sam2) — segmentation
- [BoxMOT](https://github.com/mikel-brostrom/boxmot) — tracking
- [PySide6](https://doc.qt.io/qtforpython-6/) — GUI
- PyTorch / TorchVision

## ⚙️ Installation

```bash
# Clone the repo
git clone https://github.com/LilBabines/HammerTrack.git
cd HammerTrack

# (Recommended) create a dedicated environment, then:
pip install torch torchvision  # Adapt for GPU support : https://pytorch.org/get-started/locally/ !!!
pip install ultralytics pyside6 boxmot
```

## 🚀 Usage

Run the main application (Detector Activate learning + Tracking):

```bash
python main.py
```


Additional helper and features extraction  scripts are available in `scripts/`.

## 📁 Project structure

```
HammerTrack/
├── src/              # Source code (models, pipeline, GUI)
├── scripts/          # Utility scripts + features extraction
└──  main.py           # Main entry point
```


## 🗂️ GUI workspace

When you create a new project from the GUI, HammerTrack generates a dedicated
folder under `projects/<project_name>/` with the following layout:

```
projects/<project_name>/
├── datasets/         # Annotated data, grows incrementally with active learning
├── finetune_runs/    # Ultralytics training logs (weights + metrics)
└── export/           # Exported tracks and rendered display videos
```

- **`datasets/`** — stores the annotations produced through the human-in-the-loop
  workflow. New samples are appended at each active learning iteration.
- **`finetune_runs/`** — used as the Ultralytics log directory; each fine-tuning
  run saves its model weights and training metrics here.
- **`export/`** — contains every exported tracking result and the display videos
  rendered from the GUI. One subfolder per clip:

```
export/<clip_id>/
├── per_frame/            # frame_XXXXXX.txt — one line per detection
├── per_track/            # track_XXXX.json — one file per raw tracker fragment
├── individuals/          # <name>.json — fragments MERGED per animal  ← pipeline input
├── individuals.json      # the fragment → individual mapping
├── cmc_transforms.json   # {"<frame>": affine 2x3} camera-motion compensation
└── postp_tracks/         # <name>.json — written by scripts/track_postprocess.py
```

**Track identity.** A track is identified by the *stem of its JSON filename*,
verbatim — `individuals/shark_3.json` is `shark_3`, `individuals/Bob.json` is
`Bob`. Every script uses that same string for CSV column names, video labels and
output filenames, and `scripts/track_postprocess.py` preserves it, so an
individual named in the GUI needs no renaming anywhere downstream. The rule
lives in one place: `scripts/track_io.py`.

## 📊 Features extraction

Once tracks have been exported, several scripts in `scripts/` compute
behavioural and group-level metrics from the trajectories. They all read
per-individual track JSONs and are meant to be run **after** tracking, on the
files stored in `projects/<project_name>/export/<clip_id>/`.

Run them in this order — `angle.py` needs the cohesion CSV, and
`merge_csv_per_track.py` needs both:

```
individuals/  ──► track_postprocess.py ──► postp_tracks/
                                              ├──► cohesion.py ──► cohesion.csv
                                              │                        │
                                              ├──► angle.py ◄──────────┘
                                              │       └──► *_angle_image.csv, *_angle_absolute.csv
                                              ├──► keypoints_TBF.py ──► keypoints/*.csv
                                              └──► merge_csv_per_track.py ──► merged/*.csv
```

- **`track_postprocess.py`** — *Numeric post-processing (run this first)*.

    Refines the per-individual trajectories exported by the GUI and re-emits
them under the same filename stem. **Merging is not done here**: the GUI owns
identity (it groups tracker fragments into individuals and resolves frame
collisions), so this script only applies the numeric passes, in order:

  * **outlier removal** — drop detections whose displacement from the previous
one exceeds `--max-jump-px` (or a z-score cutoff). Measured in the CMC
reference frame when `--cmc` is given, so a fast camera pan is not mistaken
for a teleporting shark,
  * **interpolation** — fill every missing frame between the first and last
detection, in the CMC reference frame, marking the result
`interpolated: true` with `confidence: 0.0` and no OBB,
  * **smoothing** — Savitzky-Golay (or moving average) on the centroids.

    Each pass can be disabled (`--outlier-method none`, `--interp-method none`,
`--smooth-method none`), and the passes actually applied are recorded in a
`postprocess` field in the output JSON. Identity keys (`uid`, `name`, `notes`,
`color`, `merged_track_ids`) pass through untouched.

    Required arguments: `--tracks`, `--output-dir`. Strongly recommended:
`--cmc`. Optional: `--video` + `--render-video` for a QC overlay.

  ```bash
  python scripts/track_postprocess.py \
      --tracks     projects/<project_name>/export/<clip_id>/individuals/ \
      --output-dir projects/<project_name>/export/<clip_id>/postp_tracks/ \
      --cmc        projects/<project_name>/export/<clip_id>/cmc_transforms.json
  ```

- **`keypoints_TBF.py`** — *Skeletal keypoints + Tail Beat Frequency*.

  > **Required** — install additional packages and download SAM2 weights:
  > ```bash
  > pip install sam2 scikit-image
  > wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
  > ```

  Re-segments each tracked individual frame by frame with **SAM2**, prompted
  by the bounding boxes/centroids stored in the post-processed tracks. From
  the resulting masks it extracts five keypoints — **head** (mid-cephalofoil),
  **center of mass**, two **articulations** (at 1/3 and 2/3 along the body
  axis), and **tail** — together with one articular angle (head–COM–tail) and
  several directional angles. The temporal evolution of these angles is the
  raw signal used downstream to estimate the **tail beat frequency**.

  Outputs (per run):
  - one CSV per track in `--csv-dir`, with columns
    `frame, head_x, head_y, com_x, com_y, art1_x, art1_y, art2_x, art2_y, tail_x, tail_y` + 8 angle columns,
  - a rendered skeleton display video at `--output-video`, with mask overlays,
    keypoints, and a side panel showing the angle time-series.

  Required arguments: `--video`, `--frames-dir`, `--tracks`, `--output-video`,
  `--csv-dir`. Useful optional flags: `--sam2-checkpoint`, `--sam2-config`,
  `--mask-threshold`, `--chunk-size`, `--max-tracks-per-batch`, `--no-graph`,
  `--no-display`. See `python scripts/keypoints_TBF.py --help` for the full list.

  Typical invocation, following the project layout above:

  ```bash
  python scripts/keypoints_TBF.py \
      --video        clips/selected/<clip_name>.mp4 \
      --frames-dir   clips/by_frames/<clip_name> \
      --tracks       projects/<project_name>/export/<clip_id>/postp_tracks/ \
      --output-video projects/<project_name>/export/<clip_id>/display_skeleton.mp4 \
      --csv-dir      projects/<project_name>/export/<clip_id>/keypoints/ \
      --sam2-checkpoint sam2.1_hiera_base_plus.pt \
      --sam2-config     configs/sam2.1/sam2.1_hiera_b+.yaml
  ```

  > Requires a CUDA-capable GPU for SAM2 inference (use `--device cpu` to
  > force CPU, much slower). `ffplay` must be installed if you want the live
  > preview window; otherwise pass `--no-display`.

- **`cohesion.py`** — *Per-frame group cohesion*.

  Quantifies how grouped or scattered the individuals are at each frame, from
  the post-processed tracks alone (no video / no GPU required). For every
  frame independently:
  - `T` = median of all bbox diagonals present in the frame (used as a scale
    that adapts to apparent shark size and camera altitude),
  - for each individual *i* present,
    `cohesion_i = quantile_q( ||c_i − c_j|| for j ≠ i ) / T`,
    where `c_*` are centroids and *q* is set with `--quantile` (default
    `0.25`, i.e. the lower quartile of pairwise distances — robust to
    isolated outliers within a school),
  - `cohesion_globale` = mean of `cohesion_i` over individuals present.

  Lower values mean tighter aggregation (in body-length-equivalent units),
  higher values mean a more dispersed school.

  Output: a single CSV with one row per frame and columns
  `frame, shark_<id>..., T, cohesion_globale`.

  Required arguments: `--tracks`, `--output-csv`. Optional: `--pattern`,
  `--quantile`.

  ```bash
  python scripts/cohesion.py \
      --tracks     projects/<project_name>/export/<clip_id>/postp_tracks/ \
      --output-csv projects/<project_name>/export/<clip_id>/cohesion.csv \
      --quantile   0.25
  ```

- **`merge_csv_per_track.py`** — *One consolidated CSV per individual*.

    Joins the post-processed track JSON with the angle CSVs and the cohesion CSV
into a single per-individual table: `frame, time_s, centroid_x, centroid_y,
interpolated, obb_x0..obb_y3, angle_image, angle_absolute, cohesion`. The join
key is the track identity, so nothing needs renaming.

    Note that `angle_absolute` holds `delta_abs` — the deviation from the group
reference heading in CMC-stabilised space, not an absolute compass bearing.

    Required arguments: `--tracks`, `--output-dir`. Optional:
`--angle-image-csv`, `--angle-absolute-csv`, `--cohesion-csv`, `--fps`.

  ```bash
  python scripts/merge_csv_per_track.py \
      --tracks              projects/<project_name>/export/<clip_id>/postp_tracks/ \
      --output-dir          projects/<project_name>/export/<clip_id>/merged/ \
      --angle-image-csv     projects/<project_name>/export/<clip_id>/display_angle_angle_image.csv \
      --angle-absolute-csv  projects/<project_name>/export/<clip_id>/display_angle_angle_absolute.csv \
      --cohesion-csv        projects/<project_name>/export/<clip_id>/cohesion.csv
  ```

- **`angle.py`** — *Per-individual angle, group orientation & overlay video*.

  Renders a full diagnostic video on top of the clip, plus per-track angle
  CSVs. For each tracked individual, the script computes the orientation of
  the body in image space from the **OBB** long axis, disambiguates the
  head/tail direction using the **smoothed trajectory** of the centroid
  trail, then stabilises the angle in a **CMC-warped** (camera-motion
  compensated) reference frame. A group-level reference angle is built from
  the first `--n-ref` frames; per-individual deviations from this reference
  are reported as `delta_abs`.

  The rendered video shows, side by side:
  - the clip with OBBs, centroid trails and per-individual heading arrows,
  - a panel with the stabilised trajectories, the per-frame cohesion bars
    (loaded from the cohesion CSV), and a compass with each individual's
    deviation from the group reference plus the mean group heading.

  Outputs (per run, written next to `--output-video`):
  - `<prefix>.mp4` — the rendered overlay video,
  - `<prefix>_angle_image.csv` — per-track heading in raw image space,
  - `<prefix>_angle_absolute.csv` — per-track deviation from the group
    reference, in CMC-stabilised space.

  Requires a **CMC JSON** (per-frame affine transforms) and the **cohesion
  CSV** produced by `scripts/cohesion.py`, so run that one first.

  Required arguments: `--video`, `--tracks`, `--cmc`, `--cohesion-csv`,
  `--output-video`. See `python scripts/angle.py --help` for the full list of
  rendering and smoothing options.

  ```bash
  python scripts/angle.py \
      --video        clips/selected/<clip_name>.mp4 \
      --tracks       projects/<project_name>/export/<clip_id>/postp_tracks/ \
      --cmc          projects/<project_name>/export/<clip_id>/cmc_transforms.json \
      --cohesion-csv projects/<project_name>/export/<clip_id>/cohesion.csv \
      --output-video projects/<project_name>/export/<clip_id>/display_angle.mp4
  ```


## 👥 Authors & Affiliations

HammerTrack is a joint effort between the **[University of Montpellier](https://www.umontpellier.fr/)** (France) and the **[University of Western Australia](https://www.uwa.edu.au/)** (Perth, Australia). The project brings together complementary expertise in computer vision, marine ecology, data acquisition  and shark conservation, and supports collaborative research on non-invasive monitoring of scalloped hammerhead populations using aerial drones.

<p align="center">
  <img src="assets/logos/logo_um.png" alt="Université de Montpellier" height="90" />
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="assets/logos/logo_uwa.svg" alt="The University of Western Australia" height="90" />
</p>

## 📚 Citation

This work is presented at **Sharks International 2026**, as part of research on non-invasive monitoring of hammerhead shark populations.

## 📄 License

Enjoy.