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
pip install torch torchvision  # Adapt for GPU support !!!
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
  rendered from the GUI.



## 📚 Citation

This work is presented at **Sharks International 2026**, as part of research on non-invasive monitoring of hammerhead shark populations.

## 📄 License

Enjoy.
