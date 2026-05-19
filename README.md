# DefectVision

**Real-time print defect inspection for cylindrical products — runs on a standard PC or Raspberry Pi CM5.**

DefectVision compares live camera frames against clean reference images to detect print quality issues — missing ink, smears, and surface debris — on a printing roller. It uses a text-centric comparison pipeline that is robust to minor positional shifts and illumination changes, and is designed for continuous deployment on production lines.

---

## Features

- **Text-structure comparison** — Binarizes and compares ink patterns rather than raw pixels, making detection invariant to small position and lighting variations
- **Three defect signals** — Recall (missing ink), Purity (extra ink / smear), and NCC (shape distortion), combined into a single composite score
- **Debris hard-override** — Flags foreign particles regardless of the overall score
- **Position locking** — NCC template matching tracks the print region as cylinders move past the camera
- **Temporal filtering** — Rolling-window consensus eliminates single-frame false positives from sensor noise
- **Multi-angle references** — Up to 7 reference frames captured at distinct rotation angles; the closest match is selected per frame
- **Background normalization** — Divides out slow-varying illumination gradients before CLAHE to handle bottle surface texture and uneven lighting
- **Multi-platform camera support** — OpenCV (DSHOW / V4L2 / AUTO) and Picamera2 (IMX296 global-shutter) via a unified interface
- **Interactive ROI selector** — Tkinter GUI with live camera feed and drag-to-select
- **CSV logging and snapshots** — Per-frame inspection log with optional defect snapshots and end-of-session statistics
- **Reference persistence** — References saved to disk and reloaded automatically on next run

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         main.py                             │
│   Camera init → ROI selection → Reference capture → Loop   │
└──────────────────────────┬──────────────────────────────────┘
                           │ frame
          ┌────────────────▼─────────────────┐
          │         position_lock.py          │
          │   NCC template matching + blur    │
          │   gate → locked ROI crop          │
          └────────────────┬─────────────────┘
                           │ cropped frame
          ┌────────────────▼─────────────────┐
          │          preprocessor.py          │
          │   Grayscale → BG normalize →      │
          │   Denoise → CLAHE                 │
          └────────────────┬─────────────────┘
                           │ normalized frame
          ┌────────────────▼─────────────────┐
          │           aligner.py             │
          │   ECC rigid alignment (optional) │
          └────────────────┬─────────────────┘
                           │ aligned frame
          ┌────────────────▼─────────────────┐
          │           inspector.py           │
          │   Binarize → Recall / Purity /   │
          │   NCC → Composite score + debris │
          └────────────────┬─────────────────┘
                           │ score + verdict
          ┌────────────────▼─────────────────┐
          │         temporal_filter.py        │
          │   Rolling-window consensus        │
          └────────┬─────────────────┬────────┘
                   │                 │
          ┌────────▼──────┐ ┌────────▼──────┐
          │ visualizer.py │ │   logger.py   │
          │ Dual-window   │ │ CSV + snaps   │
          │ overlay + HUD │ └───────────────┘
          └───────────────┘
```

---

## Defect Detection Pipeline

### Preprocessing
1. Convert frame to grayscale
2. Estimate slow-varying illumination via large-sigma Gaussian blur and divide it out (removes bottle surface texture and gradient lighting)
3. Optional Gaussian denoising
4. CLAHE for local contrast enhancement

### Text Binarization
1. Detect ink polarity (dark-on-light vs light-on-dark) via Blackhat / Tophat morphology
2. Apply Otsu thresholding on the morphological feature map
3. Filter connected components below the minimum area threshold (removes noise)
4. Flag components above the debris threshold as hard defects regardless of score

### Scoring
| Signal | Formula | Catches |
|--------|---------|---------|
| **Recall** | `(ref_mask ∩ live_mask) / ref_mask` | Missing ink, dropped characters |
| **Purity** | `1 − (extra_ink / ref_mask)` | Smears, bleed, foreign ink |
| **NCC** | Normalized cross-correlation of binary masks | Shape distortion, partial loss |

Composite score = `w_recall × Recall + w_purity × Purity + w_ncc × NCC`

All weights and the defect threshold are configurable in [config.py](DefectVision/config.py).

---

## Requirements

### Hardware
- **Development / Windows** — Any USB or MIPI camera supported by OpenCV
- **Edge deployment** — Raspberry Pi Compute Module 5 with IMX296 global-shutter camera module

### Software
- Python 3.9+
- Tkinter (usually bundled with Python; `sudo apt install python3-tk` on Pi)
- Pillow with ImageTk support

### Python Packages

**PC / Development**
```
opencv-python>=4.8.0
scikit-image>=0.21.0
numpy>=1.24.0
```

**Raspberry Pi CM5**
```bash
# System packages
sudo apt install -y python3-opencv python3-picamera2 python3-tk

# PyPI packages
pip install scikit-image numpy
```

Install on PC:
```bash
pip install -r DefectVision/requirements.txt
```

---

## Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/your-org/DefectVisionV4.git
cd DefectVisionV4/DefectVision
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run with GUI ROI selection
```bash
python main.py
```

On first launch:
1. A live camera feed opens — drag to draw the inspection ROI, then press **Enter**
2. Rotate a clean cylinder in front of the camera; the system auto-samples reference frames
3. Inspection begins automatically when enough references are collected

### 4. Run with a fixed ROI (no GUI)
```bash
python main.py --roi 100 50 400 200
```

### 5. Extract references from a pre-recorded video
```bash
python main.py --video calibration.mp4
```

---

## Keyboard Controls

| Key | Action |
|-----|--------|
| `Space` | Pause / resume inspection |
| `C` | Recalibrate — rotate a clean cylinder to re-capture references |
| `R` | Manual single-frame reference recapture |
| `V` | Load references from a video file |
| `S` | Save snapshot of current ROI to disk |
| `Q` | Quit |

---

## Configuration

All tunable parameters live in [DefectVision/config.py](DefectVision/config.py).

| Section | Key Parameters |
|---------|---------------|
| **Camera** | `CAMERA_BACKEND`, `CAMERA_INDEX`, resolution, FPS, warmup frames |
| **Preprocessing** | CLAHE clip limit, denoise kernel size, background normalization sigma |
| **Alignment** | ECC max iterations, convergence epsilon, max shift / rotation tolerances |
| **Text binarization** | Blackhat kernel size, min component area, tolerance dilation per signal |
| **Defect scoring** | `RECALL_WEIGHT`, `PURITY_WEIGHT`, `NCC_WEIGHT`, `DEFECT_THRESHOLD`, `DEBRIS_MIN_AREA` |
| **Position lock** | NCC match threshold, search margin, blur gate threshold |
| **Multi-reference** | `MAX_REFERENCES` (default 7), NCC distinctness threshold |
| **Temporal filter** | Window size (default 6 frames), defect ratio threshold (default 0.50) |
| **Logging** | Log directory, snapshot-on-defect toggle |

---

## Project Structure

```
DefectVisionV4/
├── DefectVision/
│   ├── main.py               # Application entry point and main inspection loop
│   ├── config.py             # All tunable parameters
│   ├── capture_cm5.py        # CM5 CRAFT-based text region detection helper
│   ├── requirements.txt      # PC dependencies
│   ├── requirements_cm5.txt  # CM5 deployment dependencies
│   ├── core/
│   │   ├── inspector.py      # Defect scoring engine (Recall / Purity / NCC)
│   │   ├── preprocessor.py   # Image normalization pipeline
│   │   ├── camera.py         # Multi-backend camera abstraction
│   │   ├── aligner.py        # ECC / phase-correlation alignment
│   │   ├── position_lock.py  # NCC template matching and blur gate
│   │   ├── roi_selector.py   # Interactive Tkinter ROI selector
│   │   ├── temporal_filter.py# Rolling-window consensus filter
│   │   └── visualizer.py     # Dual-window overlay and panel grid
│   ├── utils/
│   │   └── logger.py         # CSV logging and session statistics
│   ├── references/           # Persisted reference images (auto-created)
│   └── logs/                 # Inspection logs and defect snapshots (auto-created)
└── README.md
```

---

## Visualization

The system renders two windows at runtime:

**Main window** — Live camera feed with ROI border, status badge (PASS / DEFECT / WARMING / SEARCHING), and defect score overlay.

**Panel grid** — Side-by-side comparison:

| Reference | Live Frame | Defect Heatmap | Pixel Diff |
|-----------|-----------|---------------|------------|

The heatmap uses a JET colormap blended over the live frame to show where deviations are concentrated.

---

## Logging

Each session writes a CSV log to `logs/` with columns:

```
timestamp, frame_id, defect_score, recall, purity, ncc, verdict, fps
```

End-of-session statistics (total frames, defect count, defect rate %, average FPS) are printed to the console and appended to the log.

When `SNAPSHOT_ON_DEFECT = True` in config, a timestamped PNG of the ROI is saved alongside the CSV on every defect event.

---

## Deployment on Raspberry Pi CM5

1. Flash Raspberry Pi OS Bookworm (64-bit) on the CM5
2. Install system packages:
   ```bash
   sudo apt update
   sudo apt install -y python3-opencv python3-picamera2 python3-tk python3-pip
   ```
3. Install Python packages:
   ```bash
   pip install scikit-image numpy
   ```
4. Set `CAMERA_BACKEND = "picamera2"` in [config.py](DefectVision/config.py)
5. Run:
   ```bash
   python main.py
   ```

For the optional CRAFT-based text detection helper (`capture_cm5.py`), also install PyTorch (CPU-only) and the CRAFT source files — see the header of [capture_cm5.py](DefectVision/capture_cm5.py) for instructions.

---

## Contributing

1. Fork the repository and create a feature branch
2. Make your changes with clear, focused commits
3. Ensure the inspection pipeline runs end-to-end (`python main.py`) without regressions
4. Open a pull request with a description of what changed and why

---

