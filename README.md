# Object Orientation Detector

**Real-time computer vision system that detects objects and measures their orientation angle**  
Works with webcam, video files, and static images. No GPU required.
---

## Overview

This system detects one or more objects in a frame and computes their **orientation angle (0–360°)** relative to the horizontal axis. It runs in real time at 25–30 FPS on a standard laptop CPU, requires no GPU, and works on any platform that supports Python and OpenCV.

Key capabilities:
- Real-time webcam detection with live angle overlay
- Multi-object tracking with persistent IDs across frames
- Three detection algorithms that are selected automatically per object
- Angle smoothing to eliminate jitter without introducing lag
- Video output and CSV logging for data collection

---

## How It Works

### Pipeline

```
Input (webcam / video / image)
        │
        ▼
Frame Preprocessor
  Resize → Grayscale → Gaussian Blur → Adaptive Threshold → Morphological Open
        │
        ▼
Detection Engine  (per contour)
  ┌─────────────────────────────────────────────────┐
  │  Auto-selector checks aspect ratio + texture    │
  │                                                 │
  │  Blob / convex  →  Contour (minAreaRect)        │
  │  Elongated      →  PCA (eigenvector angle)      │
  │  Textured       →  ORB (keypoint circular mean) │
  └─────────────────────────────────────────────────┘
        │
        ▼
Angle Calculator
  Normalize all methods → [0°, 360°) from +X axis
  Apply EMA smoothing (α = 0.30)
  Apply dead-zone filter (±2°)
        │
        ▼
Multi-Object Tracker
  IoU cost matrix + Hungarian assignment
  Persistent IDs · angle history · lost-track pruning
        │
        ▼
Overlay Renderer
  Bounding box · axis arrow · angle arc · label · HUD
        │
        ▼
Output  (live window / saved video / CSV log)
```

### Detection Methods

| Method | Trigger condition | Algorithm | Best for |
|--------|-------------------|-----------|----------|
| `contour` | Aspect ratio < 1.5 and low texture | `cv2.minAreaRect` → angle correction | Blobs, boxes, coins |
| `pca` | Aspect ratio ≥ 1.5 | PCA on contour pixels → first eigenvector | Rods, pencils, elongated parts |
| `orb` | Harris corner count > threshold | ORB keypoint angles → circular mean | Textured objects, PCBs, logos |
| `auto` | Always (default) | Selects from above per object per frame | General use |

### Angle Convention

All three methods normalize their output to the **same convention**:

> **0° to 360°, clockwise, measured from the +X axis (horizontal right)**

This means a flat horizontal object = 0°, rotated 45° clockwise = 45°, vertical = 90°, and so on — regardless of which detection method was used.

### Tracking

Each detected object receives a persistent integer ID that survives across frames using **IoU (Intersection-over-Union) + Hungarian algorithm** matching. This correctly handles cases where objects cross paths or temporarily leave the frame.

Per-track angle stability:
- **Exponential Moving Average (EMA, α=0.30)** — smooths noisy angle readings without visible lag
- **Dead-zone filter (±2°)** — suppresses label flicker when an object is stationary

---

## Project Structure

```
orientation_detector/
├── main.py                  Entry point · CLI argument parser · main loop
├── config.py                All thresholds and constants (single source of truth)
│
├── core/
│   ├── capture.py           Threaded VideoCapture — decouples I/O from processing
│   ├── preprocessor.py      Resize → gray → blur → adaptive threshold → morph open
│   ├── detector.py          Adaptive method selector (auto / contour / pca / orb)
│   └── angle_utils.py       normalize_angle · EMA · dead-zone · minAreaRect fix
│
├── methods/
│   ├── contour.py           minAreaRect-based detection
│   ├── pca.py               PCA eigenvector-based detection
│   └── keypoint.py          ORB keypoint detection with PCA fallback
│
├── tracking/
│   └── tracker.py           IoU cost matrix · Hungarian assignment · Track class
│
├── overlay/
│   └── renderer.py          All cv2 drawing — boxes, arrows, arcs, labels, HUD
│
└── utils/
    ├── fps.py               Rolling-window FPS counter
    └── logger.py            CSV angle logger
```

---

## Installation

**Requirements:** Python 3.10 or later.

```bash
# Clone or unzip the project, then:
cd orientation_detector
pip install -r requirements.txt
```

`requirements.txt` installs:
- `opencv-python >= 4.9`
- `numpy >= 1.26`
- `scipy >= 1.13`

No GPU, no CUDA, no deep learning framework required.

---

## Usage

```bash
# Default — open webcam (index 0)
python main.py

# Specific camera
python main.py --source 1

# Video file
python main.py --source path/to/video.mp4

# Static image
python main.py --source path/to/image.jpg

# Force a specific detection method
python main.py --method contour
python main.py --method pca
python main.py --method orb

# Save the annotated output to a video file
python main.py --save output.mp4

# Log all angle data to CSV
python main.py --log angles.csv

# Headless mode — no window (useful for servers or automation)
python main.py --no-display --log angles.csv

# Combine options
python main.py --source video.mp4 --method auto --save result.mp4 --log data.csv
```

### CLI Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--source` | `0` | Webcam index, video file path, or image path |
| `--method` | `auto` | Detection method: `auto`, `contour`, `pca`, `orb` |
| `--min-area` | `3000` | Minimum contour area (px²) — raise to filter noise |
| `--save` | — | Output video file path (e.g. `output.mp4`) |
| `--log` | — | CSV log file path (e.g. `angles.csv`) |
| `--no-display` | off | Headless mode — suppress the display window |

### Keyboard Controls (live window)

| Key | Action |
|-----|--------|
| `Q` or `ESC` | Quit |
| `R` | Reset all tracker IDs |
| `S` | Save screenshot as `frame.png` |
| `1` | Switch to **auto** method |
| `2` | Switch to **contour** method |
| `3` | Switch to **pca** method |

---

## Overlay Elements

Each detected object shows:

- **Coloured bounding box** — colour indicates which method was used (blue = contour, teal = PCA, purple = ORB)
- **Green axis arrow** — points along the object's major axis at the detected angle
- **Cyan arc** — sweeps from the reference axis (0°) to the current angle
- **Angle readout** — inline on the arc and in the label (`ID:3  47.2 deg`)
- **Top-right HUD** — FPS, object count, active method, frame number
- **Bottom-left legend** — colour key for detection methods
- **Dashed horizontal line** — reference axis (0°)

---

## Tuning Guide

All parameters are in `config.py`. No magic numbers exist anywhere else in the codebase.

### Reducing false detections

```python
MIN_CONTOUR_AREA  = 3000    # Increase to ignore smaller regions
MAX_CONTOUR_AREA  = 80000   # Decrease to ignore large background regions
THRESH_BLOCK      = 31      # Larger block = less sensitivity to local texture
THRESH_C          = 8       # Higher constant = cleaner binary mask
MORPH_ITERS       = 3       # More iterations = more aggressive noise removal
```

### Improving angle stability

```python
ANGLE_EMA_ALPHA  = 0.20    # Lower = smoother (0.1 = very smooth, 0.5 = responsive)
ANGLE_DEAD_ZONE  = 3.0     # Degrees: increase to reduce label flicker
```

### Improving tracking stability

```python
IOU_THRESHOLD    = 0.15    # Lower = more lenient ID matching across frames
MAX_LOST_FRAMES  = 45      # Higher = IDs survive longer occlusions
```

### Improving performance

```python
PROCESS_WIDTH    = 480     # Smaller = faster (at some accuracy cost)
```

### Full parameter reference

| Parameter | Default | Effect |
|-----------|---------|--------|
| `PROCESS_WIDTH` | `640` | Frame width for processing (resize before detection) |
| `MIN_CONTOUR_AREA` | `3000` | Contours smaller than this (px²) are ignored |
| `MAX_CONTOUR_AREA` | `80000` | Contours larger than this are ignored |
| `BLUR_KERNEL` | `(5,5)` | Gaussian blur kernel size |
| `THRESH_BLOCK` | `31` | Adaptive threshold neighbourhood size (must be odd) |
| `THRESH_C` | `8` | Adaptive threshold constant |
| `MORPH_ITERS` | `3` | Morphological open iterations |
| `MIN_ASPECT_RATIO` | `1.5` | Aspect ratio above which PCA is used instead of contour |
| `HARRIS_THRESHOLD` | `80` | Harris corner count above which ORB is used |
| `ANGLE_EMA_ALPHA` | `0.30` | EMA weight (0 = frozen, 1 = raw) |
| `ANGLE_DEAD_ZONE` | `2.0` | Degrees below which display angle is not updated |
| `IOU_THRESHOLD` | `0.15` | Minimum IoU to match a detection to an existing track |
| `MAX_LOST_FRAMES` | `45` | Frames before a lost track is deleted |

---

## CSV Log Format

When `--log` is specified, one row is written per object per frame:

```
timestamp,frame,obj_id,cx,cy,angle_raw,angle_smooth,method,area
1718000000.1234,42,3,320,240,35.70,34.90,contour,4820
1718000000.1234,42,5,180,310,91.20,90.10,pca,6200
```

| Column | Description |
|--------|-------------|
| `timestamp` | Unix time (seconds, 4 decimal places) |
| `frame` | Frame number since start |
| `obj_id` | Persistent track ID |
| `cx`, `cy` | Object centroid in pixels |
| `angle_raw` | Raw angle before smoothing |
| `angle_smooth` | Display angle after EMA + dead-zone |
| `method` | Detection method used for this object |
| `area` | Contour area in pixels² |

---

## Performance

Tested on a mid-range laptop (Intel i5, no GPU):

| Scene | Objects | FPS |
|-------|---------|-----|
| Single object, clean background | 1 | ~30 |
| Multiple objects, method=auto | 4–6 | 25–30 |
| Busy background, method=contour | 6+ | 20–25 |

The threaded capture architecture (`core/capture.py`) ensures the processing loop never blocks on camera I/O, which is the primary bottleneck in naive implementations.

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `opencv-python` | >= 4.9 | Image capture, processing, drawing |
| `numpy` | >= 1.26 | Array math |
| `scipy` | >= 1.13 | Hungarian algorithm (`linear_sum_assignment`) |

---
