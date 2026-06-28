# Object Orientation Detector

Computer vision system that detects objects and measures their orientation angle in real time — from webcam, video file, or static image.

---

## Quick start

```bash
pip install -r requirements.txt

# Webcam
python main.py

# Video file
python main.py --source video.mp4

# Single image
python main.py --source image.jpg

# Force a specific method
python main.py --method pca

# Save output + log angles
python main.py --save output.mp4 --log angles.csv

# Headless (no window, e.g. on a server)
python main.py --no-display --log angles.csv
```

---

## Controls (live window)

| Key | Action |
|-----|--------|
| Q / ESC | Quit |
| R | Reset tracker IDs |
| S | Save screenshot as `frame.png` |
| 1 | Switch to **auto** method |
| 2 | Switch to **contour** method |
| 3 | Switch to **pca** method |

---

## How it works

### Detection methods

| Method | Best for | Algorithm |
|--------|----------|-----------|
| `contour` | Blobs, rectangles, convex shapes | `cv2.minAreaRect` → angle correction |
| `pca` | Thin / elongated shapes (rods, pencils) | PCA on contour pixels → first eigenvector |
| `orb` | Textured objects (circuit boards, logos) | ORB keypoint angles → circular mean |
| `auto` | Everything | Per-object: aspect ratio + Harris corners decide |

### Angle convention

All methods normalize to **0–360° clockwise from the +X axis** (horizontal right). This is consistent regardless of which method is used.

### Tracker

IoU + Hungarian algorithm assigns persistent IDs across frames. Each track maintains:
- **EMA smoothing** (α=0.30) — damps noise without lag
- **Dead-zone** (±2°) — prevents label flicker on stable objects

---

## Architecture

```
orientation_detector/
├── main.py              Entry point, CLI, main loop
├── config.py            All thresholds and constants
├── core/
│   ├── capture.py       Threaded VideoCapture (non-blocking)
│   ├── preprocessor.py  Resize → gray → blur → adaptive threshold → morph
│   ├── detector.py      Adaptive method selector
│   └── angle_utils.py   normalize, EMA, dead-zone
├── methods/
│   ├── contour.py       minAreaRect detection
│   ├── pca.py           PCA detection
│   └── keypoint.py      ORB detection with PCA fallback
├── tracking/
│   └── tracker.py       IoU + Hungarian multi-object tracker
├── overlay/
│   └── renderer.py      All cv2 drawing (arrows, arcs, labels, HUD)
└── utils/
    ├── fps.py            Rolling-window FPS counter
    └── logger.py         CSV angle logger
```

---

## Tuning

Edit `config.py` — all parameters live there. Key ones:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `MIN_CONTOUR_AREA` | 800 | Raise to ignore small noise |
| `ANGLE_EMA_ALPHA` | 0.30 | Lower = smoother, higher = more responsive |
| `ANGLE_DEAD_ZONE` | 2.0° | Raise to reduce flicker on jittery inputs |
| `IOU_THRESHOLD` | 0.25 | Lower = more lenient ID matching |
| `MAX_LOST_FRAMES` | 30 | Frames before a lost track is deleted |
| `MIN_ASPECT_RATIO` | 1.5 | Threshold to switch contour → PCA |
| `HARRIS_THRESHOLD` | 80 | Corner count to trigger ORB mode |

---

## CSV log format

```
timestamp, frame, obj_id, cx, cy, angle_raw, angle_smooth, method, area
1718000000.1234, 42, 3, 320, 240, 35.7, 34.9, contour, 4820
```

---

## Requirements

- Python 3.10+
- `opencv-python >= 4.9`
- `numpy >= 1.26`
- `scipy >= 1.13`
- No GPU required — runs on a laptop or Raspberry Pi 4
