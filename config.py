# ============================================================
#  config.py — Single source of truth.
# ============================================================

# ── Camera / Video ──────────────────────────────────────────
CAMERA_INDEX      = 0          # Default webcam index
FRAME_WIDTH       = 1280       # Capture width  (0 = native)
FRAME_HEIGHT      = 720        # Capture height (0 = native)
PROCESS_WIDTH     = 480        # Resize-to before processing (speed)
TARGET_FPS        = 30

# ── Preprocessing ────────────────────────────────────────────
BLUR_KERNEL       = (5, 5)     # Gaussian blur kernel
THRESH_BLOCK      = 31         # Adaptive threshold block size (odd)
THRESH_C          = 8          # Adaptive threshold constant
MORPH_KERNEL      = (3, 3)     # Morphological open kernel (noise removal)
MORPH_ITERS       = 3

# ── Detection ────────────────────────────────────────────────
MIN_CONTOUR_AREA  = 1500        # px² — ignore noise blobs
MAX_CONTOUR_AREA  = 80000    # px² — ignore frame-filling objects
MIN_ASPECT_RATIO  = 1.5        # below → contour;  at/above → PCA
HARRIS_THRESHOLD  = 80         # Harris corner count to trigger ORB mode

# ── Method selection ─────────────────────────────────────────
# "auto" | "contour" | "pca" | "orb"
DEFAULT_METHOD    = "auto"

# ── Angle ────────────────────────────────────────────────────
ANGLE_EMA_ALPHA   = 0.30       # Exponential moving average weight (0=frozen, 1=raw)
ANGLE_DEAD_ZONE   = 2.0        # Degrees: skip update if delta < this
ANGLE_CONVENTION  = "0-360"    # "0-360" clockwise from +X  |  "-180-180"

# ── Tracker ──────────────────────────────────────────────────
IOU_THRESHOLD     = 0.15       # Min IoU to match track → detection
MAX_LOST_FRAMES   = 45         # Frames before a track is deleted
HISTORY_LEN       = 5          # Angle history kept per track

# ── Overlay ──────────────────────────────────────────────────
AXIS_ARROW_LEN    = 90         # Pixels — major axis arrow
ARC_RADIUS        = 48         # Pixels — angle arc
FONT              = 0          # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE        = 0.52
FONT_THICKNESS    = 1

# Colours BGR
COL_BOX           = (255, 190, 100)   # Bounding box
COL_ARROW         = (80,  220, 80 )   # Axis arrow
COL_ARC           = (40,  210, 230)   # Angle arc
COL_TEXT_BG       = (25,  20,  40 )   # Label background
COL_TEXT          = (210, 150, 255)   # Label text
COL_CENTER        = (220, 120, 255)   # Center dot fill
COL_CENTER_OUT    = (130, 60,  180)   # Center dot outline

# ── Output ───────────────────────────────────────────────────
OUTPUT_VIDEO_FPS  = 20
LOG_DELIMITER     = ","
