# ============================================================
#  overlay/renderer.py — All cv2 drawing in one place.
# ============================================================

import cv2
import numpy as np
import math
import config as C
from tracking.tracker import Track

_METHOD_COLORS = {
    "contour"      : C.COL_BOX,
    "pca"          : (100, 230, 200),
    "orb"          : (200, 130, 255),
    "pca(fallback)": (180, 180, 80),
}


def _arrow(frame, cx, cy, angle_deg, length, color, thickness=2):
    """Draw an oriented arrow from centroid along angle_deg."""
    rad = math.radians(angle_deg)
    ex  = int(cx + length * math.cos(rad))
    ey  = int(cy + length * math.sin(rad))
    cv2.arrowedLine(frame, (cx, cy), (ex, ey),
                    color, thickness, tipLength=0.28, line_type=cv2.LINE_AA)
    # Back stub (dashed via short segments)
    bx = int(cx - length * 0.55 * math.cos(rad))
    by = int(cy - length * 0.55 * math.sin(rad))
    for t in np.arange(0, 1, 0.18):
        x1 = int(cx + (bx - cx) * t)
        y1 = int(cy + (by - cy) * t)
        x2 = int(cx + (bx - cx) * (t + 0.10))
        y2 = int(cy + (by - cy) * (t + 0.10))
        cv2.line(frame, (x1, y1), (x2, y2),
                 tuple(int(c * 0.5) for c in color), 1, cv2.LINE_AA)


def _arc(frame, cx, cy, angle_deg, radius, color, thickness=2):
    """Draw an arc from the +X axis to angle_deg (clockwise)."""
    start = -int(angle_deg)    # OpenCV: negative = counter-clockwise from +X
    cv2.ellipse(frame, (cx, cy), (radius, radius),
                0, start, 0, color, thickness, cv2.LINE_AA)


def _label(frame, cx, cy, track: Track, box_h: int):
    """Draw ID + angle + method label above the object."""
    angle_str = f"{track.angle_disp:.1f} deg"
    id_str     = f"ID:{track.id}"
    method_str = track.method
    line1      = f"{id_str}  {angle_str}"
    line2      = method_str

    lx = cx - 4
    ly = cy - box_h // 2 - 34

    for i, text in enumerate([line1, line2]):
        (tw, th), _ = cv2.getTextSize(text, C.FONT, C.FONT_SCALE, C.FONT_THICKNESS)
        tx = lx
        ty = ly + i * (th + 6)
        cv2.rectangle(frame,
                      (tx - 4, ty - th - 3),
                      (tx + tw + 4, ty + 4),
                      C.COL_TEXT_BG, -1)
        color = C.COL_TEXT if i == 0 else (160, 160, 120)
        cv2.putText(frame, text, (tx, ty), C.FONT,
                    C.FONT_SCALE, color, C.FONT_THICKNESS, cv2.LINE_AA)


def draw_track(frame: np.ndarray, track: Track):
    """Draw all overlay elements for one track."""
    cx, cy = track.centroid
    angle  = track.angle_disp
    box_color = _METHOD_COLORS.get(track.method, C.COL_BOX)

    # Bounding box
    cv2.polylines(frame, [track.box_pts], True,
                  box_color, 2, cv2.LINE_AA)

    # Axis arrow
    _arrow(frame, cx, cy, angle, C.AXIS_ARROW_LEN, C.COL_ARROW)

    # Angle arc
    _arc(frame, cx, cy, angle, C.ARC_RADIUS, C.COL_ARC)

    # Center dot
    cv2.circle(frame, (cx, cy), 7,  C.COL_CENTER,     -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 7,  C.COL_CENTER_OUT,  1, cv2.LINE_AA)

    # Arc mid-label (small angle readout on the arc itself)
    mid_rad = math.radians(-angle / 2)
    lx = int(cx + (C.ARC_RADIUS + 14) * math.cos(mid_rad))
    ly = int(cy + (C.ARC_RADIUS + 14) * math.sin(mid_rad))
    cv2.putText(frame, f"{angle:.0f} deg", (lx - 10, ly + 4),
                C.FONT, 0.42, C.COL_ARC, 1, cv2.LINE_AA)

    # Main label
    _, (bw, bh), _ = cv2.minAreaRect(np.float32([[cx, cy]]) if len(track.box_pts) == 0
                                     else track.box_pts.reshape(-1, 1, 2))
    box_h = int(max(bw, bh))
    _label(frame, cx, cy, track, box_h)


def draw_hud(frame: np.ndarray, fps: float, n_objects: int,
             method: str, frame_num: int):
    """Top-right HUD: FPS, object count, active method, frame counter."""
    h, w = frame.shape[:2]
    lines = [
        f"FPS: {fps:.1f}",
        f"Objects: {n_objects}",
        f"Method: {method}",
        f"Frame: {frame_num}",
    ]
    for i, text in enumerate(lines):
        (tw, th), _ = cv2.getTextSize(text, C.FONT, 0.50, 1)
        x = w - tw - 10
        y = 22 + i * (th + 8)
        cv2.rectangle(frame, (x - 4, y - th - 3),
                      (x + tw + 4, y + 4), (15, 10, 25), -1)
        cv2.putText(frame, text, (x, y), C.FONT, 0.50,
                    (200, 200, 200), 1, cv2.LINE_AA)

    # Legend — method colour key
    legend = [("contour", _METHOD_COLORS["contour"]),
              ("pca",     _METHOD_COLORS["pca"]),
              ("orb",     _METHOD_COLORS["orb"])]
    for i, (name, col) in enumerate(legend):
        lx, ly = 10, h - 20 - i * 20
        cv2.rectangle(frame, (lx, ly - 10), (lx + 12, ly + 2), col, -1)
        cv2.putText(frame, name, (lx + 18, ly),
                    C.FONT, 0.40, (190, 190, 190), 1, cv2.LINE_AA)

    # Reference axis (dashed horizontal through centre)
    mid_y = h // 2
    for x in range(0, w, 16):
        cv2.line(frame, (x, mid_y), (x + 8, mid_y),
                 (60, 60, 60), 1)
