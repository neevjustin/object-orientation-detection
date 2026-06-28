# ============================================================
#  methods/contour.py — Contour + minAreaRect detection
# ============================================================

import cv2
import numpy as np
import config as C
from core.angle_utils import minAreaRect_to_angle


def detect_contour(binary: np.ndarray) -> list[dict]:
    """
    Find objects via contours.

    Returns list of dicts:
        {
            "bbox"    : (x, y, w, h),
            "box_pts" : np.int32 of 4 corners,
            "centroid": (cx, cy),
            "angle"   : float in [0, 360),
            "area"    : float,
            "method"  : "contour",
        }
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    results = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (C.MIN_CONTOUR_AREA <= area <= C.MAX_CONTOUR_AREA):
            continue

        rect = cv2.minAreaRect(cnt)
        box  = np.int32(cv2.boxPoints(rect))
        (cx, cy), (w, h), _ = rect
        bx, by, bw, bh = cv2.boundingRect(cnt)

        angle = minAreaRect_to_angle(rect)

        results.append({
            "bbox"    : (bx, by, bw, bh),
            "box_pts" : box,
            "centroid": (int(cx), int(cy)),
            "angle"   : angle,
            "area"    : area,
            "contour" : cnt,
            "method"  : "contour",
        })

    return results
