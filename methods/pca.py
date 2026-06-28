# ============================================================
#  methods/pca.py — PCA-based orientation (elongated shapes)
# ============================================================

import cv2
import numpy as np
import math
import config as C
from core.angle_utils import normalize_angle


def detect_pca(binary: np.ndarray) -> list[dict]:
    """
    PCA on contour pixel coordinates.
    More stable than minAreaRect for thin / elongated objects.
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    results = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (C.MIN_CONTOUR_AREA <= area <= C.MAX_CONTOUR_AREA):
            continue
        if len(cnt) < 5:
            continue

        # All pixel coords inside the contour
        pts = cnt.reshape(-1, 2).astype(np.float64)

        mean, eigvecs = cv2.PCACompute(pts, mean=None)
        cx, cy = mean[0]

        # First eigenvector → dominant axis
        vx, vy = eigvecs[0]
        angle_rad = math.atan2(vy, vx)
        angle_deg = normalize_angle(math.degrees(angle_rad))

        rect = cv2.minAreaRect(cnt)
        box  = np.int32(cv2.boxPoints(rect))
        bx, by, bw, bh = cv2.boundingRect(cnt)

        results.append({
            "bbox"    : (bx, by, bw, bh),
            "box_pts" : box,
            "centroid": (int(cx), int(cy)),
            "angle"   : angle_deg,
            "area"    : area,
            "contour" : cnt,
            "method"  : "pca",
        })

    return results
