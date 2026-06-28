# ============================================================
#  methods/keypoint.py — ORB keypoint orientation
#  Best for textured objects where contour shape is unreliable.
#  Falls back to PCA if not enough keypoints are found.
# ============================================================

import cv2
import numpy as np
import math
import config as C
from core.angle_utils import normalize_angle
from methods.pca import detect_pca

_orb = cv2.ORB_create(nfeatures=500)


def detect_orb(binary: np.ndarray, gray: np.ndarray) -> list[dict]:
    """
    Detects keypoints with ORB and computes average keypoint angle
    within each contour's bounding area.

    Falls back to PCA per-contour if < 3 keypoints are found inside.
    """
    kps, _ = _orb.detectAndCompute(gray, None)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    results = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (C.MIN_CONTOUR_AREA <= area <= C.MAX_CONTOUR_AREA):
            continue

        # Find keypoints inside this contour
        inside = [
            kp for kp in kps
            if cv2.pointPolygonTest(cnt, kp.pt, False) >= 0
        ]

        if len(inside) >= 3:
            angles = [normalize_angle(kp.angle) for kp in inside
                      if kp.angle >= 0]
            # Circular mean of keypoint angles
            sin_sum = sum(math.sin(math.radians(a)) for a in angles)
            cos_sum = sum(math.cos(math.radians(a)) for a in angles)
            angle_deg = normalize_angle(
                math.degrees(math.atan2(sin_sum, cos_sum))
            )
            method = "orb"
        else:
            # Fallback: PCA on this single contour
            from methods.pca import detect_pca as _pca_single
            sub_binary = np.zeros_like(binary)
            cv2.drawContours(sub_binary, [cnt], -1, 255, -1)
            pca_res = _pca_single(sub_binary)
            if pca_res:
                angle_deg = pca_res[0]["angle"]
            else:
                rect = cv2.minAreaRect(cnt)
                from core.angle_utils import minAreaRect_to_angle
                angle_deg = minAreaRect_to_angle(rect)
            method = "pca(fallback)"

        rect = cv2.minAreaRect(cnt)
        box  = np.int32(cv2.boxPoints(rect))
        (cx, cy), _ , _ = rect
        bx, by, bw, bh = cv2.boundingRect(cnt)

        results.append({
            "bbox"    : (bx, by, bw, bh),
            "box_pts" : box,
            "centroid": (int(cx), int(cy)),
            "angle"   : angle_deg,
            "area"    : area,
            "contour" : cnt,
            "method"  : method,
        })

    return results
