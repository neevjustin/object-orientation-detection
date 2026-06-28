# ============================================================
#  core/detector.py — Adaptive method selector
#  Picks contour / PCA / ORB per-object based on shape properties.
# ============================================================

import cv2
import numpy as np
import config as C
from methods.contour  import detect_contour
from methods.pca      import detect_pca
from methods.keypoint import detect_orb


def _harris_corner_count(gray: np.ndarray, mask: np.ndarray) -> int:
    """Count Harris corners inside a binary mask region."""
    harris = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    harris = cv2.dilate(harris, None)
    corners = (harris > 0.01 * harris.max()) & (mask > 0)
    return int(np.sum(corners))


def detect(binary: np.ndarray, gray: np.ndarray,
           method: str = C.DEFAULT_METHOD) -> list[dict]:
    """
    Run detection with the given method.

    method = "auto"    → per-object adaptive selection
    method = "contour" → always use minAreaRect
    method = "pca"     → always use PCA
    method = "orb"     → always use ORB keypoints
    """
    if method == "contour":
        return detect_contour(binary)
    if method == "pca":
        return detect_pca(binary)
    if method == "orb":
        return detect_orb(binary, gray)

    # ── "auto" mode ──────────────────────────────────────────
    # First pass: get contours + basic geometry
    base = detect_contour(binary)
    base = sorted(base, key=lambda x: x["area"], reverse=True)[:6]
    refined = []

    for obj in base:
        cnt  = obj["contour"]
        area = obj["area"]

        # Aspect ratio from minAreaRect
        rect   = cv2.minAreaRect(cnt)
        _, (w, h), _ = rect
        aspect = max(w, h) / (min(w, h) + 1e-5)

        # Build single-contour mask for Harris
        sub_mask = np.zeros(binary.shape, dtype=np.uint8)
        cv2.drawContours(sub_mask, [cnt], -1, 255, -1)
        n_corners = _harris_corner_count(gray, sub_mask)

        if n_corners > C.HARRIS_THRESHOLD:
            # Textured object → ORB
            sub_bin = sub_mask
            result  = detect_orb(sub_bin, gray)
        elif aspect >= C.MIN_ASPECT_RATIO:
            # Elongated shape → PCA
            sub_bin = sub_mask
            result  = detect_pca(sub_bin)
        else:
            # Blobby / convex → contour is fine
            result = [obj]

        if result:
            refined.append(result[0])
        else:
            refined.append(obj)   # keep original as fallback

    return refined
