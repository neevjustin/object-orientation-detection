# ============================================================
#  core/angle_utils.py — Angle math utilities
# ============================================================

import math
import numpy as np
import config as C


def normalize_angle(angle_deg: float) -> float:
    """Bring any angle into [0, 360)."""
    return angle_deg % 360


def minAreaRect_to_angle(rect) -> float:
    """
    cv2.minAreaRect returns angle in [-90, 0).
    Convert to [0, 360) from the +X axis (clockwise).
    """
    (_, (w, h), angle) = rect
    if w < h:
        angle += 90
    else:
        angle += 180
    return normalize_angle(angle)


def pca_angle(pts: np.ndarray) -> float:
    """
    Principal Component Analysis on a contour's pixel coordinates.
    Returns angle of the first principal component in [0, 360).
    """
    pts = pts.astype(np.float64)
    mean, eigenvectors, _ = cv_pca(pts)
    angle_rad = math.atan2(eigenvectors[0, 1], eigenvectors[0, 0])
    angle_deg = math.degrees(angle_rad)
    return normalize_angle(angle_deg)


def cv_pca(pts: np.ndarray):
    """Thin wrapper: returns mean, eigenvectors, eigenvalues."""
    import cv2
    data = pts.reshape(-1, 1, 2).astype(np.float64)
    mean, eigenvectors = cv2.PCACompute(
        pts.reshape(-1, 2).astype(np.float64), mean=None
    )
    _, eigenvalues = cv2.PCACompute2(
        pts.reshape(-1, 2).astype(np.float64), mean=None
    )
    return mean, eigenvectors, eigenvalues


def ema_update(prev: float | None, new: float, alpha: float = C.ANGLE_EMA_ALPHA) -> float:
    """
    Exponential moving average with angular wrap handling.
    Returns smoothed angle in [0, 360).
    """
    if prev is None:
        return new
    # Shortest angular distance
    delta = ((new - prev + 180) % 360) - 180
    smoothed = prev + alpha * delta
    return normalize_angle(smoothed)


def apply_dead_zone(prev: float | None, new: float,
                    zone: float = C.ANGLE_DEAD_ZONE) -> float:
    """Return prev if delta < zone, else new. Prevents flicker."""
    if prev is None:
        return new
    delta = abs(((new - prev + 180) % 360) - 180)
    return prev if delta < zone else new
