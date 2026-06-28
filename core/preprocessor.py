# ============================================================
#  core/preprocessor.py — Frame → binary mask pipeline
# ============================================================

import cv2
import numpy as np
import config as C


def preprocess(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    small   : np.ndarray  — resized BGR frame used for display + detection
    binary  : np.ndarray  — binary mask (uint8, 0/255) ready for findContours
    """
    # 1. Resize for speed
    h, w = frame.shape[:2]
    if w != C.PROCESS_WIDTH:
        scale = C.PROCESS_WIDTH / w
        small = cv2.resize(frame, (C.PROCESS_WIDTH, int(h * scale)),
                           interpolation=cv2.INTER_LINEAR)
    else:
        small = frame.copy()

    # 2. Grayscale
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # 3. Gaussian blur
    blurred = cv2.GaussianBlur(gray, C.BLUR_KERNEL, 0)

    # 4. Adaptive threshold — robust against uneven lighting
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        C.THRESH_BLOCK, C.THRESH_C
    )

    # 5. Morphological open — kill tiny noise blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, C.MORPH_KERNEL)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel,
                               iterations=C.MORPH_ITERS)

    return small, binary
