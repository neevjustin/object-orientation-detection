"""
Object Orientation Detector
----------------------------
Detects an object in an image/video/webcam feed, computes its orientation
angle with respect to the horizontal axis using PCA on the largest contour,
and displays the angle on the output frame.

Usage:
    python orientation_detector.py --source 0          # webcam
    python orientation_detector.py --source video.mp4  # video file
    python orientation_detector.py --source image.jpg  # image file
"""

import cv2
import numpy as np
import argparse
import math


def get_orientation(contour, img):
    """Compute orientation angle of a contour using PCA and draw axes."""
    pts = contour.reshape(-1, 2).astype(np.float64)

    # PCA
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(pts, mean=None)
    center = (int(mean[0, 0]), int(mean[0, 1]))

    # Principal axis (largest eigenvalue direction)
    p1 = eigenvectors[0]
    angle_rad = math.atan2(p1[1], p1[0])
    angle_deg = math.degrees(angle_rad)

    # Normalize to [0, 180)
    angle_deg = angle_deg % 180

    # Draw center point
    cv2.circle(img, center, 5, (0, 0, 255), -1)

    # Draw principal axis line
    axis_len = 80
    p1_end = (
        int(center[0] + axis_len * math.cos(angle_rad)),
        int(center[1] + axis_len * math.sin(angle_rad)),
    )
    p1_start = (
        int(center[0] - axis_len * math.cos(angle_rad)),
        int(center[1] - axis_len * math.sin(angle_rad)),
    )
    cv2.line(img, p1_start, p1_end, (0, 255, 0), 2)

    return angle_deg, center


def process_frame(frame, min_area=500):
    """Detect largest object contour and compute its orientation."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold + Otsu for robust binarization
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    output = frame.copy()

    if not contours:
        cv2.putText(output, "No object detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return output

    # Largest contour by area
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        cv2.putText(output, "No object detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return output

    # Draw contour and bounding box
    cv2.drawContours(output, [largest], -1, (255, 0, 0), 2)
    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    cv2.drawContours(output, [box], 0, (0, 255, 255), 2)

    # Orientation via PCA
    angle_deg, center = get_orientation(largest, output)

    # minAreaRect angle as alternative reference
    rect_angle = rect[2]
    if rect[1][0] < rect[1][1]:
        rect_angle = rect_angle + 90

    text = f"PCA Angle: {angle_deg:.1f} deg"
    text2 = f"BBox Angle: {rect_angle:.1f} deg"
    cv2.putText(output, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(output, text2, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    return output


def main():
    parser = argparse.ArgumentParser(description="Object Orientation Detector")
    parser.add_argument("--source", default="0",
                         help="Path to image/video file, or '0' for webcam")
    parser.add_argument("--min-area", type=int, default=500,
                         help="Minimum contour area to consider as object")
    args = parser.parse_args()

    source = args.source
    is_image = source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))

    if is_image:
        frame = cv2.imread(source)
        if frame is None:
            print(f"Error: cannot read image '{source}'")
            return
        result = process_frame(frame, args.min_area)
        cv2.imshow("Orientation Detection", result)
        cv2.imwrite("output_result.jpg", result)
        print("Result saved to output_result.jpg. Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # Video or webcam
    cap_source = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(cap_source)

    if not cap.isOpened():
        print(f"Error: cannot open source '{source}'")
        return

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = process_frame(frame, args.min_area)
        cv2.imshow("Orientation Detection", result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()