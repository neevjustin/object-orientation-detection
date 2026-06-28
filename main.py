#!/usr/bin/env python3
# ============================================================
#  main.py — Object Orientation Detector  (production entry point)
#
#  Usage:
#    python main.py                          # default webcam
#    python main.py --source 1              # second camera
#    python main.py --source video.mp4
#    python main.py --source image.jpg
#    python main.py --method pca
#    python main.py --save out.mp4 --log angles.csv
#    python main.py --no-display --log data.csv   # headless
# ============================================================

import argparse
import sys
import cv2
import numpy as np

import config as C
from core.capture      import ThreadedCapture
from core.preprocessor import preprocess
from core.detector     import detect
from tracking.tracker  import MultiObjectTracker
from overlay.renderer  import draw_track, draw_hud
from utils.fps         import FPSCounter
from utils.logger      import AngleLogger


# ── CLI ─────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Object Orientation Detector",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--source",  default=0,
                   help="Webcam index (int) | video path | image path")
    p.add_argument("--method",  default=C.DEFAULT_METHOD,
                   choices=["auto", "contour", "pca", "orb"],
                   help="Detection method (default: auto)")
    p.add_argument("--min-area", type=int, default=C.MIN_CONTOUR_AREA,
                   help="Minimum contour area in pixels²")
    p.add_argument("--save",    default=None,
                   help="Path to save output video (e.g. out.mp4)")
    p.add_argument("--log",     default=None,
                   help="Path to save angle CSV log (e.g. angles.csv)")
    p.add_argument("--no-display", action="store_true",
                   help="Headless mode — no window (use with --save or --log)")
    return p.parse_args()


# ── Helpers ─────────────────────────────────────────────────

def _try_int(v):
    try:
        return int(v)
    except (ValueError, TypeError):
        return v


def _make_writer(path: str, frame: np.ndarray, fps: float):
    h, w = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps or C.OUTPUT_VIDEO_FPS, (w, h))


def print_controls():
    print("\n╔══════════════════════════════════════╗")
    print("║   Orientation Detector — Controls    ║")
    print("╠══════════════════════════════════════╣")
    print("║  Q / ESC   Quit                      ║")
    print("║  R         Reset tracker IDs         ║")
    print("║  S         Screenshot (frame.png)    ║")
    print("║  1 / 2 / 3 Switch method             ║")
    print("║            1=auto 2=contour 3=pca    ║")
    print("╚══════════════════════════════════════╝\n")


# ── Main loop ───────────────────────────────────────────────

def run():
    args = parse_args()
    source = _try_int(args.source)
    C.MIN_CONTOUR_AREA = args.min_area
    method = args.method

    print(f"[INFO] Source  : {source}")
    print(f"[INFO] Method  : {method}")

    cap     = ThreadedCapture(source)
    tracker = MultiObjectTracker()
    fps_ctr = FPSCounter()
    logger  = AngleLogger(args.log) if args.log else None
    writer  = None
    frame_n = 0

    print_controls()

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[WARN] No frame received — exiting.")
            break

        frame_n += 1

        # ── Process ───────────────────────────────────────
        small, binary = preprocess(frame)
        import cv2 as _cv2
        gray   = _cv2.cvtColor(small, _cv2.COLOR_BGR2GRAY)

        detections = detect(binary, gray, method)
        tracks     = tracker.update(detections)

        fps = fps_ctr.tick()

        # ── Draw ─────────────────────────────────────────
        vis = small.copy()
        for t in tracks:
            if t.lost == 0:
                draw_track(vis, t)

        draw_hud(vis, fps, sum(1 for t in tracks if t.lost == 0),
                 method, frame_n)

        # ── Log ──────────────────────────────────────────
        if logger:
            logger.log(frame_n, [t for t in tracks if t.lost == 0])

        # ── Display ──────────────────────────────────────
        if not args.no_display:
            cv2.imshow("Orientation Detector  [Q=quit]", vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("r"):
                tracker.reset()
                print("[INFO] Tracker reset.")
            elif key == ord("s"):
                cv2.imwrite("frame.png", vis)
                print("[INFO] Screenshot saved: frame.png")
            elif key == ord("1"):
                method = "auto";    print("[INFO] Method → auto")
            elif key == ord("2"):
                method = "contour"; print("[INFO] Method → contour")
            elif key == ord("3"):
                method = "pca";     print("[INFO] Method → pca")

        # ── Save video ───────────────────────────────────
        if args.save:
            if writer is None:
                src_fps = cap.fps or C.OUTPUT_VIDEO_FPS
                writer  = _make_writer(args.save, vis, src_fps)
            writer.write(vis)

        # Image source: single frame then exit
        if cap.is_image:
            if not args.no_display:
                print("[INFO] Showing image. Press any key to exit.")
                cv2.waitKey(0)
            break

    # ── Cleanup ──────────────────────────────────────────
    cap.release()
    if writer:
        writer.release()
    if logger:
        logger.close()
        print(f"[INFO] Angle log saved: {args.log}")
    cv2.destroyAllWindows()
    print("[INFO] Done.")


if __name__ == "__main__":
    run()
