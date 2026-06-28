# ============================================================
#  core/capture.py — Threaded VideoCapture
#  Decouples I/O from processing so the main loop never blocks.
# ============================================================

import threading
import cv2
import config as C


class ThreadedCapture:
    """
    Runs cv2.VideoCapture in a background thread.
    The main loop always gets the latest frame without waiting for .read().
    """

    def __init__(self, source):
        """
        source : int  → webcam index
                 str  → video file path or image path
        """
        self.source = source
        self._is_image = isinstance(source, str) and source.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
        )

        if self._is_image:
            frame = cv2.imread(source)
            if frame is None:
                raise FileNotFoundError(f"Cannot read image: {source}")
            self._frame  = frame
            self._ret    = True
            self._thread = None
            return

        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open source: {source}")

        # Request resolution
        if C.FRAME_WIDTH > 0:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  C.FRAME_WIDTH)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, C.FRAME_HEIGHT)

        self._ret, self._frame = self._cap.read()
        self._lock   = threading.Lock()
        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    # ── public ──────────────────────────────────────────────

    def read(self):
        """Return (ret, frame) — always the most recent frame."""
        if self._is_image:
            return self._ret, self._frame.copy()
        with self._lock:
            return self._ret, self._frame.copy() if self._ret else (False, None)

    @property
    def fps(self):
        if self._is_image or not hasattr(self, "_cap"):
            return 0.0
        return self._cap.get(cv2.CAP_PROP_FPS) or 30.0

    @property
    def is_image(self):
        return self._is_image

    def release(self):
        if self._thread:
            self._stop.set()
            self._thread.join(timeout=2)
        if hasattr(self, "_cap"):
            self._cap.release()

    # ── private ─────────────────────────────────────────────

    def _reader(self):
        while not self._stop.is_set():
            ret, frame = self._cap.read()
            if not ret:
                # End of video file — loop back
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._cap.read()
            with self._lock:
                self._ret   = ret
                self._frame = frame
