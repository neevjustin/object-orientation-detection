# ============================================================
#  utils/fps.py — Rolling-window FPS counter
# ============================================================

import time
from collections import deque


class FPSCounter:
    def __init__(self, window: int = 30):
        self._times: deque = deque(maxlen=window)
        self._last  = time.perf_counter()

    def tick(self) -> float:
        now = time.perf_counter()
        self._times.append(now - self._last)
        self._last = now
        if len(self._times) < 2:
            return 0.0
        return len(self._times) / sum(self._times)
