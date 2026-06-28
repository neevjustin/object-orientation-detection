# ============================================================
#  utils/logger.py — CSV angle logger
# ============================================================

import csv
import time
from pathlib import Path
import config as C


class AngleLogger:
    """Appends one row per object per frame to a CSV file."""

    HEADER = ["timestamp", "frame", "obj_id", "cx", "cy",
              "angle_raw", "angle_smooth", "method", "area"]

    def __init__(self, path: str):
        self._path = Path(path)
        self._file = open(self._path, "w", newline="")
        self._writer = csv.writer(self._file, delimiter=C.LOG_DELIMITER)
        self._writer.writerow(self.HEADER)

    def log(self, frame_num: int, tracks: list):
        ts = round(time.time(), 4)
        for t in tracks:
            self._writer.writerow([
                ts, frame_num, t.id,
                t.centroid[0], t.centroid[1],
                round(t.angle_raw, 2),
                round(t.angle_disp, 2),
                t.method,
                int(t.area),
            ])

    def close(self):
        self._file.close()
