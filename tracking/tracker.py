# ============================================================
#  tracking/tracker.py — IoU + Hungarian multi-object tracker
#  Persistent IDs, EMA angle smoothing, dead-zone suppression.
# ============================================================

import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import deque
import config as C
from core.angle_utils import ema_update, apply_dead_zone


def _iou(boxA, boxB) -> float:
    """Intersection-over-Union of two (x, y, w, h) boxes."""
    ax, ay, aw, ah = boxA
    bx, by, bw, bh = boxB
    ix = max(ax, bx)
    iy = max(ay, by)
    ix2 = min(ax + aw, bx + bw)
    iy2 = min(ay + ah, by + bh)
    inter = max(0, ix2 - ix) * max(0, iy2 - iy)
    union = aw * ah + bw * bh - inter
    return inter / (union + 1e-6)


class Track:
    _id_counter = 0

    def __init__(self, detection: dict):
        Track._id_counter += 1
        self.id          = Track._id_counter
        self.bbox        = detection["bbox"]
        self.box_pts     = detection["box_pts"]
        self.centroid    = detection["centroid"]
        self.angle_raw   = detection["angle"]
        self.angle       = detection["angle"]           # smoothed
        self.angle_disp  = detection["angle"]           # display (dead-zoned)
        self.method      = detection["method"]
        self.area        = detection["area"]
        self.lost        = 0
        self.age         = 0
        self.history     = deque([detection["angle"]], maxlen=C.HISTORY_LEN)

    def update(self, detection: dict):
        self.bbox     = detection["bbox"]
        self.box_pts  = detection["box_pts"]
        self.centroid = detection["centroid"]
        self.method   = detection["method"]
        self.area     = detection["area"]
        self.lost     = 0
        self.age     += 1

        raw  = detection["angle"]
        self.angle_raw  = raw
        self.angle      = ema_update(self.angle, raw)
        self.angle_disp = apply_dead_zone(self.angle_disp, self.angle)
        self.history.append(self.angle)

    def mark_lost(self):
        self.lost += 1


class MultiObjectTracker:
    def __init__(self):
        self.tracks: list[Track] = []

    def update(self, detections: list[dict]) -> list[Track]:
        """
        Match new detections to existing tracks via IoU + Hungarian,
        update matched tracks, create new tracks, age out lost ones.
        Returns the current live track list.
        """
        if not self.tracks:
            self.tracks = [Track(d) for d in detections]
            return [t for t in self.tracks]

        if not detections:
            for t in self.tracks:
                t.mark_lost()
            self.tracks = [t for t in self.tracks if t.lost <= C.MAX_LOST_FRAMES]
            return self.tracks

        # Build IoU cost matrix  (tracks × detections)
        n_t = len(self.tracks)
        n_d = len(detections)
        cost = np.zeros((n_t, n_d), dtype=np.float32)

        for i, track in enumerate(self.tracks):
            for j, det in enumerate(detections):
                cost[i, j] = 1.0 - _iou(track.bbox, det["bbox"])

        row_ind, col_ind = linear_sum_assignment(cost)

        matched_tracks = set()
        matched_dets   = set()

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < (1.0 - C.IOU_THRESHOLD):
                self.tracks[r].update(detections[c])
                matched_tracks.add(r)
                matched_dets.add(c)

        # Unmatched tracks → mark lost
        for i, track in enumerate(self.tracks):
            if i not in matched_tracks:
                track.mark_lost()

        # Unmatched detections → new tracks
        for j, det in enumerate(detections):
            if j not in matched_dets:
                self.tracks.append(Track(det))

        # Prune dead tracks
        self.tracks = [t for t in self.tracks if t.lost <= C.MAX_LOST_FRAMES]
        return self.tracks

    def reset(self):
        self.tracks.clear()
        Track._id_counter = 0
