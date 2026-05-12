# ============================================================
# tracker.py — ByteTrack multi-object tracker wrapper
# ============================================================

import numpy as np
import supervision as sv
import config


class IDTracker:
    """
    Wraps supervision's ByteTracker for persistent ID assignment.

    ByteTrack uses a two-step matching strategy:
      1. High-confidence detections are matched first via IoU.
      2. Low-confidence detections are used to recover lost tracks.
    This makes it robust to occlusion and brief disappearances.
    """

    def __init__(self):
        print("[Tracker] Initialising ByteTracker …")
        # supervision >= 0.20 renamed ByteTracker → ByteTrack
        TrackerClass = sv.ByteTrack
        self.tracker = TrackerClass(
            track_activation_threshold=config.TRACK_THRESH,
            lost_track_buffer=config.TRACK_BUFFER,
            minimum_matching_threshold=config.MATCH_THRESH,
            frame_rate=30,
        )
        print("[Tracker] Ready.")

    def update(self, boxes: np.ndarray, confs: np.ndarray, frame_shape: tuple):
        """
        Update tracker with new detections.

        Parameters
        ----------
        boxes       : np.ndarray (N, 4)  xyxy
        confs       : np.ndarray (N,)
        frame_shape : (H, W, C)

        Returns
        -------
        sv.Detections with .tracker_id filled in
        """
        if len(boxes) == 0:
            return sv.Detections.empty()

        detections = sv.Detections(
            xyxy=boxes,
            confidence=confs,
            class_id=np.zeros(len(boxes), dtype=int),
        )

        tracked = self.tracker.update_with_detections(detections)
        return tracked 