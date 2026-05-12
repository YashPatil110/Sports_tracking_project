# ============================================================
# detector.py — YOLOv8-based person detector
# ============================================================

import numpy as np
from ultralytics import YOLO
import config


class PersonDetector:
    """
    Wraps YOLOv8 for person detection.
    Returns bounding boxes and confidence scores per frame.
    """

    def __init__(self):
        print(f"[Detector] Loading model: {config.MODEL_NAME}")
        self.model = YOLO(config.MODEL_NAME)
        self.conf  = config.CONF_THRESH
        self.cls   = config.PERSON_CLASS
        print("[Detector] Model ready.")

    def detect(self, frame: np.ndarray):
        """
        Run inference on a single BGR frame.

        Returns
        -------
        boxes  : np.ndarray  shape (N, 4)  — xyxy pixel coordinates
        confs  : np.ndarray  shape (N,)    — confidence scores
        """
        results = self.model(
            frame,
            conf=self.conf,
            classes=[self.cls],
            verbose=False,
        )

        boxes_tensor = results[0].boxes
        if boxes_tensor is None or len(boxes_tensor) == 0:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

        boxes = boxes_tensor.xyxy.cpu().numpy().astype(np.float32)
        confs = boxes_tensor.conf.cpu().numpy().astype(np.float32)
        return boxes, confs
