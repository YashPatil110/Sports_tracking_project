# ============================================================
# visualizer.py — Annotation, trajectory, heatmap utilities
# ============================================================

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
import config


def get_color(track_id: int):
    """Return a consistent BGR color for a given track ID."""
    return config.COLORS[int(track_id) % len(config.COLORS)]


def draw_tracks(frame: np.ndarray, detections, track_history: dict) -> np.ndarray:
    """
    Draw bounding boxes, IDs, confidence scores, and trajectory trails.

    Parameters
    ----------
    frame         : BGR frame to annotate (modified in-place copy)
    detections    : sv.Detections with tracker_id
    track_history : dict mapping track_id → list of (cx, cy) centroids

    Returns
    -------
    Annotated frame
    """
    out = frame.copy()

    if detections.tracker_id is None or len(detections.tracker_id) == 0:
        return out

    for i, tid in enumerate(detections.tracker_id):
        x1, y1, x2, y2 = detections.xyxy[i].astype(int)
        conf = float(detections.confidence[i]) if detections.confidence is not None else 0.0
        color = get_color(tid)

        # --- Bounding box ---
        cv2.rectangle(out, (x1, y1), (x2, y2), color, config.BOX_THICKNESS)

        # --- Label ---
        label = f"ID {tid}"
        if config.SHOW_CONFIDENCE:
            label += f"  {conf:.2f}"

        label_size, baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE, 1
        )
        label_y = max(y1, label_size[1] + 4)
        cv2.rectangle(
            out,
            (x1, label_y - label_size[1] - 4),
            (x1 + label_size[0] + 4, label_y + baseline),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            out, label,
            (x1 + 2, label_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            config.FONT_SCALE,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        # --- Centroid ---
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        track_history[tid].append((cx, cy))

        # --- Trajectory trail ---
        if config.ENABLE_TRAJECTORY:
            pts = track_history[tid][-config.TRAJECTORY_LENGTH:]
            for j in range(1, len(pts)):
                alpha = j / len(pts)          # fade older points
                t_color = tuple(int(c * alpha) for c in color)
                cv2.line(out, pts[j - 1], pts[j], t_color, 2, cv2.LINE_AA)

    return out


def generate_heatmap(heatmap_acc: np.ndarray, frame_shape: tuple, output_path: str):
    """
    Save a movement heatmap overlay as a PNG.

    Parameters
    ----------
    heatmap_acc  : 2-D float array accumulating centroid visits
    frame_shape  : (H, W, C) of the original video
    output_path  : PNG file path to write
    """
    H, W = frame_shape[:2]
    norm = cv2.normalize(heatmap_acc, None, 0, 255, cv2.NORM_MINMAX)
    norm = norm.astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)

    # Blend with a white background
    background = np.ones((H, W, 3), dtype=np.uint8) * 255
    blended = cv2.addWeighted(background, 0.3, colored, 0.7, 0)

    cv2.imwrite(output_path, blended)
    print(f"[Visualizer] Heatmap saved → {output_path}")


def generate_count_plot(count_log: list, fps: float, output_path: str):
    """
    Plot number of tracked subjects over time and save as PNG.

    Parameters
    ----------
    count_log   : list of (frame_number, count) tuples
    fps         : video frames per second
    output_path : PNG file path
    """
    frames, counts = zip(*count_log) if count_log else ([], [])
    times = [f / fps for f in frames]

    plt.figure(figsize=(12, 4))
    plt.plot(times, counts, color="steelblue", linewidth=1.5)
    plt.fill_between(times, counts, alpha=0.25, color="steelblue")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Tracked Subjects")
    plt.title("Number of Tracked Subjects Over Time")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f"[Visualizer] Count plot saved → {output_path}")
