# ============================================================
# pipeline.py — Main entry point for the tracking pipeline
# ============================================================

import cv2
import numpy as np
import os
import sys
import time
from collections import defaultdict

import config
from detector import PersonDetector
from tracker import IDTracker
from visualizer import (
    draw_tracks,
    generate_heatmap,
    generate_count_plot,
)


def run(input_path: str = None, output_path: str = None):
    """
    Full detection + tracking pipeline.

    Parameters
    ----------
    input_path  : path to input video (defaults to config.INPUT_VIDEO)
    output_path : path to output video (defaults to config.OUTPUT_VIDEO)
    """
    input_path  = input_path  or config.INPUT_VIDEO
    output_path = output_path or config.OUTPUT_VIDEO

    # --- Validate input ---
    if not os.path.exists(input_path):
        print(f"[ERROR] Input video not found: {input_path}")
        print("  Please download a video and set INPUT_VIDEO in config.py")
        sys.exit(1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # --- Open video ---
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {input_path}")
        sys.exit(1)

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps    = fps / config.FRAME_SKIP
    W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[Pipeline] Video: {W}×{H}  {fps:.1f} fps  {total} frames")

    # --- Writer ---
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    # --- Components ---
    detector      = PersonDetector()
    tracker       = IDTracker()
    track_history = defaultdict(list)    # tid → [(cx,cy), …]
    heatmap_acc   = np.zeros((H, W), dtype=np.float32)
    count_log     = []                   # [(frame_num, count), …]
    screenshot_saved = 0

    print("[Pipeline] Processing …")
    frame_num   = 0
    start_time  = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- Detection (every Nth frame) ---
        if frame_num % config.FRAME_SKIP == 0:
            boxes, confs = detector.detect(frame)
            detections   = tracker.update(boxes, confs, frame.shape)

            # Heatmap accumulation
            if config.ENABLE_HEATMAP and detections.tracker_id is not None:
                for box in detections.xyxy:
                    cx = int((box[0] + box[2]) / 2)
                    cy = int((box[1] + box[3]) / 2)
                    cv2.circle(heatmap_acc, (cx, cy), 5, 1.0, -1)

            # Count log
            n = len(detections.tracker_id) if detections.tracker_id is not None else 0
            count_log.append((frame_num, n))

            # Annotate
            annotated = draw_tracks(frame, detections, track_history)
        else:
            # For skipped frames just redraw last known tracks
            annotated = frame.copy()

        # --- Frame counter overlay ---
        cv2.putText(
            annotated, f"Frame {frame_num}", (10, H - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA
        )

        writer.write(annotated)

        # --- Save sample screenshots ---
        if screenshot_saved < 5 and frame_num % max(1, total // 5) == 0:
            sc_path = f"outputs/screenshots/frame_{frame_num:05d}.jpg"
            cv2.imwrite(sc_path, annotated)
            screenshot_saved += 1

        # --- Progress ---
        if frame_num % 100 == 0:
            elapsed = time.time() - start_time
            pct = (frame_num / max(total, 1)) * 100
            print(f"  {pct:5.1f}%  frame {frame_num}/{total}  ({elapsed:.0f}s elapsed)")

        frame_num += 1

    cap.release()
    writer.release()

    elapsed = time.time() - start_time
    print(f"\n[Pipeline] Finished {frame_num} frames in {elapsed:.1f}s")
    print(f"[Pipeline] Output → {output_path}")

    # --- Optional post-processing ---
    if config.ENABLE_HEATMAP:
        generate_heatmap(heatmap_acc, (H, W, 3), "outputs/screenshots/heatmap.png")

    if config.ENABLE_COUNT_PLOT:
        generate_count_plot(count_log, fps, "outputs/screenshots/count_over_time.png")

    print("\n✅ All done! Check the outputs/ folder.")


if __name__ == "__main__":
    # Allow overriding paths via CLI:  python pipeline.py my_video.mp4 my_output.mp4
    inp = sys.argv[1] if len(sys.argv) > 1 else None
    out = sys.argv[2] if len(sys.argv) > 2 else None
    run(inp, out)
