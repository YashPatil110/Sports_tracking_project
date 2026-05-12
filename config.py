# ============================================================
# config.py — Central configuration for the tracking pipeline
# ============================================================

# --- Input / Output ---
INPUT_VIDEO  = "input_video.mp4"       # Path to your downloaded video
OUTPUT_VIDEO = "outputs/output_video.mp4"

# --- Detection ---
MODEL_NAME   = "yolov8n.pt"            # YOLOv8 medium (good speed/accuracy balance)
CONF_THRESH  = 0.45                   # Minimum detection confidence
PERSON_CLASS = 0                       # COCO class index for "person"

# --- Tracking ---
FRAME_SKIP   = 4                       # Process every Nth frame (1 = all frames)
TRACK_THRESH = 0.25                    # ByteTrack: min score to activate a track
TRACK_BUFFER = 30                      # Frames to keep a lost track alive
MATCH_THRESH = 0.8                     # IoU threshold for track-detection matching

# --- Visualization ---
TRAJECTORY_LENGTH = 40                 # Number of past positions to draw per ID
SHOW_CONFIDENCE   = True               # Display confidence score next to ID
BOX_THICKNESS     = 2
FONT_SCALE        = 0.6

# --- Optional Enhancements ---
ENABLE_TRAJECTORY = True               # Draw movement trails
ENABLE_HEATMAP    = True               # Generate heatmap at end
ENABLE_COUNT_PLOT = True               # Plot object count over time

# --- Colors (BGR) ---
COLORS = [
    (0, 255, 0),   (255, 100, 0), (0, 100, 255), (255, 0, 255),
    (0, 255, 255), (255, 255, 0), (128, 0, 255), (0, 200, 128),
    (255, 128, 0), (0, 128, 255),
]
