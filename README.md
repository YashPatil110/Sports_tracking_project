# 🎯 Multi-Object Detection & Persistent ID Tracking in Sports Footage

**Assignment:** Predusk Technology — AI / Computer Vision Assessment  
**Video Source:** *(Add your public video URL here)*

---

## 📌 Project Overview

This pipeline detects all people in a sports/event video and assigns each person a **unique, persistent ID** that stays consistent across the entire video — even when subjects overlap, move fast, or temporarily leave the frame.

---

## 🗂️ Project Structure

```
sports_tracker/
├── config.py            # All tunable settings (model, thresholds, etc.)
├── detector.py          # YOLOv8-based person detector
├── tracker.py           # ByteTrack multi-object tracker
├── visualizer.py        # Bounding boxes, trajectories, heatmaps, count plots
├── pipeline.py          # Main entry point — runs the full pipeline
├── download_video.py    # Helper to download a public video via yt-dlp
├── requirements.txt
├── README.md
└── outputs/
    ├── output_video.mp4         # Annotated output
    └── screenshots/
        ├── frame_XXXXX.jpg      # Sample frames
        ├── heatmap.png          # Movement heatmap
        └── count_over_time.png  # Tracked subject count chart
```

---

## ⚙️ Installation

### 1. Clone / unzip the project
```bash
cd sports_tracker
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **GPU (optional):** If you have an NVIDIA GPU, install the CUDA version of PyTorch first for faster inference:  
> `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

---

## ▶️ How to Run

### Step 1 — Download your video
```bash
python download_video.py "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"

my video link "https://www.youtube.com/watch?v=CSTRAWxWBwQ"
```
This saves the video as `input_video.mp4`.

### Step 2 — Run the pipeline
```bash
python pipeline.py
```

Or pass custom paths directly:
```bash
python pipeline.py my_video.mp4 outputs/my_output.mp4
```

### Step 3 — View results
- **Annotated video:** `outputs/output_video.mp4`
- **Heatmap:** `outputs/screenshots/heatmap.png`
- **Count chart:** `outputs/screenshots/count_over_time.png`
- **Sample frames:** `outputs/screenshots/frame_*.jpg`

---

## 🔧 Configuration

Edit `config.py` to tune the pipeline without changing any code:

| Parameter         | Default       | Description                                      |
|-------------------|---------------|--------------------------------------------------|
| `MODEL_NAME`      | `yolov8m.pt`  | YOLOv8 variant (n / s / m / l / x)              |
| `CONF_THRESH`     | `0.30`        | Minimum detection confidence                     |
| `FRAME_SKIP`      | `2`           | Process every Nth frame (1 = all frames)         |
| `TRACK_BUFFER`    | `30`          | Frames to keep a lost track alive                |
| `TRAJECTORY_LENGTH` | `40`        | Trail length per tracked subject                 |
| `ENABLE_HEATMAP`  | `True`        | Generate movement heatmap                        |
| `ENABLE_COUNT_PLOT` | `True`      | Generate count-over-time chart                   |

---

## 🧠 Model & Tracker Choices

### Detector — YOLOv8 (medium)
- Pre-trained on COCO dataset; "person" class available out of the box
- Excellent speed/accuracy trade-off
- Single-stage: no separate proposal network needed
- `yolov8m.pt` chosen over `yolov8s` for better recall in crowded scenes

### Tracker — ByteTrack (via `supervision`)
- Two-stage IoU matching: high-confidence detections matched first, low-confidence used to recover lost tracks
- Kalman filter predicts each track's next position — handles brief occlusion without ID switch
- No appearance embedding required → works on CPU without GPU
- Outperforms DeepSORT in most sports benchmarks (SORT + second-association step)

---

## 🔄 How ID Consistency Is Maintained

1. **Kalman filter** predicts where each tracked subject will be in the next frame.
2. **IoU matching** links predictions to new detections (Hungarian algorithm).
3. **Second association pass** rescues detections that scored below the main threshold — key for crowded frames.
4. Tracks that go unmatched are held in a "lost" buffer (`TRACK_BUFFER` frames) before being deleted — so subjects that briefly leave frame or are occluded keep their ID when they reappear.

---

## ⚠️ Assumptions

- Input video is in `.mp4` format (or any format OpenCV can read)
- Subjects are primarily **people** (COCO class 0)
- Camera may move (pan/zoom) — tracker handles this via relative IoU
- `FRAME_SKIP=2` is used by default (slight speed boost, negligible quality loss at 25–30 fps)

---

## 🚧 Limitations

| Issue | Cause | Workaround |
|-------|-------|------------|
| ID switch on full occlusion | Both subjects fully overlap | Increase `TRACK_BUFFER` |
| False detections (referee, crowd) | Low confidence threshold | Increase `CONF_THRESH` |
| Slow on CPU for HD video | Inference per frame cost | Use `yolov8n.pt` or increase `FRAME_SKIP` |
| Similar-looking subjects | No appearance embeddings | Upgrade to BoT-SORT or StrongSORT with ReID |

---

## 🚀 Possible Improvements

- **ReID model** (e.g., OSNet, BoT-SORT) — appearance embeddings for robust re-identification after long occlusion
- **Team clustering** — K-means on jersey colour histograms to separate teams
- **Top-down projection** — homography to map positions onto a bird's-eye field view
- **Speed estimation** — pixel displacement × calibration factor per second
- **Model comparison** — benchmark YOLOv8n vs YOLOv8m vs YOLOv9 on same video

---

## 📦 Deliverables

- [x] Clean, modular Python codebase
- [x] `README.md` with full instructions
- [x] Annotated output video (`outputs/output_video.mp4`)
- [x] Public video source link *(add yours above)*
- [x] Movement heatmap (`outputs/screenshots/heatmap.png`)
- [x] Object count over time plot (`outputs/screenshots/count_over_time.png`)
- [x] Sample screenshots (`outputs/screenshots/frame_*.jpg`)

---

## 📄 License

For assessment/evaluation purposes only.
