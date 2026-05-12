# Technical Report
## Multi-Object Detection and Persistent ID Tracking in Sports Footage

---

### 1. Model / Detector Used

**YOLOv8 Medium (`yolov8m.pt`)** — Ultralytics, 2023

YOLOv8 is a single-stage anchor-free object detector pre-trained on the COCO dataset (80 classes). The "person" class (class index 0) is used directly without any fine-tuning. The medium variant (`yolov8m`) was selected as the best trade-off between inference speed (~30 ms/frame on CPU) and detection recall in crowded or partially occluded scenes.

Confidence threshold set to **0.30** to capture partially visible players while limiting false positives from background crowd members.

---

### 2. Tracking Algorithm Used

**ByteTrack** (Zhang et al., 2022), accessed via the `supervision` library (Roboflow).

ByteTrack extends the SORT framework with a two-step detection–track association:
- **First pass:** High-confidence detections (above threshold) are matched to existing tracks using IoU and the Hungarian algorithm.
- **Second pass:** Low-confidence detections (often partially occluded subjects) are matched against unmatched tracks from the first pass.

A **Kalman filter** models each track's motion state `[x, y, w, h, vx, vy, vw, vh]` and predicts its position before each matching step.

---

### 3. Why This Combination Was Selected

| Criterion | YOLOv8 + ByteTrack |
|-----------|-------------------|
| No fine-tuning needed | ✅ COCO "person" class works directly |
| CPU-compatible | ✅ Both run efficiently without GPU |
| Handles occlusion | ✅ ByteTrack's second-pass association |
| Open source | ✅ MIT / Apache licensed |
| Active maintenance | ✅ Ultralytics + Roboflow communities |
| Sports-specific benchmarks | ✅ ByteTrack state-of-art on MOT17/MOT20 |

Alternatives evaluated and reasons for not selecting:
- **DeepSORT** — requires a separately trained ReID model; heavier, slower on CPU.
- **StrongSORT** — excellent accuracy but slower; better suited when GPU is available.
- **SORT (baseline)** — no second-pass association; struggles with occlusion.

---

### 4. How ID Consistency Is Maintained

1. **Kalman prediction** — before each frame, each active track predicts its next bounding box based on its velocity state. This compensates for missing detections.

2. **IoU-based matching** — predicted boxes are matched to detected boxes using the Hungarian (Jonker–Volgenant) algorithm, minimising total IoU cost.

3. **Two-stage association** — ByteTrack's key innovation: detections below the main confidence threshold are not discarded. They participate in a second matching pass against tracks that went unmatched in step 2. This is critical for tracking partially occluded players.

4. **Lost track buffer** — tracks that fail to match for up to `TRACK_BUFFER` (30) frames are kept in a "lost" state. If a matching detection appears within that window, the original ID is re-used rather than creating a new one.

5. **Trajectory history** — each track's centroid history is stored for trail visualisation, providing visual verification of ID continuity.

---

### 5. Challenges Faced

| Challenge | Description |
|-----------|-------------|
| Dense groupings | Multiple players in close proximity generate overlapping bounding boxes and increase ID-switch risk |
| Camera motion | Wide-angle pans momentarily shift all predicted positions, temporarily reducing IoU match scores |
| Similar appearance | Players in identical jerseys provide no visual distinction; tracker relies entirely on motion continuity |
| Partial entry/exit | Players entering/exiting frame edges are assigned new IDs even if they have appeared before |
| Frame skip trade-off | Skipping frames speeds up processing but widens the gap between Kalman predictions and actual positions |

---

### 6. Failure Cases Observed

- **Full mutual occlusion:** When two players cross and completely overlap for several frames, their IDs sometimes swap after they separate.
- **Fast camera cuts:** Hard scene cuts (replays, angle changes) destroy all track continuity — every subject gets a new ID.
- **Crowd bleed:** Seated spectators in the foreground can be detected at low confidence, generating spurious short-lived tracks.
- **Long absence:** Players off-camera for more than `TRACK_BUFFER` frames (e.g., substitution) are given a new ID on re-entry.

---

### 7. Possible Improvements

1. **Appearance ReID embeddings (BoT-SORT / StrongSORT):** Adding a lightweight ReID network (e.g., OSNet-x0.25) to compute appearance descriptors per crop would allow re-identification after long occlusion or camera cuts, at the cost of ~10 ms extra per frame.

2. **Domain-specific fine-tuning:** Fine-tuning YOLOv8 on a labelled sports dataset (e.g., SoccerNet, SportsMOT) would improve recall on partially occluded players and reduce false positives from referees/crowd.

3. **Team clustering:** Running K-means (k=2) on HSV histograms of each player's torso crop would label players by team, enabling team-specific tracking statistics.

4. **Homographic projection:** Estimating a homography from field line correspondences would allow plotting player positions on a standard bird's-eye field diagram, useful for tactical analysis.

5. **Speed estimation:** Dividing pixel displacement by a calibration factor (metres per pixel, derived from known field dimensions) and the frame interval gives an approximate speed in m/s.

6. **Evaluation metrics:** Computing MOTA (Multiple Object Tracking Accuracy) and IDF1 against a hand-annotated ground truth on a short clip would enable quantitative comparison between tracker configurations.

---

*Report length: ~650 words (within the 1–2 page guideline)*
