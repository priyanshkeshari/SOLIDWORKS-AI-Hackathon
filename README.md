# SOLIDWORKS-AI-Hackathon
SOLIDWORKS AI Hackathon, hosted at IIT Madras Mission in this competition is to build a machine learning model that can correctly identify which mechanical parts appear in a synthetic image.


---

## 🎭 Model Architecture

The final solution is a **modular YOLOv8m-based counting system** with **test-time augmentation (TTA)** and **median aggregation**, specifically designed to achieve robust, integer‑accurate counts under the exact‑match metric.[conversation_history:1]

### 🔹 Evolution of Architectures

1. **ResNet50 Regression**
   - Image → ResNet50 → FC(4) → continuous counts → rounded to integers  
   - Score: **0.9750**  
   - Issue: No localization; struggled with overlapping parts and dense scenes.[conversation_history:1]

2. **ResNet50 Multi‑Head Classification**
   - Shared backbone: ResNet50 (FC removed)  
   - Four independent heads (bolt, pin, nut, washer), each classifying count ∈ {0,…,4}  
   - Score: **0.9921**  
   - Issue: Still blind to spatial structure; confusion in overlapping objects.[conversation_history:1]

3. **YOLOv8m – Base Detector**
   - Standard object detection: Image → YOLOv8m → detections → count per class  
   - Training: 20 epochs, batch 16, image size 640, NVIDIA P100 GPU.[conversation_history:1]  
   - Score: **0.9986**  
   - Breakthrough: Explicit localization solved most errors.

4. **YOLOv8m + naïve TTA**
   - Used `augment=True`, `conf=0.2`, `iou=0.6` directly in Ultralytics inference.  
   - Score dropped to **0.9929** due to extra false positives (“ghost” parts) from aggressive augmentation.[conversation_history:1]

5. **YOLOv8m + Median Voting (Final)**
   - Custom TTA + median aggregation + class‑wise thresholds + domain logic.  
   - Score: **1.0000** – **perfect exact‑match accuracy** on leaderboard.[conversation_history:1]

---

## 🧱 Final Model Design

### 1️⃣ Backbone: YOLOv8m

- **Task:** Detection (`task=detect`)  
- **Hyperparameters (key):**
  - `epochs=20`, `batch=16`, `imgsz=640`
  - `lr0=0.01`, `momentum=0.937`, `weight_decay=0.0005`
  - `hsv_s=0.7`, `hsv_v=0.4`, `mosaic=1.0`, `fliplr=0.5`[conversation_history:1]  
- Trained on `yolov8m.pt` pretrained weights with full box + class supervision.

### 2️⃣ Robust Inference: TTA + Median Aggregation

For each test image:

1. **Generate 3 Views**
   - Original image  
   - Horizontal flip (`cv2.flip(img, 1)`)  
   - Vertical flip (`cv2.flip(img, 0)`)

2. **Per‑View Detection**
   - Run YOLOv8m independently on each view (`iou=0.6`, default NMS).
   - Apply **per‑class confidence thresholds** (tuned via error analysis):
     - `bolt`: **0.50** – precise, avoids elongated artifacts  
     - `locatingpin`: **0.35** – intermediate threshold  
     - `nut`: **0.35** – intermediate threshold  
     - `washer`: **0.15** – permissive to recover low‑contrast thin washers[conversation_history:1]

3. **Per‑View Logic Constraint**
   - Enforce **maximum of 4** per part type, based on dataset statistics:  
     `count[class] = min(count[class], 4)`.[conversation_history:1]

4. **Median Voting Across Views**
   - For each class, collect 3 counts `[c₁, c₂, c₃]` from the three views.  
   - Final count = `median(c₁, c₂, c₃)` (integer).  
   - Rationale:
     - **Mean** is sensitive to outliers (one over‑detection inflates result).  
     - **Max** is optimistic and tends to over‑count.  
     - **Median** serves as a **majority vote** for discrete counts; robust to one bad view.

> “With three augmented views, median aggregation acts as a majority‑vote mechanism for counts, improving robustness without introducing over‑count bias.”[conversation_history:1]

---

## 🌐 Features

Key characteristics of the solution:

- **Bounding‑Box–Aware Training:** Uses full box supervision for high‑precision localization.
- **Model‑Agnostic Dataset Pipeline:** CSV → YOLO format is implemented generically, so any YOLO variant (v8, v9, v10) can be swapped in.
- **Flexible Inference Modes:**
  - Standard single‑pass YOLO inference (fast, near‑perfect).  
  - Custom TTA + Median Voting (slower but maximally robust).
- **Domain Constraints:** Hard caps on maximum counts per class (≤4) remove impossible predictions.
- **Research‑Grade Error Analysis:** Per‑class thresholds and median aggregation were derived through manual inspection of failure modes, not blind grid search.[conversation_history:1]

---

## 📏 Evaluation

### Metric: Exact‑Match Accuracy

For each image, the model outputs four integers:

\[
(\hat{b}, \hat{p}, \hat{n}, \hat{w})
\]

The prediction is **correct** only if:

\[
(\hat{b}, \hat{p}, \hat{n}, \hat{w}) = (b, p, n, w)
\]

Final score = (#exactly correct images) / (total test images).

- ResNet Regression: **0.9750**  
- ResNet Multi‑Head: **0.9921**  
- YOLOv8m Base: **0.9986**  
- YOLOv8m + naïve TTA: **0.9929**  
- **YOLOv8m + Median Voting: 1.0000 (Perfect)**[conversation_history:1]

---

## 🔄 Workflow

### 1️⃣ Data Preparation

1. Load `train_bboxes.csv` with columns: `image_name, class, x_min, y_min, x_max, y_max`.  
2. Convert each bounding box to YOLO format `(x_center, y_center, w, h)` normalized by image width/height.  
3. Write one `.txt` file per image with lines:


