# IronID Project

### Smart Gym Equipment Recognition System  
A deep learning system for recognizing gym equipment from real-world photos using **MobileNetV3-Large**, optimized for mobile inference and offline functionality.

---

## Section 1: Data Strategy & Architecture Design

### 1.1 Identification and Justification of Data Sources

**Self-captured images:**  
We will capture photos of real gym machines using different mobile devices and lighting conditions. Each image will be manually labeled with its correct machine name (e.g., *Leg Press*, *Lat Pulldown*, *Chest Press*).  
This ensures realistic, privacy-compliant data that represents real gym environments.

**Public image datasets:**  
We will supplement our dataset with open-source images from:
1. [Gym Equipment Dataset 1](https://www.kaggle.com/datasets/rifqilukmansyah381/gym-equipment) — Dumbbells, Elliptical Machines, Recumbent Bikes  
2. [Gym Equipment Dataset 2](https://www.kaggle.com/datasets/aadarshvelu/gym-equipements-classification) — BenchPress, DumbBell, KettleBell, PullBar, TreadMill  
3. [Gym Tools Dataset](https://www.kaggle.com/datasets/teguhbudi/gym-tools) — Barbell, Dumbell, Leg Press, Treadmill, etc.  
4. [Gym Equipment Dataset 3](https://www.kaggle.com/datasets/dutt2302/gym-equipment) — Rowing Machine, Multi Machine, Treadmill  
5. [Gym Equipment Image Set](https://www.kaggle.com/datasets/rifqilukmansyah381/gym-equipment-image/data) — Smith Machine, Bench Press, Elliptical, Recumbent Bike  

**Justification:**  
This hybrid approach provides diverse and realistic coverage of the most common gym machine types while remaining ethical and privacy-safe.

---

### 1.2 Data Collection and Cleaning Plan

#### A) Image Capture and Scraping Protocol
- **Devices:** Smartphones from team members.  
- **Angles:** Front, left, right, back, top (if possible) — each at multiple distances.  
- **Lighting:** Bright/dim, daylight/warm LED, with/without mirrors.  
- **Target:** ~100–150 images per class.  
- **Naming Convention:** `class_device_initials_index.jpg` (e.g., `leg_press_s23_MA_0041.jpg`).

#### B) Data Cleaning
- Remove diagrams, infographics, or photos with people.
- Skip unreadable files; rotate and strip EXIF data for privacy.
- Convert all to `.jpg` (quality 90), RGB color.
- Remove blurry or duplicate images (using pHash/dHash & SSIM).
- Crop or discard images where the machine occupies <40% of the frame.
- Normalize lighting with CLAHE or white balance if needed.
- Verify label-content consistency.

#### C) Data Preprocessing
- Resize → 224×224 pixels.
- Normalize → `[0, 1]` scale (model-specific if needed).
- Augment underrepresented classes to maintain balance.

#### D) Data Augmentation (Training Only)
- Random rotation (±20°), flip, zoom (±15%), perspective warp.
- Adjust brightness, contrast, add slight blur/noise.
- Applied only to training data.

#### E) Splits
- **Train:** 80%  
- **Validation:** 10%  
- **Test:** 10%  
(Stratified by class; duplicates remain within one split.)

---

### Data Preprocessing Summary

- Resized all images to **224×224 RGB**.
- Normalized pixel intensities to `[0,1]`.
- Augmented data to simulate real-world variation:
  - Rotation, flip, zoom, brightness/contrast, Gaussian blur, perspective.
- Maintained class balance.

---

## Section 2: Detailed System Design

### Logical Architecture
**UI (Mobile App):**
- Capture or upload gym equipment photos.
- Display classification results and confidence scores.
- Provide bilingual guidance content.
- Works fully offline for privacy and reliability.

---

### Model Training & Experimentation

**Preprocessing Steps:**
- Resize: 224×224 pixels  
- Crop: Center-crop for focus  
- Normalize: Scale pixels to `[0–1]` or `[-1–1]`

**Model:**
- **Architecture:** MobileNetV3-Large (INT8 quantized)
- **Framework:** TensorFlow / Keras
- **Optimizer:** Adam
- **Loss Function:** Categorical Cross-Entropy
- **Metrics:** Accuracy, Precision, Recall, F1-score
- **Early stopping** and **learning rate scheduling** used.

**Deployment:**
- Export best model as TensorFlow Lite (TFLite)
- Optimized for mobile CPUs (inference <200 ms)
- Offline operation with update support through API Gateway.

---

### API Gateway & Model Inference Service

**Core Functionality:**
- Receives image → preprocess (resize, normalize)
- Runs MobileNetV3-Large TFLite inference
- Returns top predicted label + confidence score

**Post-processing:**
- Confidence thresholding  
- Sends result to UI  
- Triggers usage guide display  

**Performance:**
- Inference latency: **<200 ms**  
- Fully offline operation  
- Model updates available via API Gateway

---

### Data Pipeline Overview
```bash
Data Sources
↓
Main Folder (/dataset/)
↓
Class Folders (/dataset/<class_name>/)
↓
Filtering & Cleaning
↓
Preprocessing & Augmentation
↓
Train / Val / Test Split
↓
Versioned Dataset (v1.0)
↓
Model Training
```

#### **Dataset structure example:**
```bash 
/dataset/
 ├─ leg_press/
 ├─ treadmill/
 ├─ bench_press/
 ├─ pull_bar/
 ├─ elliptical/
 ├─ recumbent_bike/
 ├─ rowing_machine/
 ├─ smith_machine/
 └─ static_bicycle/
```


---

## Section 3: Model Development Plan

### Baseline Model Choice
**Model:** MobileNetV3-Large (pre-trained on ImageNet)  
**Justification:**
- Optimized for **mobile and embedded devices**
- Compact (≈21 MB FP32, ≤10 MB INT8 quantized)
- Excellent accuracy-speed tradeoff
- Supports **transfer learning** → less data, faster convergence

---

### Experimentation Plan & Evaluation Metrics

| Metric Type | Metric | Target |
|--------------|---------|--------|
| **Model** | Top-1 Accuracy | ≥ 80% |
| **Model** | Confusion Matrix | To track look-alike misclassifications |
| **System** | Model Size | ≤ 25 MB |
| **System** | Inference Latency | ≤ 200 ms |

---

### Class Labels
```bash
Leg Press
Static Bicycle
Treadmill
Bench Press
Pull Bar
Elliptical Machine
Recumbent Bike
Rowing Machine
Smith Machine
```

---

## Running the Project

### Install Dependencies
```bash
pip install -r requirements.txt 
```
### Train the Model
```bash
python src/train.py
```
```bash
python src/evaluate.py
```
### Final Output
The trained model will be saved under /models/ (if configured).
Evaluation script prints final test accuracy and confusion matrix.

### Final Test Accuracy
Final Test Accuracy: [Add after running evaluation]

### Repository Structure
```bash
IronID_Project/
├── .gitignore
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_model_prototyping.ipynb
└── src/
    ├── __init__.py
    ├── data_loader.py
    ├── model.py
    ├── train.py
    └── evaluate.py
```
