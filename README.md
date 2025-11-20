#  IronID — AI-Powered Gym Assistant

> A hybrid AI mobile app that identifies gym equipment offline and provides personalized workout guidance online.

---

## 🚀 Overview

**IronID Trainer** is designed to eliminate *gym anxiety* for beginners by helping them understand and properly use gym equipment without relying on trainers, labels, or prior experience.

The app combines:

- **Offline Computer Vision** — Detects gym machines directly on-device using TensorFlow Lite.
- **Online AI Coach** — Provides tailored workout routines, tips.

All vision processing is fully private and never leaves the device.

---

## 🔍 Problem

Beginners often feel overwhelmed in the gym due to:

- Lack of guidance
- Confusing machine usage
- Risk of injury from poor form
- Fear of looking inexperienced

Traditional trainers are not always available, and online guides lack personalization.

---

## ✔️ Solution

IronID provides a **dual-system AI pipeline**:

| Component | Description |
|----------|------------|
| **Vision Model (Offline)** | Identifies equipment from camera input without internet |
| **AI Chatbot (Online)** | Generates routines, answers questions, workout plan for beginners |

This ensures **instant recognition + personalized guidance**—anywhere.

---

## Features

### **1. Snap & Identify (Offline)**
- Recognizes fitness machines in real-time
- Runs fully on-device using TFLite
- Shows usage tips + quick tutorial videos
- Privacy-first (no uploads)

### **2. AI Workout Coach (Online)**
Ask natural language questions like:
**Example Flow**

- **🤵:** My weight is 123 KG
- **🤖:** Noted — your current weight is 123 KG.
- **🤵:** My goal is Fat Loss
- **🤖:** Generating a personalized weekly plan.

**Sample Output (Week Overview)**

- **Monday — Full Body Circuit**  
   Squats, Push-ups, Rows, Burpees — 3×12
  
- **Tuesday — Cardio & Core**  
   30-min HIIT, Planks, Crunches — 3×10
  
- **Wednesday — Upper Body Strength**  
   Bench Press, Pull-ups, Shoulder Press — 3×12

---

## Supported Equipment

The current model detects **10 classes**:

| Class | Example |
|-------|---------|
| Barbell | Deadlifts, squats |
| Bench Press | Flat/incline/decline |
| Dumbbell | Free weights |
| Kettlebell | Swings, snatches |
| Leg Press | Lower body machine |
| Punching Bag | Boxing equipment |
| Ab Roller | Core training |
| Stationary Bicycle | Cardio equipment |
| Step Platform | Aerobic workouts |
| Treadmill | Running/walking |

Dataset is custom-curated and labeled.

---

## Section 1: Data Strategy & Architecture Design

### 1.1 Identification and Justification of Data Sources

**Public image datasets:**  
We will supplement our dataset with open-source images from Kaggle 

**Justification:**  
This hybrid approach provides diverse and realistic coverage of the most common gym machine types while remaining ethical and privacy-safe.

---

### **1.2 Data Preprocessing Pipeline**

#### **1.2.1 Dataset Loading**
- Loaded using `image_dataset_from_directory()`
- Automatic label assignment based on folder structure
- Batch size: **32**
- Target resolution: **224×224×3**
- Shuffled only during training

#### **1.2.2 Data Normalization**
- Converted to RGB if needed
- Normalized using model-specific preprocessing:
  - `mobilenet_v3.preprocess_input` (MobileNet models)
  - `efficientnet.preprocess_input` (EfficientNet models)
- Scales pixel values to normalized float ranges


#### **1.2.3 Data Structure**

📂 Dataset/  
├ 📁 train/  
├ 📁 val/  
└ 📁 test/  

- Clean separation of evaluation data  
- Prevents data leakage across splits


#### **1.2.4 Data Integrity & Cleaning**
- Removed corrupted/empty files
- Excluded images with people, posters, or diagrams
- Discarded images where equipment <40% of frame
- Manually verified folder labels

#### **1.2.5 Augmentation (Training Only)**
Improves robustness across lighting + angles:

- Rotation (±20°)
- Random zoom (±15%)
- Horizontal flip
- Brightness/contrast changes
- Mild blur & noise
- Perspective distortion


#### **1.2.6 Train/Val/Test Split**
- **Train:** 80%  
- **Validation:** 10%  
- **Test:** 10%  
- Stratified by class
- Split before augmentation to prevent duplicates across sets

---
## Section 2: Model Architecture & Training

### **2.1 Model Architecture**

#### **2.1.1 Base Network**
- MobileNetV3-Large pretrained on ImageNet
- Top removed, frozen as feature extractor
- Optimized for fast mobile inference

#### **2.1.2 Custom Classification Head**
- Global Average Pooling
- Softmax layer for 10 classes
- Lightweight and TFLite-ready
---
### **2.2 Training Configuration**

#### **2.2.1 Training Info**
- Loss: Sparse Categorical Crossentropy
- batch size 32
- 10 epochs (transfer learning)

#### **2.2.2 Evaluation Pipeline**
- Run predictions using `model.predict()`
- Evaluate using accuracy + F1/precision/recall
- Visualize results with Confusion matrix

#### **2.2.3 Model Selection Summary**
| Model | Size | Accuracy | Verdict |
|-------|------|----------|---------|
| MobileNetV3-Small | 3.6 MB | 92% | Fast, weak performance |
| EfficientNet-B0 | 15.5 MB | 99% | High accuracy, too heavy |
| **MobileNetV3-Large** | **11.5 MB** | **97%** | **Selected (best balance)** |

---
## Section 3: Optimization (Pruning, Quantization, Distillation)
### **3.1 Pruning**
- Polynomial decay pruning
- Final sparsity: **50%**
- Applied to Dense + Conv layers

→ Reduced computation cost before quantization.
---
### **3.2 Quantization Outputs**

| Format | Size | Accuracy |
|--------|------|-------|
| Float32 TFLite | ~11MB | 96.08% |
| Float16 | ~6MB | 96.08% |
| Int8 Dynamic | ~3MB | 97.06% |
| Full Int8 | ~3MB | 87.25% |

Final deployed model:  
**`model_pruned_int8_dynamic.tflite`**
---
### **3.3 Knowledge Distillation**
- Teacher: EfficientNet-B0
- Student: MobileNetV3-Small
- Temperature: **3**
- Alpha: **0.5**

| Model | Accuracy | Size |
|--------|----------|------|
| Student Baseline | 92% | 3.6MB |
| **Distilled Student** | **95%** | 3.6MB |


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
 ├─ Barbell/
 ├─ Bench Press/
 ├─ Dumbbell/
 ├─ Kettlebell/
 ├─ Leg Press/
 ├─ Punching Bag/
 ├─ Ab Roller/
 ├─ Stationary Bicycle/
 ├─ Step Platform/
 └─ Treadmill/
```

---

### Class Labels
```bash
Barbell  
Bench Press  
Dumbbell  
Kettlebell  
Leg Press  
Punching Bag  
Ab Roller  
Stationary Bicycle  
Step Platform  
Treadmill  

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
