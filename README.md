# 🚦 Traffic Violation Detection System

> **Real-time Traffic Violation Detection System** utilizing YOLO Object Detection and CLAHE Image Enhancement.

---

## 📋 Project Overview

| Details | Description |
|-----------|----------|
| **Project Name** | Real-time Traffic Violation Detection System Based on YOLO26n Architecture and Multi-stage Image Enhancement |
| **Objective** | Detect red light violations from traffic camera video footage |
| **AI Technologies** | YOLOv8/YOLO26, BotSORT Tracking |
| **Image Enhancement** | Double-pass CLAHE for License Plate visibility |
| **Dataset** | 5 classes: Stop line, Green/Red/Yellow light, Sub light |

---

## 🏗️ System Architecture

1. **YOLO Traffic Light Model**: Detects the state of traffic lights and stop lines.
2. **YOLO Vehicle Detection**: Detects vehicles and assigns unique IDs using BotSORT tracking.
3. **Rule-based Logic**: Determines a Violation if a vehicle crosses the stop line during a red or yellow light.
4. **License Plate Detection**: Crops the violating vehicle, detects the license plate, and enhances it using Double CLAHE.

---

## 📁 Repository Structure

This GitHub version is a lightweight repository, containing only the core Source Code and sample data.

```text
violation_detection/
├── Code/                                 # Entire project source code
│   ├── detection/                        # Main violation detection pipeline (Final_RUN.py)
│   ├── data_processing/                  # Scripts for data handling, auto-labeling, cleaning
│   ├── validation/                       # Scripts for checking model/data accuracy
│   └── training/                         # Model training scripts
│
├── data_samples/                         # Sample image data and labels (format reference)
│   ├── images/                           
│   └── labels/                           
│
├── Model/weights/                        # Placeholder for traffic light detection weights (.pt)
├── LP_v26/model/plate_model/weights/     # Placeholder for license plate detection weights (.pt)
│
├── dataset_samples.png                   # Sample dataset visualization
├── .gitignore                            # Ignoring non-essential files on Git
└── README.md                             # This documentation file
```

> **Note:** The fully trained models (e.g., `best.pt`) and the full large-scale dataset have been excluded from this repository to keep the remote size small. Please place your `.pt` weights into the corresponding placeholder folders before running.

---

## 🚀 Getting Started

### Requirements
- Python 3.9+
- A CUDA-enabled GPU is highly recommended for real-time inference.

1. **Install dependencies:**
   ```bash
   pip install ultralytics opencv-python numpy tqdm albumentations pillow easyocr
   ```

2. **Run Detection Pipeline (offline video):**
   Open `Code/detection/Final_RUN.py` to configure your neural network weights and input video paths, then execute:
   ```bash
   cd Code/detection
   python Final_RUN.py
   ```

3. **Train Model from Scratch:**
   Modify the configurations inside `Code/train.py` and run:
   ```bash
   cd Code
   python train.py
   ```

---

## ⚠️ Current Limitations
- **No License Plate Recognition (OCR):** While the system correctly detects, crops, and enhances license plates using YOLO and CLAHE, it does not currently feature an Optical Character Recognition (OCR) module to extract the vehicle's alphanumeric text automatically.

---

## 👥 Contributions
- **Main Pipeline** is located in `Code/detection/`.
- Please ensure large files (videos, zip files, or heavy weights) are ignored via `.gitignore` before pushing.
