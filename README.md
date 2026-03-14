# PARKXPLORE

## 🚗 Project Overview
ParkXplore is an AI-driven smart parking system designed to identify available parking spaces in unmarked and unstructured environments. Unlike traditional solutions that depend on fixed sensors or painted slot lines, ParkXplore combines YOLO vehicle detection with point-based slot mapping to determine occupancy in real time

## 🔍 Key Features
- **Boundary-Free Detection:** Detects parking availability without relying on physical slot markings.
- **Real-Time Monitoring:** Processes live video frames to provide instant parking status.
- **High Accuracy:** Uses YOLOv11 for robust vehicle detection in challenging scenes.
- **Hardware Agnostic:** Works with standard 720p HD cameras and supports mobile/monitor visualization.

## 🧠 Technical Architecture
The pipeline converts raw video frames into actionable parking occupancy status:
1. **Camera Input:** Captures live video from a parking area.
2. **Vehicle Detection:** Runs YOLOv11 to detect vehicles and produce bounding boxes.
3. **Slot Mapping:** Stores predefined slot coordinates in `parking_slots.json` using a custom mapping tool.
4. **Occupancy Analysis:** Uses point-based computation to determine which slots are occupied.
5. **Visualization:** Displays total slots, occupied slots, and free slots in real time.

## 🧩 Software & Libraries
- **Models:** Ultralytics YOLOv11
- **Framework:** PyTorch
- **Libraries:** OpenCV, NumPy, Albumentations
- **Dataset Tools:** Roboflow (for dataset labeling and export)

## 📁 Repository Structure
- `DA/` – Data augmentation dataset and YAML files
- `dataset/` – Train/valid/test images and labels
- `output/` – Inference outputs
- `runs/` – Training run logs, weights, metrics
- `parking_slots.json` - Parking slots coordinate mapping data
- `check_labels.py`, `check_occupancy.py`, `mark_slots.py` – Data integrity and slot tools
- `finetune.py`, `testmodel.py` – Model training and testing scripts
- `preprocess_and_augment.py` – Data preprocessing and augmentation pipeline

## ▶️ Quick Start
1. Install dependencies (PyTorch, OpenCV, Ultralytics YOLO).
2. Prepare dataset using `dataset/data.yaml` or `DA/data.yaml`.
3. Map parking slots with `mark_slots.py` and save to `parking_slots.json`.
4. Train YOLOv11 with `finetune.py`.
5. Run occupancy inference with your chosen script (e.g., `testmodel.py`).

## 📌 Notes
- Keep camera perspective stable for consistent slot mapping.
- Validate label quality using `check_labels.py` before retraining.
- Use `check_occupancy.py` to verify occupancy logic on test frames.

## 💡 Contact
For questions or improvements, open an issue or pull request in this repo.
