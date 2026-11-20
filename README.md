# Helmet Detection Using YOLOv8

## Overview
A computer vision-based project that detects helmet usage among two-wheeler riders from traffic video footage. The system uses YOLOv8 for object detection to identify riders and classify helmet usage, helping improve road safety monitoring and identify potential traffic violations.

## Problem Statement
Monitoring helmet compliance manually across busy roads is time-consuming and difficult. This project explores an automated approach to detect riders wearing helmets and those riding without helmets using deep learning and object detection.

## Technologies Used
- **Python** – Core programming language
- **YOLOv8 (Ultralytics)** – Object detection model
- **OpenCV** – Video processing and frame analysis
- **PyTorch** – Deep learning framework

## How It Works
1. **Video Input:** Traffic video footage is provided as input.
2. **Rider Detection:** A pretrained YOLOv8 model detects people and motorcycles.
3. **Helmet Detection:** A custom-trained YOLOv8 model identifies helmet-related classes.
4. **Classification:** Riders are analyzed based on helmet usage.
5. **Visualization:** Detection results are displayed using bounding boxes and labels on video frames.

## Model Details
The project uses a two-stage detection approach:

- **Pretrained Model (`yolov8n.pt`):** Used to detect persons and motorcycles.
- **Custom-Trained Model:** Trained on a helmet detection dataset to identify:
  - Number Plate
  - Face With No Helmet
  - Face With Good Helmet
  - Face With Bad Helmet
  - Rider

### Training Configuration
| Parameter | Value |
|---|---|
| Model | YOLOv8 |
| Model Variant | YOLOv8n |
| Epochs | 25 |
| Batch Size | 16 |
| Image Size | 640 × 640 |
| Pretrained Weights | Enabled |
| Training Device | CPU |

## Key Features
- Automated detection of two-wheeler riders.
- Helmet and non-helmet detection.
- Video-based object detection and visualization.
- Custom-trained object detection model.
- Integration of pretrained and custom-trained YOLOv8 models.

## Potential Improvements
- Improve detection performance in crowded traffic scenes.
- Handle occlusion and overlapping riders.
- Extend detection to distinguish between rider and pillion passenger helmet compliance.
- Integrate real-time CCTV feeds for continuous traffic monitoring.
- Generate violation statistics and automated reports.

## Project Structure
```text
Helmet-Detection-using-YOLO/
│
├── dataset/
├── runs/
│   └── detect/
│       └── train/
│           └── weights/
│               └── best.pt
│
├── dataset.yaml
├── main.py
├── requirements.txt
└── README.md
```

*Note: The structure above is illustrative. Adjust it to match the actual files in your repository.*

## Installation and Execution

### 1. Clone the repository
```bash
git clone https://github.com/PrathiAmrutha/Helmet-Detection-using-YOLO.git
cd Helmet-Detection-using-YOLO
```

### 2. Install dependencies
```bash
pip install ultralytics opencv-python torch
```

### 3. Run the project
Run the main detection script:
```bash
python main.py
```

*Ensure that the required model weights and input video are available at the paths specified in the script.*

## Applications
- Traffic surveillance systems.
- Road safety monitoring.
- Automated helmet compliance analysis.
- Smart city traffic management.

## Future Scope
The system can be extended into a real-time traffic monitoring solution with automated violation tracking, rider-wise helmet compliance analysis, and integration with intelligent transportation systems.
