# Real-Time Face Mask Detection System

A lightweight, real-time computer vision system that detects faces and classifies whether each person is wearing a mask or not.

## Features
- **Face Detection:** Uses OpenCV's Haar Cascade for efficient, real-time face localization.
- **Mask Classification:** Uses a pre-trained CNN model (MobileNetV2 based) to classify mask usage.
- **Real-Time Visualization:** Draws bounding boxes (GREEN for Mask, RED for No Mask) with confidence scores.
- **Modular Pipeline:** Cleanly separated modules for detection, classification, and utilities.

## Folder Structure
```text
Mask-Detection/
├── main.py            # Entry point for webcam stream
├── detect.py          # Face detection module
├── classify.py        # Mask classification module
├── utils.py           # Preprocessing and visualization utilities
├── setup_project.py   # Script to download necessary pre-trained models
├── requirements.txt   # Required Python libraries
└── README.md          # Project documentation
```

## Setup Instructions

### 1. Prerequisites
- Python 3.7+
- A working webcam

### 2. Environment Setup
Clone the repository (if not already done) and install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Model Downloads
Run the following script to automatically download the pre-trained face and mask detector models:
```bash
python setup_project.py
```
*Note: This will download `haarcascade_frontalface_default.xml` and `mask_detector.h5` into your project directory.*

## Running the System

To start the real-time detection via your webcam, simply run:
```bash
python main.py
```

### Optional Arguments
You can customize the detection threshold or the model path using command line arguments:
```bash
python main.py --confidence 60.0 --model mask_detector.h5
```

## Usage Controls
- Press **'Q'** on your keyboard to exit the application window.

## Author
Vijay Rajesh R

