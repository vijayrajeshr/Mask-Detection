# Face Mask Detection Project

This is my project for detecting face masks in real-time. I used OpenCV to find faces and a deep learning model (TensorFlow) to check if the person is wearing a mask.

## How it works
1. **Face Detection:** I used Haar Cascades to find faces in the webcam feed.
2. **Mask Classification:** Each face is cropped and sent to a CNN model (MobileNetV2) to predict if there is a mask.
3. **Display:** I draw green boxes for masks and red boxes if no mask is found.

## Folder Structure
- `main.py`: The main script to run the webcam.
- `detect.py`: Code for finding faces.
- `classify.py`: Code for predicting masks.
- `utils.py`: Some extra functions for drawing and cropping.
- `setup_project.py`: Run this first to download the models.
- `requirements.txt`: Libraries needed (OpenCV, TensorFlow, etc).

## How to run the project

### 1. Install libraries
```bash
pip install -r requirements.txt
```

### 2. Download the models
I didn't include the large model files in git, so run this to get them:
```bash
python setup_project.py
```

### 3. Start the program
```bash
python main.py
```
You can also change the confidence threshold:
```bash
python main.py --conf 60.0
```

## Challenges I faced
- At first, the face detection was a bit slow, so I switched to grayscale.
- Getting the image resizing right (224x224) was important because MobileNet is picky about input size.
- Sometimes the lighting makes it hard to see the mask, but the model is pretty good!

## Author
Vijay Rajesh R

