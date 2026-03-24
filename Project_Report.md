# Project Report: Real-Time Face Mask Detection System

## 1. Project Overview
The objective of this project is to develop a real-time face mask detection system that can identify individuals wearing masks versus those without, using a standard webcam feed. The system provides immediate visual feedback through color-coded bounding boxes and confidence scores. This project demonstrates the application of deep learning models in real-world safety scenarios.

## 2. Problem Statement
The global emphasis on health safety in public indoor spaces has necessitated automated solutions for monitoring mask-wearing compliance. Manual monitoring is neither scalable nor consistently reliable. A computer-vision-based solution offers a non-intrusive, automated, and high-accuracy alternative for real-time monitoring and alerting in environments such as malls, offices, and educational institutions.

## 3. Methodology & Deep Learning Analysis

### 3.1. Face Detection Module
We utilize **OpenCV’s Haar Cascade Classifier** for face localization. 
- **Analysis:** Haar Cascades are based on a series of basic features (like edges, lines, and rectangles) that are used for object detection. They are computationally efficient because they use "Integral Images" to calculate these features rapidly. They are particularly suitable for edge-device processing where high-end GPU resources are limited.

### 3.2. Mask Classification Module
For classification, a deep learning model based on the **MobileNetV2** architecture was employed. 
- **Architecture Analysis:** MobileNetV2 uses depthwise separable convolutions which significantly reduce the number of parameters and computation cost while maintaining high accuracy. This is ideal for real-time video processing.
- **Data Preprocessing:** Each detected face ROI (Region of Interest) is cropped and undergoes:
    1. Conversion to RGB color space.
    2. Resizing to 224x224 pixels (model input shape).
    3. Normalization into the range [-1.0, 1.0].
- **Prediction:** The model utilizes a Softmax layer as the final activation to output probability scores for two classes: "Mask" and "No Mask."

### 3.3. Real-Time Processing Pipeline
The system operates on an infinite frame-by-frame loop:
1. **Frame Capture:** OpenCV reads the video stream buffer.
2. **Face Localization:** The Face Detector scans the frame for feature patterns matches.
3. **Classification Inference:** 
   - Each face is extracted and preprocessed.
   - The TensorFlow/Keras session performs inference on the ROI.
4. **Overlay Visualization:** Results (Label, Percent Confidence, Color-coded Bounding Box) are merged with the original frame.
5. **Human Interaction:** The processed frame is displayed via a GUI window.

## 4. Syllabus Coverage & Course Concepts
This project demonstrates several key concepts covered in the course:
- **Computer Vision Basics:** Image processing, color space mapping, and feature extraction.
- **Deep Learning Foundations:** Using pre-trained Convolutional Neural Networks (CNNs) for classification tasks.
- **Model Deployment:** Deploying a trained model on a local system for inference.
- **Optimization:** balancing detection speed (FPS) with classification accuracy using lightweight architectures like MobileNetV2.

## 5. Implementation Details
- **Programming Language:** Python
- **Key Libraries:** OpenCV-Python, TensorFlow-Keras, NumPy, imutils.
- **Hardware Requirements:** Standard laptop webcam, minimum 4GB RAM.
- **Modular Design:** Separation of concerns between detection, classification, and rendering scripts.

## 6. Performance and Results
The system achieves smooth frames per second (FPS) on standard CPU-only hardware.
- **Accuracy:** The classification model generally provides >90% accuracy.
- **Scalability:** The system successfully handles multiple faces in the frame.

## 7. Reflections and Challenges
During implementation, one of the primary challenges was ensuring the system remained "real-time" on standard hardware. While robust models like ResNet or VGG16 offer higher precision, their computational cost makes them stutter on a live stream. Choosing MobileNetV2 was a critical design decision to ensure the pipeline remained smooth.

Another challenge was handling varying lighting conditions. Future versions of this project could include histogram equalization to improve detection in low-light environments.

## 8. Conclusion
This project successfully demonstrates a real-time, lightweight solution for automated face mask detection, proving the effectiveness of combining classical computer vision with modern deep learning for practical safety applications.
