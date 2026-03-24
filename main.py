import cv2
import time
import os
import argparse
from detect import FaceDetector
from classify import MaskClassifier
from utils import draw_prediction, preprocess_face

def run_webcam(model_path="mask_detector.h5", confidence_threshold=50):
    """
    Main loop for webcam processing.
    """
    print("[INFO] Starting video stream...")
    vs = cv2.VideoCapture(0) # Standard webcam
    
    # Initialize detector and classifier
    detector = FaceDetector()
    classifier = MaskClassifier(model_path=model_path)
    
    # Allow camera sensor to warm up
    time.sleep(2.0)

    while True:
        # Read frame from stream
        ret, frame = vs.read()
        if not ret:
            print("[ERROR] Failed to read frame from webcam.")
            break

        # Detect faces in frame
        faces = detector.detect_faces(frame)

        # Iterate over each face
        for (x, y, w, h) in faces:
            # Crop face for classification
            face_img = preprocess_face(frame, (x, y, w, h))
            
            # Predict mask/no mask
            label, confidence = classifier.predict(face_img)

            # Draw prediction result on frame
            frame = draw_prediction(frame, (x, y, w, h), label, confidence, threshold=confidence_threshold)

        # Show the output frame
        cv2.imshow("Mask Detection System", frame)

        # Check for 'q' key to quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # Clean up
    print("[INFO] Stopping system...")
    cv2.destroyAllWindows()
    vs.release()

if __name__ == "__main__":
    # Add command line arguments for configurability
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="mask_detector.h5",
        help="path to trained face mask detector model")
    parser.add_argument("-c", "--confidence", type=float, default=50.0,
        help="minimum confidence threshold to filter weak detections")
    args = vars(parser.parse_args())

    run_webcam(model_path=args["model"], confidence_threshold=args["confidence"])
