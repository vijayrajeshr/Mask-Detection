import cv2
import time
import os
import argparse
# Import my other files
from detect import FaceDetector
from classify import MaskClassifier
from utils import draw_prediction, preprocess_face

def run_mask_detection(model_path="mask_detector.h5", confidence_threshold=50):
    
    print("Starting the video stream... Please wait")
    # 0 is usually the default webcam
    cap = cv2.VideoCapture(0) 
    
    # Load my detector and classifier classes
    face_detector = FaceDetector()
    mask_classifier = MaskClassifier(model_path=model_path)
    
    # Give the camera some time to start
    time.sleep(2.0)

    try:
        while True:
            # Get the next frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read from webcam.")
                break

            # Find faces in the current frame
            faces = face_detector.detect_faces(frame)

            # Check each face found
            for (x, y, w, h) in faces:
                # Get the face part and resize it for the model
                face_img = preprocess_face(frame, (x, y, w, h))
                
                # Predict if they have a mask or not
                label, confidence = mask_classifier.predict(face_img)

                # Draw the box and label on the screen
                frame = draw_prediction(frame, (x, y, w, h), label, confidence, threshold=confidence_threshold)

            # Display the result
            cv2.imshow("Mask Detection System", frame)

            # Press 'q' to stop the program
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as e:
        print(f"Something went wrong: {e}")
    finally:
        # Stop everything and close windows
        print("Closing the system...")
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Setup some basic arguments so I can change settings from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mask_detector.h5")
    parser.add_argument("--conf", type=float, default=50.0)
    args = parser.parse_args()

    run_mask_detection(model_path=args.model, confidence_threshold=args.conf)

