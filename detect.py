import cv2
import os

# This class handles finding faces in a frame
class FaceDetector:
    def __init__(self, model_path=None):
        # Default face model file
        cascade_name = 'haarcascade_frontalface_default.xml'
        
        if model_path is None:
            # Check if we have the file in our folder
            if os.path.exists(cascade_name):
                model_path = cascade_name
            else:
                # Use the one that comes with opencv
                model_path = cv2.data.haarcascades + cascade_name
        
        if not os.path.exists(model_path):
            print(f"Error: {model_path} not found!")
            return
            
        self.face_cascade = cv2.CascadeClassifier(model_path)

    def detect_faces(self, frame):
        # Change to gray for better detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find the faces
        # scaleFactor 1.1 and minNeighbors 5 are standard values I found online
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        return faces
