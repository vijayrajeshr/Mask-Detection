import cv2
import os

class FaceDetector:
    """
    A class to handle face detection using OpenCV's Haar Cascade.
    """
    def __init__(self, model_path=None):
        """
        Initializes the face detector with a Haar Cascade model.
        Args:
            model_path (str, optional): Path to the haarcascade_frontalface_default.xml file.
                                       If None, it tries to load from the local directory or
                                       OpenCV's default data folder.
        """
        # Use default OpenCV haar cascade if no path provided
        if model_path is None:
            # Check local directory first, then fallback to cv2 default
            cascade_name = 'haarcascade_frontalface_default.xml'
            if os.path.exists(cascade_name):
                model_path = cascade_name
            else:
                model_path = cv2.data.haarcascades + cascade_name
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Haar Cascade model not found at {model_path}")
            
        self.face_cascade = cv2.CascadeClassifier(model_path)

    def detect_faces(self, frame):
        """
        Detects faces in a BGR frame using a grayscale conversion.
        Args:
            frame: The input image/frame from the webcam.
        Returns:
            list: A list of bounding boxes (x, y, w, h) for each detected face.
        """
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detectMultiScale parameters: 
        # scaleFactor=1.1: Compensates for faces appearing smaller/larger.
        # minNeighbors=5: Reduces false positives by requiring consensus.
        # minSize=(30, 30): Minimum size of a detected face.
        faces = self.face_cascade.detectMultiScale(
            gray_frame, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        return faces
