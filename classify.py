import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import os

class MaskClassifier:
    """
    A class to handle mask classification using a pre-trained Keras model.
    """
    def __init__(self, model_path="mask_detector.h5"):
        """
        Initializes the mask classifier with a pre-trained Keras model.
        Args:
            model_path (str): Path to the .h5 or SavedModel directory.
        """
        if not os.path.exists(model_path):
            print(f"[WARNING] Model not found at {model_path}. Please run setup_project.py.")
            self.model = None
        else:
            # We load the model once to avoid re-initializing the prediction session
            self.model = load_model(model_path, compile=False)

    def predict(self, face_image):
        """
        Extracts features from a face image and uses the model to predict the class.
        Args:
            face_image (numpy array): Input face region from the BGR frame.
        Returns:
            tuple: (label: str, confidence_score: float)
        """
        if self.model is None:
            # Fallback label if the model isn't loaded correctly
            return ("No Model Loaded", 0.0)

        # Preprocessing Workflow:
        # 1. Color Space: Convert from BGR back to RGB for model compatibility.
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        # 2. Geometry: Models require a fixed input size (224x224 for MobileNetV2).
        face_image = cv2.resize(face_image, (224, 224))
        # 3. Normalization: mobilenet_v2.preprocess_input scales data to [-1, 1].
        face_image = img_to_array(face_image)
        face_image = tf.keras.applications.mobilenet_v2.preprocess_input(face_image)
        # 4. Dimension Expansion: Inference requires a batch dimension (1, 224, 224, 3).
        face_image = np.expand_dims(face_image, axis=0)

        # Model Inference
        (mask, withoutMask) = self.model.predict(face_image)[0]
        
        # Determine the label based on the highest probability component
        label = "Mask" if mask > withoutMask else "No Mask"
        confidence = max(mask, withoutMask) * 100
        
        return label, confidence

# Note: Importing cv2 here to ensure preprocess works, but keeping it inside method
import cv2 
