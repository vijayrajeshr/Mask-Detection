import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import os
import cv2

# This class takes a face and says if there's a mask
class MaskClassifier:
    def __init__(self, model_path="mask_detector.h5"):
        # Let's see if the file is there
        if not os.path.exists(model_path):
            print(f"Warning: Model not found at {model_path}!")
            self.model = None
        else:
            # Load the model
            self.model = load_model(model_path, compile=False)

    def predict(self, face_image):
        if self.model is None:
            return ("No Model", 0.0)

        # Preprocessing:
        # Convert BGR to RGB 
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
        # Resize to 224x224 because that's what MobileNet needs
        face_image = cv2.resize(face_image, (224, 224))
        
        # Turn it into a numpy array and scale it
        face_image = img_to_array(face_image)
        face_image = tf.keras.applications.mobilenet_v2.preprocess_input(face_image)
        
        # Add the batch dimension
        face_image = np.expand_dims(face_image, axis=0)

        # Make the prediction
        (mask, withoutMask) = self.model.predict(face_image, verbose=0)[0]
        
        if mask > withoutMask:
            label = "Mask"
            confidence = mask * 100
        else:
            label = "No Mask"
            confidence = withoutMask * 100
        
        return label, confidence
