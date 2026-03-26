import cv2

def draw_prediction(frame, face_coords, label, confidence, threshold=50):
    
    if confidence < threshold:
        return frame

    x, y, w, h = face_coords
    
    # Choose color based on label (Green for mask, Red for no mask)
    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
    
    # Draw rectangle around face
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    
    # Prepare text for label and confidence
    text = f"{label}: {confidence:.2f}%"
    
    # Draw label background
    cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    
    return frame

def preprocess_face(frame, face_coords):
    
    x, y, w, h = face_coords
    # Ensure cropping doesn't go out of bounds
    startX = max(0, x)
    startY = max(0, y)
    endX = min(frame.shape[1], x + w)
    endY = min(frame.shape[0], y + h)

    face = frame[startY:endY, startX:endX]
    return face
