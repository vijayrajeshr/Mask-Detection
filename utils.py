import cv2

# This function draws the box and the label on the frame
def draw_prediction(frame, face_coords, label, confidence, threshold=50):
    
    # Only show if we are sure enough
    if confidence < threshold:
        return frame

    x, y, w, h = face_coords
    
    # Green for yes, red for no
    if label == "Mask":
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)
    
    # Draw the rectangle
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    
    # Put the text on top
    text = f"{label}: {confidence:.2f}%"
    cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    
    return frame

# This cuts the face out from the main frame
def preprocess_face(frame, coords):
    x, y, w, h = coords
    
    # Make sure we stay within the image borders
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(frame.shape[1], x + w)
    y2 = min(frame.shape[0], y + h)

    # Crop it
    face = frame[y1:y2, x1:x2]
    return face

