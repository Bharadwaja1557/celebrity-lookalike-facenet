import cv2
import numpy as np

def extract_face_from_array(image, detector, required_size=(160, 160)):
    """
    image: OpenCV BGR image
    detector: MTCNN detector
    returns: cropped and resized face array or None
    """
    # Convert BGR -> RGB for MTCNN
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces
    results = detector.detect_faces(image_rgb)
    if len(results) == 0:
        return None  # No face detected

    # Take the first detected face
    x, y, width, height = results[0]['box']

    # Sometimes MTCNN returns negative coords
    x, y = max(0, x), max(0, y)

    face = image_rgb[y:y+height, x:x+width]

    if face.size == 0:
        return None

    # Resize safely
    try:
        face = cv2.resize(face, required_size)
    except cv2.error:
        return None

    # Normalize to 0-1 float32
    face = face.astype('float32') / 255.0

    return face
