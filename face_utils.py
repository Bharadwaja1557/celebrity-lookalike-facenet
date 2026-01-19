import cv2
import numpy as np

def extract_face_from_array(image, detector, required_size=(160, 160)):
    results = detector.detect_faces(image)

    if len(results) == 0:
        return None

    x, y, width, height = results[0]['box']
    x, y = max(0, x), max(0, y)

    face = image[y:y + height, x:x + width]

    if face.size == 0:
        return None

    face = cv2.resize(face, required_size)
    face = face.astype("float32")

    return face
