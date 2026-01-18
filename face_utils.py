import cv2
from mtcnn import MTCNN

detector = MTCNN()

def extract_face_from_array(image, required_size=(160, 160)):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = detector.detect_faces(image)
    if len(results) == 0:
        return None

    x, y, w, h = results[0]['box']
    x, y = abs(x), abs(y)

    face = image[y:y+h, x:x+w]
    if face.size == 0:
        return None

    face = cv2.resize(face, required_size)
    return face
