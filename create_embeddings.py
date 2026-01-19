import os
import pickle
import numpy as np
import cv2
from mtcnn import MTCNN
from keras_facenet import FaceNet

DATASET_DIR = "dataset"
OUTPUT_DIR = "embeddings"

os.makedirs(OUTPUT_DIR, exist_ok=True)

detector = MTCNN()
embedder = FaceNet()

embeddings = []
image_paths = []

def extract_face(img_path, required_size=(160, 160)):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = detector.detect_faces(img)
    if len(results) == 0:
        return None

    x, y, w, h = results[0]["box"]
    x, y = abs(x), abs(y)
    face = img[y:y+h, x:x+w]

    face = cv2.resize(face, required_size)
    return face

def get_embedding(face):
    face = face.astype("float32")
    face = np.expand_dims(face, axis=0)
    return embedder.embeddings(face)[0]

for person in os.listdir(DATASET_DIR):
    person_dir = os.path.join(DATASET_DIR, person)
    if not os.path.isdir(person_dir):
        continue

    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)

        face = extract_face(img_path)
        if face is None:
            continue

        embedding = get_embedding(face)

        embeddings.append(embedding)
        image_paths.append(img_path)

embeddings = np.array(embeddings)

with open(os.path.join(OUTPUT_DIR, "celebrity_embeddings.pkl"), "wb") as f:
    pickle.dump(embeddings, f)

np.save(os.path.join(OUTPUT_DIR, "image_paths.npy"), np.array(image_paths))

print(f"Saved {len(embeddings)} embeddings")
