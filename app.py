import streamlit as st
import numpy as np
import cv2
import pickle

from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet

from face_utils import extract_face_from_array
from embedding_utils import get_embedding
from similarity_utils import cosine_similarity

st.set_page_config(page_title="Celebrity Lookalike", layout="centered")

st.title("Celebrity Lookalike Finder")

# Load models
detector = MTCNN()
embedder = FaceNet()

# Load embeddings
with open("embeddings/celebrity_embeddings.pkl", "rb") as f:
    celebrity_embeddings = pickle.load(f)

image_paths = np.load("embeddings/image_paths.npy")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    face = extract_face_from_array(image, detector)

    if face is None:
        st.error("No face detected. Please upload a clearer image.")
        st.stop()

    query_embedding = get_embedding(face, embedder)

    similarities = []

    for name, emb in celebrity_embeddings:
        score = cosine_similarity(query_embedding, emb)
        similarities.append((name, score))


    best_match = max(similarities, key=lambda x: x[1])
    best_name, best_score = best_match

    st.subheader("Best Match")
    st.write(f"Celebrity: **{best_name}**")
    st.write(f"Similarity Score: **{best_score:.4f}**")

