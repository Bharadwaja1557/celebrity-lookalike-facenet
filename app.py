import streamlit as st
import numpy as np
import pickle
import cv2
from face_utils import extract_face_from_array
from embedder import get_embedding
from search import find_top_k
import os

# -------------------------------
# Caching functions for efficiency
# -------------------------------

@st.cache_resource
def load_models():
    from keras_facenet import FaceNet
    from mtcnn import MTCNN
    embedder_model = FaceNet()
    face_detector = MTCNN()
    return embedder_model, face_detector

@st.cache_data
def load_embeddings():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    emb_path = os.path.join(base_dir, "embeddings", "celebrity_embeddings.pkl")
    img_path = os.path.join(base_dir, "embeddings", "image_paths.npy")

    with open(emb_path, "rb") as f:
        celeb_db = pickle.load(f)

    image_paths_db = np.load(img_path, allow_pickle=True)  # <-- fixed: remove .item()
    return celeb_db, image_paths_db

# -------------------------------
# Load cached resources
# -------------------------------

embedder, detector = load_models()
celeb_db, image_paths_db = load_embeddings()

# -------------------------------
# Streamlit UI
# -------------------------------

st.title("Celebrity Lookalike Finder")
st.write("Upload a face image and find your top 5 celebrity lookalikes!")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    image_array = np.frombuffer(uploaded.read(), np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    st.image(image, caption="Uploaded Image", channels="BGR")

    # Extract face
    face = extract_face_from_array(image, detector)

    if face is None:
        st.error("No face detected. Please upload a clear frontal face.")
    else:
        # Ensure correct shape for FaceNet
        if len(face.shape) == 3:
            face = np.expand_dims(face, axis=0)

        # Generate embedding
        query_embedding = get_embedding(face, embedder)

        # Find top 5 matches
        results = find_top_k(query_embedding, celeb_db, image_paths_db, k=5)

        st.subheader("Top Matches")
        for celeb, score, img_path in results:
            col1, col2 = st.columns([1, 2])

            match_img = cv2.imread(img_path)
            if match_img is not None:
                match_img = cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB)
                col1.image(match_img, width=120)
            else:
                col1.write("Image not found")

            col2.write(f"**{celeb}**")
            col2.write(f"Similarity: {score:.3f}")
