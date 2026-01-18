import streamlit as st
import numpy as np
import pickle
import cv2

from face_utils import extract_face_from_array
from embedder import get_embedding
from search import find_top_k

# Load data
with open("embeddings/celebrity_embeddings.pkl", "rb") as f:
    celeb_db = pickle.load(f)

image_paths_db = np.load("embeddings/image_paths.npy", allow_pickle=True).item()

st.title("Celebrity Lookalike Finder")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = np.frombuffer(uploaded.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    st.image(image, caption="Uploaded Image", channels="BGR")

    face = extract_face_from_array(image)

    if face is None:
        st.error("No face detected.")
    else:
        query_embedding = get_embedding(face)

        results = find_top_k(query_embedding, celeb_db, image_paths_db)

        st.subheader("Top Matches")
        for celeb, score, img_path in results:
            col1, col2 = st.columns([1, 2])

            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            col1.image(img, width=120)
            col2.write(f"**{celeb}**")
            col2.write(f"Similarity: {score:.3f}")
