import streamlit as st
import numpy as np
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import tempfile
import os

# -----------------------------
# App configuration
# -----------------------------
st.set_page_config(
    page_title="Celebrity Look-Alike Finder",
    layout="centered"
)

st.title("Celebrity Look-Alike Finder")

# -----------------------------
# Load models (cached)
# -----------------------------
@st.cache_resource
def load_models():
    detector = MTCNN()
    embedder = FaceNet()
    return detector, embedder

detector, embedder = load_models()

# -----------------------------
# Load embeddings (cached)
# -----------------------------
@st.cache_data
def load_data():
    embeddings = np.load("embeddings.npy")
    labels = np.load("labels.npy")
    image_paths = np.load("image_paths.npy")
    return embeddings, labels, image_paths

embeddings, labels, image_paths = load_data()

# -----------------------------
# Face extraction (NO OpenCV)
# -----------------------------
def extract_face(image_path, target_size=(160, 160)):
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        return None

    img_np = np.asarray(img)
    results = detector.detect_faces(img_np)

    if not results:
        return None

    # pick largest detected face
    results = sorted(
        results,
        key=lambda r: r["box"][2] * r["box"][3],
        reverse=True
    )

    x, y, w, h = results[0]["box"]
    x, y = max(0, x), max(0, y)

    face = img_np[y:y + h, x:x + w]
    if face.size == 0:
        return None

    face_img = Image.fromarray(face)
    face_img = face_img.resize(target_size)
    face_array = np.asarray(face_img).astype("float32")

    return face_array

# -----------------------------
# Matching logic
# -----------------------------
def find_celebrity(face):
    face = np.expand_dims(face, axis=0)

    embedding = embedder.embeddings(face)
    embedding = embedding / np.linalg.norm(embedding)

    similarities = cosine_similarity(embedding, embeddings)[0]
    best_idx = np.argmax(similarities)

    return best_idx, similarities[best_idx]

# -----------------------------
# UI
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a face image",
    type=["jpg", "jpeg", "png"]
)

SIM_THRESHOLD = st.slider(
    "Similarity threshold",
    min_value=0.50,
    max_value=0.90,
    value=0.65,
    step=0.01
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        temp_path = tmp.name

    face = extract_face(temp_path)

    if face is None:
        st.error("No face detected in the uploaded image.")
    else:
        idx, score = find_celebrity(face)

        if score < SIM_THRESHOLD:
            st.warning("No close celebrity match found.")
        else:
            celeb_name = labels[idx]
            match_img = Image.open(image_paths[idx])

            st.success(f"Matched Celebrity: {celeb_name}")
            st.write(f"Similarity score: **{score:.4f}**")
            st.image(match_img, caption=celeb_name, use_container_width=True)

    os.remove(temp_path)
