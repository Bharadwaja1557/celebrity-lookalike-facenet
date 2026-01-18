# Celebrity Lookalike Finder 🎭

A deep learning–based computer vision application that finds the **celebrity you look most similar to** from an open-set database of celebrities. The system uses **face embeddings (FaceNet)** and **similarity search**, not classification, making it robust, scalable, and industry-aligned.

---

## 🚀 Project Overview

Given an input image uploaded by a user, the system:

1. Detects the face using **MTCNN**
2. Extracts a **512‑D face embedding** using **FaceNet (InceptionResNetV1)**
3. Compares the embedding against a database of celebrity embeddings
4. Ranks celebrities using **cosine similarity (identity-level aggregation)**
5. Displays the **Top‑K lookalike celebrities** with similarity scores and images

This is an **open-set face similarity / retrieval system**, not a closed-set classifier.

---

## 🧠 Key Technical Concepts

* Face Detection: MTCNN
* Face Representation: FaceNet embeddings (512‑D)
* Learning Paradigm: Metric Learning (Triplet Loss)
* Similarity Metric: Cosine Similarity
* Aggregation Strategy: Celebrity-level mean similarity
* Deployment: Streamlit

---

## 📂 Dataset

* **100 celebrities**
* **≥ 60 images per celebrity**
* **Total images:** 8,566
* **Valid embeddings generated:** 8,459

### Dataset Structure

```
dataset/
├── Celebrity_1/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── Celebrity_2/
│   └── ...
└── Celebrity_100/
```

Corrupted images and images without detectable faces are skipped safely during preprocessing.

---

## 🏗️ Project Structure

```
celebrity-lookalike/
│
├── app.py                     # Streamlit application
├── face_utils.py              # Face detection & preprocessing
├── embedder.py                # FaceNet embedding logic
├── search.py                  # Similarity search & ranking
│
├── embeddings/
│   ├── celebrity_embeddings.pkl
│   └── image_paths.npy
│
├── requirements.txt
└── README.md
```

---

## 🔬 Methodology

### 1. Face Detection

Faces are detected using **MTCNN**, ensuring the model operates only on facial regions.

### 2. Face Embedding

Each detected face is converted into a **512‑dimensional vector** using a pretrained **FaceNet** model.

### 3. Embedding Database

All celebrity face embeddings are stored and grouped by identity.

### 4. Celebrity-Level Matching

Instead of nearest-image matching, similarity is computed as:

> **Mean cosine similarity between the query embedding and all embeddings of a celebrity**

This avoids bias toward celebrities with more images.

### 5. Ranking & Output

The system returns the **Top‑K celebrities** ranked by similarity, along with representative images.

---

## 🖥️ Streamlit Application

### Features

* Image upload
* Face detection feedback
* Top‑K celebrity matches
* Similarity score display
* Representative celebrity images

### Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---


## ✅ Why This Approach?

* ❌ Not a simple CNN classifier
* ✅ Open-set recognition
* ✅ Scales to unseen identities
* ✅ Industry-standard face representation
* ✅ Explainable and debuggable

This is the same paradigm used in **FaceNet, ArcFace, DeepFace**, etc.

---

## 🧪 Environment

* Python 3.10+
* TensorFlow 2.x
* Tested on Google Colab (TPU)

---

## 📌 Future Improvements

* Face alignment using landmarks
* Approximate nearest neighbor search (FAISS)
* Confidence calibration
* Multiple-face handling
* FastAPI backend + React frontend
