# 🎭 Celebrity Look-Alike Finder (FaceNet + MTCNN)

A deep learning–based web application that finds the **closest celebrity look-alike** for a given face image.  
Built using **FaceNet embeddings**, **MTCNN face detection**, and deployed with **Streamlit**.

---

## 🚀 Live Demo
Deployable on **Streamlit Community Cloud**

> Upload a face image → Get the most similar celebrity from the dataset.

---

## 📌 Features

- Face detection using **MTCNN**
- Face embedding extraction using **FaceNet**
- Similarity matching using **cosine similarity**
- Clean **two-column UI**:
  - Left: Input image
  - Right: Matched celebrity image
- Adjustable similarity threshold
- Fully **cloud-deployable** (no system dependencies)
- Uses **relative dataset paths** (portable across environments)

---

## 🧠 Model Pipeline

1. **Input Image**
2. **Face Detection** (MTCNN)
3. **Face Cropping & Alignment**
4. **Face Embedding** (FaceNet – 128-D)
5. **Cosine Similarity Matching**
6. **Best Celebrity Match Returned**

---

## 🗂️ Project Structure

```
celebrity-lookalike-facenet/
│
├── app.py
├── requirements.txt
├── embeddings.npy
├── labels.npy
├── image_paths.npy
├── dataset/
│   ├── Akshaye_Khanna/
│   │   ├── Akshaye_Khanna.1.jpg
│   │   └── ...
│   └── ...
└── README.md
```

---

## 📦 Requirements

```
streamlit
numpy
scikit-learn
Pillow
tensorflow==2.15.0
keras-facenet
mtcnn
opencv-python-headless
```

---

## ▶️ Run Locally

```bash
git clone https://github.com/Bharadwaja1557/celebrity-lookalike-facenet.git
cd celebrity-lookalike-facenet
pip install -r requirements.txt
streamlit run app.py
```

---

## 👤 Author

Built by **[Bharadwaja](https://github.com/Bharadwaja1557)**



