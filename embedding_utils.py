import numpy as np

def get_embedding(face, embedder):
    face = np.expand_dims(face, axis=0)
    embedding = embedder.embeddings(face)
    return embedding[0]
