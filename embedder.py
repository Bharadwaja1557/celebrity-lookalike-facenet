import numpy as np

def get_embedding(face, embedder=None):
    """
    face: numpy array (H,W,C) scaled 0-1
    embedder: keras-facenet FaceNet instance
    returns: 512-d embedding
    """
    if embedder is None:
        from keras_facenet import FaceNet
        embedder = FaceNet()

    # Ensure batch dimension
    if len(face.shape) == 3:
        face = np.expand_dims(face, axis=0)  # (1,H,W,C)

    embedding = embedder.embeddings(face)
    return embedding[0]
