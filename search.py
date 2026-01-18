from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def find_top_k(query_embedding, celeb_db, image_paths_db, k=5):
    results = []

    for celeb, emb_matrix in celeb_db.items():
        sims = cosine_similarity(query_embedding, emb_matrix)
        score = sims.mean()

        best_idx = np.argmax(sims)
        best_image = image_paths_db[celeb][best_idx]

        results.append((celeb, score, best_image))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:k]
