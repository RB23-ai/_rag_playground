import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class Retriever:
    def __init__(self, embedded_chunks):
        self.chunks = embedded_chunks
        # Convert list of vectors into a matrix for fast math
        self.vector_matrix = np.array([c["vector"] for c in embedded_chunks])

    def search(self, query_vector, top_k=3):
        """Returns the top K most similar chunks."""
        # Reshape query for sklearn
        qv = np.array(query_vector).reshape(1, -1)
        
        # Calculate similarity (returns a list of scores between 0 and 1)
        scores = cosine_similarity(qv, self.vector_matrix)[0]
        
        # Get indices of the highest scores
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "content": self.chunks[idx]["content"],
                "score": float(scores[idx]),
                "source": self.chunks[idx]["metadata"]["source"]
            })
        return results