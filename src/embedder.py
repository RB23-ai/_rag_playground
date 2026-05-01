from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print(f" Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)

    def embed_chunks(self, chunks):
        """Adds a 'vector' key to each chunk dictionary."""
        texts = [c["content"] for c in chunks]
        vectors = self.model.encode(texts, show_progress_bar=True)
        
        for chunk, vector in zip(chunks, vectors):
            chunk["vector"] = vector.tolist()
        return chunks

    def embed_query(self, query):
        """Turns a user question into a vector."""
        return self.model.encode(query).tolist()