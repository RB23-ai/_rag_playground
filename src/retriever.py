import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

class Retriever:
    def __init__(self, embedded_chunks):
        self.chunks = embedded_chunks
        
        # 1. Setup Dense Retrieval (Vector)
        self.vector_matrix = np.array([c["vector"] for c in embedded_chunks])
        
        # 2. Setup Sparse Retrieval (BM25)
        print(" Building BM25 Index...")
        self.tokenized_corpus = [c["content"].lower().split() for c in embedded_chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        # 3. Setup Cross-Encoder for Re-ranking
        print(" Loading Cross-Encoder (ms-marco-MiniLM-L-6-v2)...")
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def _normalize(self, scores):
        """Helper to scale scores between 0 and 1 so we can combine them safely."""
        low, high = np.min(scores), np.max(scores)
        if high - low == 0:
            return scores
        return (scores - low) / (high - low)

    def search_hybrid(self, query_text, query_vector, top_k=20, alpha=0.5):
        """Combines Vector Math + Keyword Matching"""
        # --- Dense Score (Cosine) ---
        qv = np.array(query_vector).reshape(1, -1)
        dense_scores = cosine_similarity(qv, self.vector_matrix)[0]
        dense_scores = self._normalize(dense_scores)
        
        # --- Sparse Score (BM25) ---
        tokenized_query = query_text.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_scores = self._normalize(bm25_scores)
        
        # --- Hybrid Scoring (Weighted Sum) ---
        hybrid_scores = (alpha * dense_scores) + ((1 - alpha) * bm25_scores)
        
        # Get top indices
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "content": self.chunks[idx]["content"],
                "score": float(hybrid_scores[idx]),
                "source": self.chunks[idx]["metadata"]["source"]
            })
        return results

    def search_with_reranking(self, query_text, query_vector, final_k=3, initial_k=20):
        """Retrieves top 20 fast, then uses AI to rerank down to top 3."""
        start_time = time.time()
        
        # 1. Fast Hybrid Retrieval
        initial_results = self.search_hybrid(query_text, query_vector, top_k=initial_k)
        retrieval_time = time.time() - start_time
        
        # 2. Format pairs for Cross-Encoder
        pairs = [[query_text, res['content']] for res in initial_results]
        
        # 3. Re-rank
        rerank_start = time.time()
        scores = self.reranker.predict(pairs)
        rerank_time = time.time() - rerank_start
        
        # 4. Sort by the new AI scores
        for i, score in enumerate(scores):
            initial_results[i]['rerank_score'] = float(score)
            
        reranked_results = sorted(initial_results, key=lambda x: x['rerank_score'], reverse=True)
        total_latency = time.time() - start_time
        
        print(f"   [Latency] Hybrid Fetch: {retrieval_time:.3f}s | Rerank: {rerank_time:.3f}s | Total: {total_latency:.3f}s")
        
        return reranked_results[:final_k]