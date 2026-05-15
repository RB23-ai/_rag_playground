import json
import os
from dotenv import load_dotenv

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingest import PDFIngestor
from src.chunker import DocumentChunker
from src.embedder import Embedder
from src.retriever import Retriever
from src.generator import Generator

load_dotenv()

golden_data = [
    {"question": "What is the main topic of the RAG survey paper?", "ground_truth": "The paper surveys Retrieval-Augmented Generation (RAG) methodologies, architectures, and evaluations for Large Language Models."},
    {"question": "What is the battery life of the device?", "ground_truth": "The battery lasts up to 12 hours on a single charge."},
    {"question": "Where is C.V. Raman born?", "ground_truth": "C.V. Raman was born in India."},
    {"question": "What is hybrid search?", "ground_truth": "Hybrid search combines dense vector retrieval (like cosine similarity) with sparse keyword retrieval (like BM25) to improve search accuracy."},
    {"question": "What are the limitations of Cross-Encoders?", "ground_truth": "Cross-Encoders are computationally expensive and slow, making them unsuitable for searching across large datasets directly without a first-stage retriever."}
]

def build_evaluation_set():
    print(" Initializing Pipeline for Dataset Generation...")
    # Pointing to the data directory in the parent folder
    ingestor = PDFIngestor(data_dir="data")
    chunks = DocumentChunker().run(ingestor.load_pdfs(), strategy="recursive")
    embedder = Embedder()
    retriever = Retriever(embedder.embed_chunks(chunks))
    generator = Generator()

    dataset_dict = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }

    print(" Running Golden Set through RAG Pipeline...")
    for item in golden_data:
        q = item["question"]
        
        # Run Retrieval
        query_vec = embedder.embed_query(q)
        # Using the advanced re-ranking retriever you built!
        retrieved_docs = retriever.search_with_reranking(q, query_vec, final_k=3, initial_k=20)
        contexts = [doc['content'] for doc in retrieved_docs]
        
        # Run Generation
        answer = generator.generate(q, retrieved_docs)
        
        # Append to dataset
        dataset_dict["question"].append(q)
        dataset_dict["answer"].append(answer)
        dataset_dict["contexts"].append(contexts)
        dataset_dict["ground_truth"].append(item["ground_truth"])
        print(f" Processed: {q}")

    os.makedirs("evals", exist_ok=True)
    with open("evals/ragas_dataset.json", "w") as f:
        json.dump(dataset_dict, f, indent=4)
    print("\n Dataset saved to evals/ragas_dataset.json")

if __name__ == "__main__":
    build_evaluation_set()