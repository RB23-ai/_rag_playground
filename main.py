from src.ingest import PDFIngestor
from src.chunker import DocumentChunker
from src.embedder import Embedder
from src.retriever import Retriever
from src.generator import Generator

# 1. Setup
print("🚀 Initializing RAG Pipeline...")
ingestor = PDFIngestor()
chunker = DocumentChunker()
embedder = Embedder()
generator = Generator()

# 2. Process Documents
raw_docs = ingestor.load_pdfs()
chunks = chunker.run(raw_docs, strategy="recursive")
embedded_chunks = embedder.embed_chunks(chunks)
retriever = Retriever(embedded_chunks)

# 3. Test Queries
queries = [
    "What is the main topic of these documents?",
    "Can you summarize the introduction of the NLP paper?",
    "What are the key findings mentioned?",
    "Is there any mention of 'Vector Databases'?",
    "Give me a 3-bullet point summary of the content."
]

print("\n--- STARTING TEST QUERIES ---")
for q in queries:
    print(f"\n❓ QUERY: {q}")
    # Retrieve
    query_vec = embedder.embed_query(q)
    relevant_chunks = retriever.search(query_vec, top_k=3)
    
    # Generate
    answer = generator.generate(q, relevant_chunks)
    print(f"💡 ANSWER: {answer}")