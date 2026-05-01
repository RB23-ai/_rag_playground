from src.ingest import PDFIngestor
from src.chunker import DocumentChunker
from src.embedder import Embedder
from src.retriever import Retriever

# 1. Ingest
raw_docs = PDFIngestor().load_pdfs()

# 2. Chunk
chunks = DocumentChunker().run(raw_docs, strategy="recursive")

# 3. Embed
embedder = Embedder()
embedded_chunks = embedder.embed_chunks(chunks)

# 4. Retrieve
retriever = Retriever(embedded_chunks)
query = "What is the main finding of the RAG paper?"
query_vec = embedder.embed_query(query)
results = retriever.search(query_vec)

print(f"\n🔍 Query: {query}")
for r in results:
    print(f"\n[{r['score']:.2f}] Source: {r['source']}\nContent: {r['content'][:200]}...")