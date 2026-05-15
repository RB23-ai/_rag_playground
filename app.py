import streamlit as st
from src.ingest import PDFIngestor
from src.chunker import DocumentChunker
from src.embedder import Embedder
from src.retriever import Retriever
from src.generator import Generator

st.set_page_config(page_title="RAG Playground", layout="wide")
st.title(" RAG Playground v1.0")

@st.cache_resource
def load_pipeline():
    ingestor = PDFIngestor(data_dir="data")
    chunker = DocumentChunker()
    embedder = Embedder()
    raw_docs = ingestor.load_pdfs()
    chunks = chunker.run(raw_docs, strategy="recursive")
    embedded_chunks = embedder.embed_chunks(chunks)
    return Retriever(embedded_chunks), embedder, Generator()

try:
    retriever, embedder, generator = load_pipeline()
    st.success("Knowledge Base Loaded from /data folder!")
except Exception as e:
    st.error(f" Failed to load pipeline: {e}")

query = st.text_input("Ask a question about your documents:")
if query:
    with st.spinner("Searching and Generating..."):
        query_vec = embedder.embed_query(query)
        # Using the advanced Cross-Encoder re-ranking
        results = retriever.search_with_reranking(query, query_vec, final_k=3, initial_k=20)
        answer = generator.generate(query, results)
        
        st.subheader(" Answer")
        st.write(answer)
        
        with st.expander("View Retrieved Sources"):
            for res in results:
                st.write(f"**Source:** {res['source']} (Score: {res['rerank_score']:.2f})")
                st.info(res['content'])