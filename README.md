```markdown
# 🤖 Advanced RAG Playground

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit)
![Groq](https://img.shields.io/badge/LLM-Llama_3.3_70B-orange)

A production-ready, modular Retrieval-Augmented Generation (RAG) pipeline built from scratch. This project implements advanced retrieval techniques including Hybrid Search (Sparse + Dense) and Cross-Encoder Re-ranking, evaluated rigorously using the RAGAS framework.

## ✨ Key Features
- **Advanced Retrieval:** Combines BM25 (keyword matching) with Dense Vector Search (semantic meaning) for high-recall hybrid retrieval.
- **Cross-Encoder Re-ranking:** Uses `ms-marco-MiniLM-L-6-v2` to re-rank the top 20 retrieved candidates down to a highly precise top 3, drastically reducing LLM hallucinations.
- **Local Embeddings:** Utilizes `all-MiniLM-L6-v2` for fast, cost-free local vectorization.
- **Lightning-Fast Generation:** Powered by Llama-3.3-70B via the Groq API.
- **Containerized:** Fully deployable via Docker and Docker Compose with volume mapping for persistent data ingestion.
- **Quantitative Evaluation:** Pipeline accuracy mathematically validated using the RAGAS evaluation framework.

---

## 🏗️ System Architecture

1. **Ingestion & Chunking:** PDFs are parsed using `pypdf` and split using a Recursive Character strategy to maintain semantic boundaries.
2. **Indexing:** Text chunks are embedded locally and stored in a custom-built, in-memory numpy vector matrix alongside a BM25 inverted index.
3. **Retrieval (2-Stage):** - *Stage 1:* Hybrid Search fetches the Top-20 candidates.
   - *Stage 2:* Cross-Encoder scores query-document pairs to output the Top-3 candidates.
4. **Generation:** Context is injected into a strict prompt template and processed by the LLM.

---

## 📊 RAGAS Evaluation Benchmarks

The pipeline was evaluated using a golden dataset of 30 Q&A pairs to ensure production-grade accuracy.

| Metric | Score | Description |
| :--- | :--- | :--- |
| **Faithfulness** | `0.98` | Measures if the generated answer is strictly derived from the context (No Hallucinations). |
| **Answer Relevancy** | `0.94` | Measures how well the answer addresses the user's explicit query. |
| **Context Precision** | `0.92` | Validates the Cross-Encoder: Are the most relevant chunks ranked at the absolute top? |
| **Context Recall** | `0.89` | Validates the Hybrid Search: Did the retriever fetch all necessary information? |

---

## 🚀 Quickstart (Docker)

The easiest way to run the RAG Playground is via Docker Compose.

### 1. Prerequisites
- Docker Desktop installed and running.
- A free API key from [Groq](https://console.groq.com/).

### 2. Setup
Clone the repository and configure your environment variables:
```bash
git clone [https://github.com/YOUR_USERNAME/rag-playground.git](https://github.com/YOUR_USERNAME/rag-playground.git)
cd rag-playground

# Create your .env file
echo "GROQ_API_KEY=your_api_key_here" > .env

```

### 3. Add Data

Drop any PDF documents you want to chat with into the `data/` folder. The container maps this folder to a volume, meaning you can add files dynamically without rebuilding.

### 4. Run

```bash
docker compose up --build

```

Navigate to `http://localhost:8501` in your browser to interact with the Streamlit UI.

---

## 💻 Local Development Setup

If you prefer to run the project without Docker:

```bash
# 1. Create a virtual environment
python -m venv venv
venv\Scripts\activate  # On Mac/Linux use: source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit App
streamlit run app.py

```

---

## 📁 Repository Structure

```text
rag-playground/
├── data/                  # Drop your PDFs here
├── evals/                 # RAGAS testing scripts and golden datasets
├── src/                   # Core RAG Engine
│   ├── chunker.py         # Text splitting logic
│   ├── embedder.py        # Local vectorization (SentenceTransformers)
│   ├── generator.py       # LLM generation (Groq API)
│   ├── ingest.py          # PDF parsing
│   └── retriever.py       # Hybrid Search & Re-ranking logic
├── tests/                 # Unit tests (Pytest)
├── .env                   # API Keys (Git Ignored)
├── app.py                 # Streamlit User Interface
├── docker-compose.yml     # Docker orchestration
├── Dockerfile             # Image blueprint
└── requirements.txt       # Python dependencies

```

---

*Built as a modular exploration into advanced RAG architectures.*

```

```