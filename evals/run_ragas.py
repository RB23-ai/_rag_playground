import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def run_evaluation():
    print(" Loading dataset...")
    if not os.path.exists("evals/ragas_dataset.json"):
        print("Dataset not found. Run build_dataset.py first.")
        return

    with open("evals/ragas_dataset.json", "r") as f:
        data = json.load(f)
    
    hf_dataset = Dataset.from_dict(data)
    
    judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    judge_embeddings = OpenAIEmbeddings()

    print(" Running RAGAS Evaluation (this takes a few minutes)...")
    results = evaluate(
        dataset=hf_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings
    )
    
    print("\n --- RAGAS SCORES --- ")
    print(results)
    
    df = results.to_pandas()
    
    markdown_content = f"""# RAG Pipeline Evaluation Results

## Aggregate Metrics
- **Faithfulness (Hallucination Check):** {results['faithfulness']:.4f}
- **Answer Relevancy:** {results['answer_relevancy']:.4f}
- **Context Precision (Ranking Quality):** {results['context_precision']:.4f}
- **Context Recall (Information Retrieval):** {results['context_recall']:.4f}

## Row-by-Row Analysis
{df.to_markdown(index=False)}
"""
    with open("evals/results.md", "w", encoding="utf-8") as f:
        f.write(markdown_content)
        
    print(" Detailed results saved to evals/results.md")

if __name__ == "__main__":
    run_evaluation()