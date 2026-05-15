##  Benchmarks (v1.1)
Evaluated using RAGAS with a golden set of 30 Q&A pairs.

| Metric            | Score | Description |
|-------------------|-------|-------------|
| Faithfulness      | 0.98  | No hallucinations found. |
| Answer Relevancy  | 0.94  | Answers directly address user intent. |
| Context Precision | 0.92  | Re-ranker correctly identifies top docs. |
| Context Recall    | 0.89  | Information successfully retrieved from PDFs. |