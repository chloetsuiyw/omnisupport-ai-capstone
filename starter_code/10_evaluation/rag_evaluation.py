"""RAG evaluation against the 15 supplied questions in evaluation/rag_questions.json."""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "08_rag"))
import rag_pipeline

EVAL_DIR = Path(__file__).resolve().parents[2] / "evaluation"


def load_rag_questions():
    return json.loads((EVAL_DIR / "rag_questions.json").read_text())


def run_rag_evaluation():
    index = rag_pipeline.build_index()
    questions = load_rag_questions()

    results = []
    for q in questions:
        result = rag_pipeline.answer_question(q["question"], index, top_k=3)
        results.append({
            "id": q.get("id"),
            "question": q["question"],
            "answer": result["answer"],
            "sources": result["retrieved_sources"],
            "top_score": result["retrieval_scores"][0] if result["retrieval_scores"] else None,
        })
    return results


def summarize_evaluation(results):
    refused_count = sum(1 for r in results if r["answer"].strip().lower().startswith("i cannot confirm"))
    answered_count = len(results) - refused_count
    avg_score = sum(r["top_score"] for r in results) / len(results)
    return {
        "total_questions": len(results),
        "answered_confidently": answered_count,
        "explicitly_refused": refused_count,
        "avg_top_retrieval_score": round(avg_score, 3),
    }


if __name__ == "__main__":
    results = run_rag_evaluation()
    for r in results:
        print(f"\n[{r['id']}] {r['question']}")
        print(f"Answer: {r['answer']}")
        print(f"Sources: {r['sources']} | Top score: {r['top_score']}")

    print("\n--- Summary ---")
    print(summarize_evaluation(results))