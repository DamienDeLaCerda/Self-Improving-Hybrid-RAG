def evaluate_rag(query, answer, contexts):
    """
    Evaluate RAG output using RAGAS when available.
    Falls back with a clear message if ragas/deps are missing.
    """
    try:
        from ragas.metrics import faithfulness, answer_relevancy
        from ragas import evaluate
        from datasets import Dataset
    except Exception as e:
        return {
            "status": "unavailable",
            "reason": f"RAGAS import failed: {e}",
            "note": "Core RAG (retrieve + answer + feedback) still works.",
        }

    data = Dataset.from_dict({
        "question": [query],
        "answer": [answer],
        "contexts": [contexts],
    })

    result = evaluate(
        data,
        metrics=[faithfulness, answer_relevancy],
    )

    return result
