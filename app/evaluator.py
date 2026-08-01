from ragas.metrics import faithfulness, answer_relevancy
from ragas import evaluate
from datasets import Dataset


def evaluate_rag(query, answer, contexts):
    """
    Evaluate RAG output using RAGAS
    """

    data = Dataset.from_dict({
        "question": [query],
        "answer": [answer],
        "contexts": [contexts]
    })

    result = evaluate(
        data,
        metrics=[faithfulness, answer_relevancy]
    )

    return result