from sentence_transformers import SentenceTransformer, util

# Simple semantic expansion using embedding similarity

model = SentenceTransformer('all-MiniLM-L6-v2')


def rewrite_query(query: str) -> str:
    """
    Lightweight query enhancement without API
    """

    # Basic expansion rules (you can improve later)
    expansions = [
        "detailed explanation",
        "architecture",
        "working mechanism",
        "applications in machine learning",
        "deep learning context"
    ]

    expanded_query = query + " " + " ".join(expansions)

    return expanded_query