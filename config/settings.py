import os
from pathlib import Path

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Optional fallback (no crash)
if OPENAI_API_KEY is None:
    print("OPENAI_API_KEY not found (running in fallback mode)")

# Prefer a local copy so startup works when Hugging Face CDN is blocked
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LOCAL_EMBEDDING_MODEL = _PROJECT_ROOT / "models" / "all-MiniLM-L6-v2"
EMBEDDING_MODEL_NAME = (
    str(_LOCAL_EMBEDDING_MODEL)
    if _LOCAL_EMBEDDING_MODEL.exists()
    else "sentence-transformers/all-MiniLM-L6-v2"
)

# Cross-encoder used after hybrid retrieval (override with local models/ copy if present)
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
