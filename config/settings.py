import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Optional fallback (no crash)
if OPENAI_API_KEY is None:
    print("⚠️ OPENAI_API_KEY not found (running in fallback mode)")