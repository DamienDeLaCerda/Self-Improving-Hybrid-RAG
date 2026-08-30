from openai import OpenAI

from config.settings import OPENAI_API_KEY

USE_OPENAI = bool(OPENAI_API_KEY)
client = OpenAI(api_key=OPENAI_API_KEY) if USE_OPENAI else None
LAST_ERROR = None


def generate_answer(query, context_docs):
    """
    Generate an answer from retrieved context.
    Returns a fallback string if OpenAI is unavailable or the API call fails.
    """
    global LAST_ERROR
    LAST_ERROR = None

    context = "\n".join(context_docs[:3])

    if not USE_OPENAI or client is None:
        LAST_ERROR = "OPENAI_API_KEY is not set in this process."
        return _fallback(context, LAST_ERROR)

    prompt = f"""Answer the question using only the context below.
If the context is insufficient, say what is missing.
Keep the answer concise and clear.

Context:
{context}

Question:
{query}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        LAST_ERROR = f"{type(e).__name__}: {e}"
        print("OpenAI failed, using fallback:", LAST_ERROR)
        return _fallback(context, LAST_ERROR)


def _fallback(context, reason):
    return (
        "[Fallback Answer]\n"
        f"OpenAI call did not succeed ({reason})\n\n"
        "Based on retrieved documents:\n\n"
        f"{context}"
    )
