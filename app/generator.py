from openai import OpenAI
from config.settings import OPENAI_API_KEY

# Try to initialize OpenAI
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
    USE_OPENAI = True
except:
    USE_OPENAI = False


def generate_answer(query, context_docs):
    context = "\n".join(context_docs[:3])

    # ✅ If API available
    if USE_OPENAI:
        try:
            prompt = f"""
Answer the question based on the context.

Context:
{context}

Question:
{query}
"""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )

            return response.choices[0].message.content

        except Exception as e:
            print("⚠️ OpenAI failed, using fallback:", e)

    # 🔥 Fallback (IMPORTANT)
    return f"""
[Fallback Answer]

Based on retrieved documents:

{context}
"""