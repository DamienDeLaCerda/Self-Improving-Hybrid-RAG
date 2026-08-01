import streamlit as st
import sys
import os

# 🔥 Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.retriever import build_retriever, hybrid_search
from app.query_rewriter import rewrite_query
from app.generator import generate_answer
from app.evaluator import evaluate_rag
from app.feedback import save_feedback
from app.improver import improve_queries

# ---------------- INIT ---------------- #

st.set_page_config(page_title="Self-Improving RAG", layout="wide")

st.title("🧠 Self-Improving RAG System")
st.markdown("Hybrid Retrieval + Evaluation + Feedback Loop")

# Cache retriever (important)
@st.cache_resource
def load_system():
    return build_retriever("data/arXiv_scientific dataset.csv")

collection, bm25, chunks = load_system()

# ---------------- UI INPUT ---------------- #

query = st.text_input("Enter your query")

if query:
    # Rewrite
    try:
        rewritten_query = rewrite_query(query)
    except:
        rewritten_query = query

    st.subheader("🔄 Rewritten Query")
    st.write(rewritten_query)

    # Retrieve
    results = hybrid_search(rewritten_query, collection, bm25, chunks)

    # Generate Answer
    answer = generate_answer(query, results)

    st.subheader("🧠 Final Answer")
    st.write(answer)

    # Show context
    with st.expander("📄 Retrieved Context"):
        for i, doc in enumerate(results[:3]):
            st.write(f"{i+1}. {doc}")

    # Evaluation
    try:
        eval_result = evaluate_rag(query, answer, results)
        st.subheader("📊 Evaluation Metrics")
        st.write(eval_result)
    except:
        st.warning("Evaluation not available")

    # Feedback
    st.subheader("⭐ Rate this answer")

    rating = st.slider("Rating", 1, 5, 3)

    if st.button("Submit Feedback"):
        save_feedback(query, answer, rating)
        st.success("Feedback saved!")

        improve_queries()