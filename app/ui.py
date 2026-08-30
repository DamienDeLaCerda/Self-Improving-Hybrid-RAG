import html
import os
import sys

import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.evaluator import evaluate_rag
from app.feedback import save_feedback
from app.generator import generate_answer
from app.improver import improve_queries
from app.query_rewriter import rewrite_query
from app.retriever import build_retriever, hybrid_search
from config.settings import OPENAI_API_KEY

st.set_page_config(
    page_title="Self-Improving RAG",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

RATE_META = {
    1: ("Poor", "#b91c1c", "#fef2f2"),
    2: ("Weak", "#c2410c", "#fff7ed"),
    3: ("Fair", "#a16207", "#fffbeb"),
    4: ("Good", "#047857", "#ecfdf5"),
    5: ("Excellent", "#0f766e", "#f0fdfa"),
}

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

    :root {
        --ink: #0b1220;
        --muted: #64748b;
        --line: #e6ebf0;
        --paper: #ffffff;
        --canvas: #eef2f6;
        --accent: #0f766e;
        --sidebar: #0b1220;
    }

    html, body, [class*="css"] {
        font-family: "IBM Plex Sans", sans-serif;
    }

    .stApp { background: var(--canvas); }
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2.5rem;
        max-width: 1180px;
    }

    [data-testid="stSidebar"] {
        background: var(--sidebar);
        border-right: 1px solid #1e293b;
    }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] small,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #94a3b8 !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-size: 1.35rem !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] * {
        color: #94a3b8 !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: #1e293b !important;
        margin: 0.9rem 0;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 0.35rem;
    }
    .brand-mark {
        width: 2rem;
        height: 2rem;
        border-radius: 8px;
        background: #0f766e;
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .brand h1 {
        margin: 0;
        font-size: 1.55rem;
        font-weight: 600;
        letter-spacing: -0.03em;
        color: var(--ink);
    }
    .subtitle {
        margin: 0 0 1.25rem 0;
        color: var(--muted);
        font-size: 0.95rem;
        line-height: 1.45;
        max-width: 42rem;
    }

    .surface {
        background: var(--paper);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 1.15rem 1.3rem 1.25rem;
        margin-bottom: 1rem;
    }
    .section-label {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 0.7rem;
    }
    .answer-text {
        font-size: 1.05rem;
        line-height: 1.7;
        color: var(--ink);
        white-space: pre-wrap;
    }
    .mono-box {
        font-family: "IBM Plex Mono", monospace;
        font-size: 0.84rem;
        line-height: 1.55;
        color: #334155;
        background: #f8fafc;
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 0.85rem 1rem;
    }
    .source-item {
        display: grid;
        grid-template-columns: 2.2rem 1fr;
        gap: 0.65rem;
        padding: 0.75rem 0;
        border-bottom: 1px solid var(--line);
    }
    .source-item:last-child { border-bottom: none; padding-bottom: 0; }
    .source-item:first-child { padding-top: 0; }
    .source-num {
        width: 1.7rem;
        height: 1.7rem;
        border-radius: 999px;
        background: #f0fdfa;
        color: var(--accent);
        border: 1px solid #99f6e4;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 0.1rem;
    }
    .source-body {
        font-size: 0.9rem;
        line-height: 1.5;
        color: #334155;
    }

    .status-row {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        margin: 0.4rem 0 0.55rem;
    }
    .dot {
        width: 0.55rem;
        height: 0.55rem;
        border-radius: 999px;
        flex-shrink: 0;
    }
    .dot-on { background: #2dd4bf; box-shadow: 0 0 0 3px #134e4a; }
    .dot-off { background: #f87171; box-shadow: 0 0 0 3px #7f1d1d; }
    .status-text {
        font-size: 0.82rem;
        font-weight: 500;
        color: #e2e8f0 !important;
    }

    .empty-state {
        background: var(--paper);
        border: 1px dashed #cbd5e1;
        border-radius: 14px;
        padding: 2.4rem 1.5rem;
        text-align: center;
        color: var(--muted);
    }
    .empty-state strong {
        display: block;
        color: var(--ink);
        font-size: 1.05rem;
        margin-bottom: 0.35rem;
    }

    .rate-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.45rem 0.8rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid transparent;
        margin-top: 0.35rem;
    }
    .eval-muted {
        font-size: 0.88rem;
        color: var(--muted);
        line-height: 1.5;
    }

    div[data-testid="stForm"] {
        background: var(--paper);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 1rem 1.15rem 0.85rem;
        margin-bottom: 1rem;
    }
    textarea {
        border-radius: 10px !important;
    }
    div[data-testid="stButton"] > button,
    div[data-testid="stFormSubmitButton"] > button {
        background: var(--ink) !important;
        color: #f8fafc !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 500 !important;
        padding: 0.45rem 1.15rem !important;
    }
    div[data-testid="stButton"] > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover {
        background: #1e293b !important;
        color: #f8fafc !important;
    }

    div[data-testid="stTabs"] button[data-baseweb="tab"] {
        font-weight: 500;
    }

    #MainMenu, footer, header { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading retrieval index…")
def load_system():
    return build_retriever("data/arXiv_scientific dataset.csv")


def run_pipeline(query: str):
    try:
        rewritten = rewrite_query(query)
    except Exception:
        rewritten = query

    results = hybrid_search(rewritten, collection, bm25, chunks)
    answer = generate_answer(query, results)

    try:
        evaluation = evaluate_rag(query, answer, results)
    except Exception as exc:
        evaluation = {"status": "unavailable", "reason": str(exc)}

    return {
        "query": query,
        "rewritten": rewritten,
        "results": results,
        "answer": answer,
        "evaluation": evaluation,
    }


def preview_text(text: str, limit: int = 360) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


collection, bm25, chunks = load_system()
llm_on = bool(OPENAI_API_KEY)

# ----- Sidebar -----
with st.sidebar:
    st.markdown("#### Workspace")
    st.caption("Retrieval · generation · feedback")
    st.divider()

    st.markdown(
        f"""
        <div class="status-row">
          <span class="dot {"dot-on" if llm_on else "dot-off"}"></span>
          <span class="status-text">{"LLM connected" if llm_on else "Fallback mode"}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "OpenAI generation enabled."
        if llm_on
        else "No API key set. Answers use retrieved context."
    )

    st.divider()
    m1, m2 = st.columns(2)
    m1.metric("Chunks", f"{len(chunks):,}")
    m2.metric("Retriever", "Hybrid")

    st.divider()
    st.caption("BM25 + vector · MiniLM · optional RAGAS eval")

# ----- Header -----
st.markdown(
    """
    <div class="brand">
      <div class="brand-mark">◈</div>
      <h1>Self-Improving RAG</h1>
    </div>
    <p class="subtitle">
      Query the scientific corpus, inspect retrieved evidence, score answers,
      and feed low ratings into the improvement loop.
    </p>
    """,
    unsafe_allow_html=True,
)

# ----- Query -----
with st.form("query_form", clear_on_submit=False):
    st.markdown('<div class="section-label">Question</div>', unsafe_allow_html=True)
    query = st.text_area(
        "Question",
        placeholder="How does retrieval-augmented generation improve factual answers?",
        height=96,
        label_visibility="collapsed",
    )
    run_col, _ = st.columns([1, 4])
    with run_col:
        submitted = st.form_submit_button("Run query", use_container_width=True)

if submitted and query.strip():
    with st.spinner("Running retrieval and generation…"):
        st.session_state["last_result"] = run_pipeline(query.strip())
        st.session_state.pop("feedback_saved", None)

result = st.session_state.get("last_result")

if not result:
    st.markdown(
        """
        <div class="empty-state">
          <strong>Ready when you are</strong>
          Enter a question above and run the query to see answer, sources, and feedback.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ----- Results -----
left, right = st.columns([1.4, 1], gap="medium")

with left:
    st.markdown(
        f"""
        <div class="surface">
          <div class="section-label">Answer</div>
          <div class="answer-text">{html.escape(result["answer"].strip())}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if result["answer"].lstrip().startswith("[Fallback Answer]"):
        st.warning(
            "OpenAI did not return an answer. The fallback text includes the error reason. "
            "Fix the key/billing issue, then stop Streamlit (Ctrl+C) and start it again "
            "in the same terminal where OPENAI_API_KEY is set."
        )

    st.markdown(
        f"""
        <div class="surface">
          <div class="section-label">Rewritten query</div>
          <div class="mono-box">{html.escape(result["rewritten"])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with right:
    docs = result["results"][:5] or []
    source_html = []
    if not docs:
        source_html.append('<div class="eval-muted">No documents retrieved.</div>')
    else:
        for i, doc in enumerate(docs, start=1):
            source_html.append(
                f"""
                <div class="source-item">
                  <div class="source-num">{i}</div>
                  <div class="source-body">{html.escape(preview_text(doc))}</div>
                </div>
                """
            )

    st.markdown(
        f"""
        <div class="surface">
          <div class="section-label">Retrieved context · {len(docs)}</div>
          {''.join(source_html)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    evaluation = result["evaluation"]
    if isinstance(evaluation, dict) and evaluation.get("status") == "unavailable":
        eval_body = (
            f'<div class="eval-muted">{html.escape(evaluation.get("reason", "Evaluation unavailable"))}</div>'
        )
        if evaluation.get("note"):
            eval_body += f'<div class="eval-muted" style="margin-top:0.35rem">{html.escape(evaluation["note"])}</div>'
    else:
        eval_body = f'<div class="mono-box">{html.escape(str(evaluation))}</div>'

    st.markdown(
        f"""
        <div class="surface">
          <div class="section-label">Evaluation</div>
          {eval_body}
        </div>
        """,
        unsafe_allow_html=True,
    )

# ----- Feedback -----
st.markdown('<div class="surface">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Feedback</div>', unsafe_allow_html=True)
st.caption("Score answer quality. Ratings below 3 are kept for the improvement loop.")

fb_left, fb_right = st.columns([2.4, 1])

with fb_left:
    rating = st.select_slider(
        "Quality",
        options=[1, 2, 3, 4, 5],
        value=st.session_state.get("pending_rating", 3),
        format_func=lambda n: f"{n} · {RATE_META[n][0]}",
        label_visibility="collapsed",
    )
    st.session_state["pending_rating"] = rating

label, color, bg = RATE_META[rating]
with fb_right:
    st.markdown(
        f"""
        <div class="rate-chip" style="background:{bg};color:{color};border-color:{color}33;">
          <span>{rating}/5</span>
          <span>· {label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Submit feedback", use_container_width=True):
        save_feedback(result["query"], result["answer"], rating)
        improve_queries()
        st.session_state["feedback_saved"] = rating

if st.session_state.get("feedback_saved"):
    saved = st.session_state["feedback_saved"]
    st.success(f"Saved — {saved}/5 ({RATE_META[saved][0]})")

st.markdown("</div>", unsafe_allow_html=True)
