# Self-Improving Hybrid RAG

**Author:** Damien De La Cerda

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)
![Retrieval](https://img.shields.io/badge/Retrieval-BM25%20%2B%20Vector-0D9488)
![LLM](https://img.shields.io/badge/LLM-OpenAI%20%7C%20Fallback-6B7280)

A modular **Retrieval-Augmented Generation** demo that combines hybrid search, optional LLM answers, evaluation hooks, and a user feedback loop for scientific-style Q&A.

---

## Overview

**Self-Improving Hybrid RAG** answers questions over a document corpus by:

1. Expanding the user query  
2. Retrieving evidence with **BM25 + vector search**  
3. Generating an answer with **OpenAI** (or a context fallback)  
4. Optionally scoring the answer  
5. Collecting ratings so low-quality responses can drive improvement  

It is designed as a clear GenAI learning / portfolio project: each pipeline stage lives in its own module and can be extended independently.

---

## Pipeline

```text
User query
    → Query rewriter
    → Hybrid retrieval (BM25 ∪ Chroma vector search)
    → Answer generator (OpenAI gpt-4o-mini / fallback)
    → Evaluation (RAGAS when available)
    → Feedback (1–5) → feedback.json
    → Improver (processes low ratings)
```

---

## Features

| Feature | Description |
|---|---|
| Hybrid retrieval | Keyword (BM25) + semantic (MiniLM embeddings + Chroma) |
| Query rewriting | Lightweight expansion before search |
| Answer generation | OpenAI when `OPENAI_API_KEY` is set and billed; otherwise retrieved-context fallback |
| Evaluation | RAGAS faithfulness / answer relevancy when dependencies allow |
| Feedback loop | Ratings saved to `feedback.json`; scores below 3 feed the improver |
| Dual interface | Streamlit UI and CLI |

---

## Tech stack

- **UI:** Streamlit  
- **Retrieval:** `rank_bm25`, ChromaDB, Sentence Transformers (`all-MiniLM-L6-v2`)  
- **Chunking:** LangChain text splitters  
- **LLM:** OpenAI (`gpt-4o-mini`), optional  
- **Evaluation:** RAGAS (optional)  
- **Language:** Python 3.10+

---

## Project structure

```text
self-improving-rag/
├── app/
│   ├── ui.py              # Streamlit interface
│   ├── main.py            # CLI entrypoint
│   ├── retriever.py       # Load, chunk, index, hybrid search
│   ├── query_rewriter.py  # Query expansion
│   ├── generator.py       # LLM / fallback answers
│   ├── evaluator.py       # RAGAS evaluation wrapper
│   ├── feedback.py        # Persist user ratings
│   └── improver.py        # Process low-rated queries
├── config/
│   └── settings.py        # API key + embedding model path
├── models/                # Local MiniLM weights (optional, gitignored)
├── data/                  # Optional corpus CSV (gitignored)
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone and enter the project

```bash
git clone https://github.com/YOUR_USERNAME/self-improving-rag.git
cd self-improving-rag
```

### 2. Create a virtual environment (Windows)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Embedding model

The app expects Sentence Transformers **`all-MiniLM-L6-v2`**.

- If `models/all-MiniLM-L6-v2/` exists, that local copy is used (recommended when Hugging Face CDN is blocked).  
- Otherwise the model is downloaded from Hugging Face on first run.

### 5. OpenAI API key (optional, for real LLM answers)

Set the key in the **same terminal** before starting the app:

```powershell
$env:OPENAI_API_KEY = "sk-..."
```

Notes:

- Without a key (or if the API call fails), the UI still runs and returns a **fallback** answer from retrieved chunks.  
- A valid key with **available credits** is required for generated answers. Quota / billing errors appear in the fallback message.  
- A Hugging Face token is only for model downloads; it does **not** generate answers in this project.

---

## Dataset (optional)

Place an arXiv-style CSV at:

```text
data/arXiv_scientific dataset.csv
```

The loader looks for a `summary` or `abstract` column and uses up to 2000 rows.

If the file is missing, the app uses a small built-in sample corpus so you can still demo the pipeline.

---

## Run

### Streamlit UI (recommended)

```powershell
streamlit run app/ui.py
```

Open the local URL (usually `http://localhost:8501`).

### CLI

```powershell
python -m app.main
```

---

## Evaluation

When RAGAS and its dependencies load successfully:

- **Faithfulness** — is the answer grounded in retrieved context?  
- **Answer relevancy** — does the answer address the question?

If evaluation is unavailable, the UI shows a clear status and the rest of the pipeline continues.

---

## Feedback loop

1. User rates each answer from **1 (poor)** to **5 (excellent)**.  
2. Ratings are appended to `feedback.json`.  
3. Ratings below **3** are treated as bad feedback for `improver.py`.

This gives the architecture of a self-improving system. Extending the improver to update retrieval, prompts, or stored rewrites is a natural next step.

---

## Current limitations

- Chroma runs in-memory and is rebuilt on cold start.  
- Hybrid search merges BM25 and vector hits by ID union (no score fusion / re-ranking yet).  
- Query rewriting uses rule-based expansion (not an LLM rewrite).  
- The improver currently analyzes low ratings; it does not yet update the index or generation policy.

---

## Roadmap

- Reciprocal rank fusion (RRF) or cross-encoder re-ranking  
- Persistent vector store  
- Closed-loop improvement from feedback (rewrite cache / hard negatives)  
- Stronger evaluation defaults and tracing (e.g. LangSmith)  
- Multi-source document ingestion  

---

## License

MIT

---

Built by **Damien De La Cerda**.
