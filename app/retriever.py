import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import os

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


def embed(texts):
    return embedding_model.encode(texts, batch_size=32, show_progress_bar=True).tolist()


# ---------------- LOAD DATA ---------------- #

def load_data(file_path, sample_size=2000):
    if os.path.exists(file_path):
        print("📂 Loading local dataset...")
        df = pd.read_csv(file_path)

        print("Columns:", df.columns)

        if "summary" in df.columns:
            texts = df["summary"].dropna().tolist()
        elif "abstract" in df.columns:
            texts = df["abstract"].dropna().tolist()
        else:
            raise ValueError("No valid text column found")

        texts = texts[:sample_size]

    else:
        print("⚠️ Dataset not found. Using fallback sample data...")

        texts = [
            "Transformer models are used in deep learning for NLP tasks.",
            "RAG combines retrieval and generation for better AI responses.",
            "Machine learning enables systems to learn from data.",
            "Neural networks are inspired by the human brain.",
            "Large language models are trained on vast text data."
        ]

    print(f"Using {len(texts)} documents")

    return texts


# ---------------- CHUNK ---------------- #

def chunk_data(texts):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = []
    for text in texts:
        chunks.extend(splitter.split_text(text))

    return chunks


# ---------------- VECTOR DB ---------------- #

def store_in_chroma(chunks):
    client = chromadb.Client()
    collection = client.create_collection(name="rag_db")

    batch_size = 64

    for i in tqdm(range(0, len(chunks), batch_size)):
        batch_chunks = chunks[i:i + batch_size]
        embeddings = embed(batch_chunks)

        ids = [str(j) for j in range(i, i + len(batch_chunks))]

        collection.add(
            documents=batch_chunks,
            ids=ids,
            embeddings=embeddings
        )

    return collection


# ---------------- BM25 ---------------- #

def build_bm25(chunks):
    tokenized_corpus = [chunk.split() for chunk in chunks]
    return BM25Okapi(tokenized_corpus)


# ---------------- HYBRID SEARCH ---------------- #

def hybrid_search(query, collection, bm25, chunks, k=5):
    bm25_scores = bm25.get_scores(query.split())
    bm25_top = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k]

    query_embedding = embed([query])[0]
    vector_results = collection.query(query_embeddings=[query_embedding], n_results=k)

    vector_ids = [int(i) for i in vector_results["ids"][0]]

    combined_ids = list(set(bm25_top + vector_ids))

    return [chunks[i] for i in combined_ids]


# ---------------- MAIN ---------------- #

def build_retriever(file_path):
    print("Loading data...")
    texts = load_data(file_path)

    print("Chunking...")
    chunks = chunk_data(texts)

    print(f"Total chunks: {len(chunks)}")

    print("Building Vector DB...")
    collection = store_in_chroma(chunks)

    print("Building BM25...")
    bm25 = build_bm25(chunks)

    print("✅ Hybrid Retriever Ready!")

    return collection, bm25, chunks