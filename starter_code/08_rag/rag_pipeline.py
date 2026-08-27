"""RAG starter.
TODO: load Markdown policies, chunk documents, create embeddings, store vectors in FAISS/Chroma, retrieve evidence, build a grounded prompt, show citations, and evaluate known questions.
"""

from pathlib import Path
import os
import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

KB_DIR = Path(__file__).resolve().parents[2] / "knowledge_base"


def load_policy_documents():
    docs = []
    for path in sorted(KB_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        docs.append({"filename": path.name, "text": text})
    return docs


def chunk_document(doc, chunk_size=300, overlap=50):
    """Simple word-based chunking with overlap."""
    words = doc["text"].split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_text = " ".join(words[start:end])
        chunks.append({"filename": doc["filename"], "chunk_text": chunk_text, "chunk_start": start})
        start += chunk_size - overlap
    return chunks


def build_index(policy_dir=KB_DIR):
    docs = load_policy_documents()
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))

    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [c["chunk_text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    return {
        "chunks": all_chunks,
        "embeddings": embeddings,
        "model": model,
    }

import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")
llm_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("LLM_API_KEY"),
)
MODEL_NAME = os.getenv("MODEL_NAME")

RAG_PROMPT = """You are a customer support policy assistant. Answer the question using ONLY the retrieved policy excerpts below. Cite the source document(s) you used by filename.

If the retrieved excerpts do not contain enough information to answer confidently, say clearly that you cannot confirm the answer from available policy documents, rather than guessing.

Retrieved policy excerpts:
{context}

Question: {question}

Answer (include citations by filename):"""


def retrieve(question, index, top_k=3):
    query_embedding = index["model"].encode([question])
    sims = cosine_similarity(query_embedding, index["embeddings"])[0]
    top_idx = sims.argsort()[::-1][:top_k]
    results = []
    for i in top_idx:
        chunk = index["chunks"][i]
        results.append({**chunk, "similarity": float(sims[i])})
    return results

def build_index_whole_documents(policy_dir=KB_DIR):
    """Alternative chunking strategy: no splitting, one chunk per whole document."""
    docs = load_policy_documents()
    all_chunks = [{"filename": d["filename"], "chunk_text": d["text"], "chunk_start": 0} for d in docs]

    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [c["chunk_text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=False)

    return {"chunks": all_chunks, "embeddings": embeddings, "model": model}


def compare_chunking_strategies(question):
    index_wordchunk = build_index()
    index_small = build_index_small_chunks()

    result_wordchunk = answer_question(question, index_wordchunk, top_k=3)
    result_small = answer_question(question, index_small, top_k=3)

    return {
        "word_chunked (300w, 50w overlap)": {
            "num_chunks": len(index_wordchunk["chunks"]),
            "sources": result_wordchunk["retrieved_sources"],
            "scores": result_wordchunk["retrieval_scores"],
        },
        "small_chunks (40w, 10w overlap)": {
            "num_chunks": len(index_small["chunks"]),
            "sources": result_small["retrieved_sources"],
            "scores": result_small["retrieval_scores"],
        },
    }

def answer_question(question, index, top_k=3):
    retrieved = retrieve(question, index, top_k=top_k)
    context = "\n\n".join(
        f"[Source: {r['filename']}]\n{r['chunk_text']}" for r in retrieved
    )
    prompt = RAG_PROMPT.format(context=context, question=question)

    response = llm_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=400,
    )
    answer = response.choices[0].message.content.strip()

    return {
        "question": question,
        "answer": answer,
        "retrieved_sources": [r["filename"] for r in retrieved],
        "retrieval_scores": [round(r["similarity"], 3) for r in retrieved],
    }
def build_index_small_chunks(policy_dir=KB_DIR, chunk_size=40, overlap=10):
    """Alternative chunking strategy: small chunks, to test whether finer granularity changes retrieval."""
    docs = load_policy_documents()
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, chunk_size=chunk_size, overlap=overlap))

    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [c["chunk_text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=False)

    return {"chunks": all_chunks, "embeddings": embeddings, "model": model}

if __name__ == "__main__":
    index = build_index()
    print(f"Total documents: {len(list(KB_DIR.glob('*.md')))}")
    print(f"Total chunks: {len(index['chunks'])}")
    print(f"Embedding shape: {index['embeddings'].shape}")

    test_question = "How many days do I have to return an unused item?"
    print(f"\n--- Question: {test_question} ---")
    result = answer_question(test_question, index)
    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['retrieved_sources']}")
    print(f"Retrieval scores: {result['retrieval_scores']}")

    print("\n--- Testing conflict/ambiguity handling ---")
    conflict_question = "If two policies disagree about something, which one should I follow?"
    result2 = answer_question(conflict_question, index)
    print(f"Answer: {result2['answer']}")
    print(f"Sources: {result2['retrieved_sources']}")

    print("\n--- Testing 'insufficient evidence' handling ---")
    no_answer_question = "What is the CEO's personal email address?"
    result3 = answer_question(no_answer_question, index)
    print(f"Answer: {result3['answer']}")
    print(f"Sources: {result3['retrieved_sources']}")
    print(f"Retrieval scores: {result3['retrieval_scores']}")

    print("\n--- Chunking Strategy Comparison ---")
    comparison = compare_chunking_strategies(test_question)
    for strategy, result in comparison.items():
        print(f"\n{strategy}:")
        print(f"  Sources: {result['sources']}")
        print(f"  Scores: {result['scores']}")

    for strategy, result in comparison.items():
        print(f"\n{strategy}:")
        print(f"  Num chunks: {result['num_chunks']}")
        print(f"  Sources: {result['sources']}")
        print(f"  Scores: {result['scores']}")