"""Semantic-search starter.
TODO: create embeddings using a course-approved provider, compare cosine similarity results, and document retrieval quality.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

SUBSET_PATH = Path(__file__).resolve().parents[2] / "data" / "subsets" / "transformer_finetune_10000.parquet"

def load_corpus(n=500):
    df = pd.read_parquet(SUBSET_PATH)
    df = df[df["issue_description"].astype(str).str.strip() != ""]
    return df.sample(n, random_state=42).reset_index(drop=True)

def build_embeddings(corpus_texts):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(corpus_texts, show_progress_bar=True)
    return embeddings, model

def semantic_search(query, corpus_df, embeddings, model, top_k=5):
    query_embedding = model.encode([query])
    sims = cosine_similarity(query_embedding, embeddings)[0]
    top_idx = sims.argsort()[::-1][:top_k]
    results = corpus_df.iloc[top_idx].copy()
    results["similarity"] = sims[top_idx]
    return results

def compare_keyword_vs_semantic(query, corpus_df, embeddings, model, top_k=5):
    # Keyword baseline: TF-IDF + cosine similarity
    vectorizer = TfidfVectorizer()
    corpus_vec = vectorizer.fit_transform(corpus_df["issue_description"])
    query_vec = vectorizer.transform([query])
    keyword_sims = cosine_similarity(query_vec, corpus_vec)[0]
    keyword_top = corpus_df.iloc[keyword_sims.argsort()[::-1][:top_k]].copy()
    keyword_top["similarity"] = keyword_sims[keyword_sims.argsort()[::-1][:top_k]]

    # Semantic search
    semantic_top = semantic_search(query, corpus_df, embeddings, model, top_k=top_k)

    return keyword_top, semantic_top

if __name__ == "__main__":
    corpus_df = load_corpus(n=500)
    print("Corpus size:", len(corpus_df))

    print("\n--- Building embeddings (may take a moment on first run, downloads model) ---")
    embeddings, model = build_embeddings(corpus_df["issue_description"].tolist())
    print("Embedding shape:", embeddings.shape)

    query = "my package arrived broken and damaged"
    print(f"\n--- Semantic search for: '{query}' ---")
    results = semantic_search(query, corpus_df, embeddings, model, top_k=5)
    print(results[["issue_category", "issue_description", "similarity"]])

    paraphrased_query = "the thing I ordered showed up smashed to pieces"
    print(f"\n--- Comparing keyword vs semantic for paraphrased query: '{paraphrased_query}' ---")
    keyword_results, semantic_results = compare_keyword_vs_semantic(paraphrased_query, corpus_df, embeddings, model)

    print("\nKeyword (TF-IDF) results:")
    print(keyword_results[["issue_category", "similarity"]])

    print("\nSemantic results:")
    print(semantic_results[["issue_category", "similarity"]])