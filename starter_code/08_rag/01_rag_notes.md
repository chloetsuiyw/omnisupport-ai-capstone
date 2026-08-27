# Phase 10 — RAG: Policy Knowledge Assistant

## Setup

40 Markdown policy documents were loaded and chunked (word-based, 300-word chunks with 50-word overlap). Given the documents were uniformly short (each corresponding to a single focused policy topic), every document remained a single chunk, no document required splitting. Embeddings were generated using sentence-transformers (all-MiniLM-L6-v2, same model used in Phase 8's semantic search), and retrieval used cosine similarity over the 40 chunk embeddings.

## Retrieval and Grounded Answering

For each question, the top-3 most similar chunks are retrieved and passed to the LLM (google/gemini-2.5-flash) within a prompt that explicitly instructs it to answer only from retrieved evidence, cite sources by filename, and state clearly when evidence is insufficient rather than guessing.

## Test Case 1: Multi-Source Synthesis

**Question:** "How many days do I have to return an unused item?"

The system retrieved three related but distinct policies (standard returns, electronics returns, marketplace returns) and correctly synthesized a single, properly caveated answer distinguishing the three return windows (30 days standard/electronics, 21 days marketplace), with each claim individually cited to its source document. This demonstrates the system does not simply parrot the single top-ranked chunk, but reasons across multiple retrieved sources when a question genuinely has a nuanced, multi-part answer.

## Test Case 2: Conflict Resolution

**Question:** "If two policies disagree about something, which one should I follow?"

The system correctly retrieved 32_rag_conflict.md (the document specifically written to guide conflict-resolution behavior) and accurately reproduced its guidance, prefer the more specific policy, flag uncertainty for human review when unresolvable, with correct citation.

## Test Case 3: Insufficient Evidence (Deliberate Out-of-Scope Question)

**Question:** "What is the CEO's personal email address?" — a deliberately unanswerable question, outside the scope of any policy document.

| Test | Top Retrieval Score | Behavior |
|---|---|---|
| Answerable questions (1 & 2) | 0.50 – 0.70 | Confident, cited answer |
| Out-of-scope question | 0.17 – 0.18 | Explicit refusal, no fabrication |

The retrieval scores for this question (0.171–0.177) were dramatically lower than for the answerable questions (0.50–0.70), correctly indicating weak semantic relevance. Critically, the LLM did not fabricate an answer despite being given retrieved chunks, it recognized the retrieved account-security documents did not actually address the question and explicitly stated it could not confirm an answer from available policy documents. This is the single most important behavior to demonstrate in a RAG system: refusing to answer beyond what evidence supports, directly satisfying the requirement in 31_rag_citations.md that the assistant "should say that it cannot confirm the answer rather than filling the gap from general memory."

## Evaluation Summary

Across three deliberately varied test cases, multi-source synthesis, conflict resolution, and out-of-scope refusal, the RAG pipeline demonstrated grounded, cited, and appropriately cautious behavior in every case. The retrieval similarity score itself proved to be a useful, interpretable signal for evidence quality: a clear numerical gap separated confidently-answerable questions from the out-of-scope one, suggesting a similarity threshold could be used as an automated guardrail (e.g. flagging any answer where top retrieval score falls below ~0.3 for mandatory human review), a concrete mechanism worth carrying into Phase 12's evaluation and guardrails work.

## Chunking Strategy Comparison

Two chunking strategies were compared: the default word-based chunking (300 words, 50-word overlap, producing 40 chunks, effectively one per document, since no document exceeded 300 words) against a much finer strategy (40 words, 10-word overlap, producing 121 chunks).

| Strategy | Chunks | Top Score | 2nd Score | 3rd Score |
|---|---|---|---|---|
| Word-chunked (300w) | 40 | 0.696 | 0.520 | 0.503 |
| Small chunks (40w) | 121 | 0.786 | 0.594 | 0.553 |

An initial comparison against a whole-document (no-splitting) strategy produced identical results to the 300-word chunking, since every document in this knowledge base is under 300 words, confirming that chunk size only matters once it becomes smaller than the source documents themselves.

The 40-word chunking strategy produced meaningfully higher retrieval similarity scores across all three top results, and shifted the relative ranking of two sources (04_marketplace_returns.md moved ahead of 02_returns_electronics.md). This is explained by embedding granularity: smaller chunks contain less surrounding, potentially irrelevant text, so their embeddings align more precisely with a focused question, the same principle behind why TF-IDF or embedding matches tend to sharpen as noise is reduced.

Trade-off: finer chunking improves retrieval precision but produces 3x more chunks to store, embed, and search, and risks fragmenting context that spans a chunk boundary (a real concern for longer source documents, though not observed here given short policy documents). For this specific knowledge base, 40 short, single-topic policy documents, the practical choice is close to a wash: whole-document or coarse chunking is simpler and computationally cheaper with only a modest precision cost. For a knowledge base with longer, multi-topic documents, the finer chunking strategy's advantage would likely be more consequential.