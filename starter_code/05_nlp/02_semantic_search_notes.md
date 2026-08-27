# Phase 8b — Semantic Search

## Setup

Embeddings generated locally using sentence-transformers (all-MiniLM-L6-v2, 384-dimensional), over a 500-ticket sample of non-blank descriptions from the same 10,000-row subset used in the classical NLP task. Cosine similarity used for retrieval.

## Direct Query Test

Query: "my package arrived broken and damaged". Top 5 retrieved results were all correctly damaged_item tickets, with similarity scores ranging 0.60-0.62, despite the query using different phrasing than the ticket templates.

## Keyword vs. Semantic Comparison

A deliberately paraphrased query with no shared vocabulary, "the thing I ordered showed up smashed to pieces", was used to directly compare TF-IDF keyword matching against semantic embedding search:

| Method | Top Results (Category) | Similarity Range | Correct? |
|---|---|---|---|
| TF-IDF (keyword) | wrong_item (all 5) | 0.325-0.334 | No |
| Semantic (embeddings) | damaged_item (all 5) | 0.569-0.601 | Yes |

The keyword baseline failed entirely, retrieving the wrong category with weak similarity scores, since the paraphrased query shared no meaningful vocabulary with the damaged_item template phrases ("crushed," "broken"). The semantic embedding approach correctly identified all 5 results as damaged_item, recognizing "smashed to pieces" as semantically equivalent to the templated damage-related language, with notably higher confidence than even the keyword method's (incorrect) top matches.

## Conclusion

This demonstrates the core value proposition of semantic search over keyword matching: robustness to paraphrasing and vocabulary variation. For a real customer support system, where customers describe the same underlying issue in many different words, semantic search would substantially outperform keyword-based retrieval for tasks like finding similar past tickets or matching customer messages to relevant knowledge-base articles, particularly where customer language diverges from internal terminology or ticket templates.