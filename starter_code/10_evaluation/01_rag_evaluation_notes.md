# Phase 12a — RAG Evaluation (15 Supplied Questions)

## Method

All 15 questions from evaluation/rag_questions.json were run against the Phase 10 RAG pipeline (top-3 retrieval, google/gemini-2.5-flash for answer generation). Each answer was manually reviewed against its retrieved sources for correctness and appropriate citation.

## Results Summary

| Metric | Value |
|---|---|
| Total questions | 15 |
| Answered confidently and correctly | 13 |
| Explicitly refused (insufficient evidence) | 2 |
| Average top retrieval score | 0.547 |

## Correct, Well-Grounded Answers (13/15)

The system correctly handled a wide range of question types, including: precise numeric policy facts (RAG02: £100 frontline refund limit), multi-condition reasoning (RAG06: warranty vs. return-window distinction for a 5-month-old fault), safety escalation (RAG08, RAG13: smoke/appliance and suspected account takeover both correctly identified as requiring immediate escalation), and system-behavior policy (RAG09: refusing to claim a cancellation succeeded without tool confirmation; RAG10: correct handling of invalid order IDs).

## Appropriate Refusals (2/15)
- RAG04 ("Is a marketplace order 25 days after delivery definitely returnable under the standard 30-day rule?"): correctly identified that the marketplace-specific 21-day policy applies instead, and that eligibility cannot be confirmed without checking the seller-specific policy — a nuanced "it depends" answer rather than a simple yes/no.
- RAG12 ("When can CSAT create leakage in a ticket escalation model?"): correctly refused to answer, since this is a machine-learning methodology question, not something the policy knowledge base was written to address. This is the clearest case of the system correctly distinguishing "the retrieved documents are topically related but do not actually answer this" from "the documents answer this."

## Evaluation Methodology Note

An earlier automated pass mis-classified 3 answers as refusals, based on a keyword check for the phrase "cannot confirm" anywhere in the answer text. Manual review revealed one of these (RAG15) was not a refusal at all, it was a correct answer explaining the RAG system's refusal policy, which happens to contain that same phrase. The evaluation script was corrected to check only whether an answer begins with a refusal, rather than merely containing refusal-adjacent language. This is worth documenting as a reminder that automated evaluation heuristics require the same scrutiny as the system under test, a naive keyword check would have overstated the refusal rate and misrepresented the system's actual behavior.

## Interpretation

13/15 (87%) confidently correct, source-cited answers, with the remaining 2/15 being appropriate refusals rather than failures, represents strong evidence that the RAG pipeline built in Phase 10 generalizes well beyond the small set of ad-hoc questions tested there. No hallucinated or fabricated policy claims were observed across all 15 independently-authored test questions.