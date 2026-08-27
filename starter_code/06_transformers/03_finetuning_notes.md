# Phase 8e — Fine-Tuning DistilBERT (Mandatory, Session 22)

## Setup and Compute Decisions

Given CPU-only training, deliberate compute-time decisions were made rather than attempting the full 10,000-row subset: a random sample of 1,500 training examples and 300 validation examples, 2 training epochs, max sequence length 64 tokens, batch size 16. These choices are documented explicitly as compute-time trade-offs, not oversights, training on the full subset would likely have taken proportionally longer with limited additional benefit, given the task's demonstrated separability (see Phase 8a).

Model: distilbert-base-uncased, with a freshly initialized classification head (pre_classifier and classifier layers) replacing the original masked-language-model head, the standard fine-tuning approach for adapting a pretrained language model to a new downstream classification task.

## Training

Training completed in approximately 5.5 minutes (338 seconds) for 2 epochs (188 steps total) on CPU, faster than initially estimated. Training loss decreased smoothly and consistently: from 2.10 (near-random, consistent with 10-way classification) at the start, down to 0.036 by the end of training, with no instability or divergence observed. Validation loss followed the same pattern (0.094 after epoch 1, 0.031 after epoch 2), tracking training loss closely with no sign of overfitting within this short training run.

## Results

| Model | Macro F1 |
|---|---|
| TF-IDF + Logistic Regression (Phase 8a) | 0.9925 |
| Fine-tuned DistilBERT (this phase) | 1.0000 |

The fine-tuned transformer slightly outperformed the classical TF-IDF baseline, achieving perfect macro F1 on the 300-example validation set.

## Interpretation

Consistent with prior findings in this project (perfect image classification in Phase 7, near-perfect TF-IDF classification in Phase 8a), this result reflects the genuinely high separability of this synthetic dataset's category-defining vocabulary, rather than indicating the fine-tuned model captured meaningfully deeper linguistic understanding than TF-IDF. Given TF-IDF already achieved 0.9925 macro F1 with dramatically less compute (seconds vs. minutes) and no GPU/fine-tuning infrastructure required, the marginal F1 improvement from fine-tuning does not, on its own, justify the added complexity and training cost for this specific task.

The genuine business case for fine-tuning over TF-IDF would emerge on messier, more naturally varied real-world ticket text, where TF-IDF's reliance on literal keyword overlap would degrade faster than a fine-tuned transformer's contextual understanding, similar to the semantic search vs. keyword-matching contrast demonstrated in Phase 8b. This synthetic dataset's templated structure does not exercise that advantage.

## Reproducibility

random_state=42 used for all data sampling and splitting; deterministic model initialization not separately seeded but training completed in a single deterministic run given fixed sample selection.