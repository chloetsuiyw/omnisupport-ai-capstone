# Phase 8c — Attention Worked Example (Mandatory)

## Setup

A minimal, from-scratch self-attention mechanism was implemented (not using a pretrained model) to demonstrate the core mechanics: token embedding, sinusoidal positional encoding, learned Query/Key/Value projections, scaled dot-product attention, and softmax normalization. This was applied to one real-style support message: "delivery driver dropped my parcel outside despite clear instructions" (9 tokens).

## Mechanism
1. Embedding: each token is mapped to an 8-dimensional vector.
2. Positional encoding: sinusoidal encoding (as used in the original Transformer architecture) is added to each embedding, injecting information about token order, without this step, the model would have no way to distinguish "driver dropped parcel" from "parcel dropped driver."
3. Q/K/V projections: three separate learned linear transformations project each token's embedding into a Query, Key, and Value vector.
4. Scaled dot-product attention: each token's Query is compared against every token's Key (via dot product, scaled by 1/√d_model), producing a raw compatibility score, then normalized via softmax so each row sums to 1, representing how much attention that token pays to every other token (including itself).

## Attention Weight Matrix

The resulting 9×9 attention matrix (rows = query token, columns = key token) is visualized as a heatmap in outputs/06_transformers/attention_heatmap.png. Notably, all tokens attend most strongly to "despite" (attention weights of 0.20-0.34, versus 0.03-0.18 for other tokens). Since the Q/K/V weight matrices here are randomly initialized rather than trained on any task, this specific pattern is not linguistically meaningful, it reflects the random initialization, not learned relationships. This is stated explicitly rather than over-interpreted: the worked example demonstrates the mechanism correctly (valid softmax-normalized attention distributions, full token-to-token relationship computation), while acknowledging that meaningful, interpretable attention patterns only emerge after training on real language data, as happens inside pretrained models like BERT.

## Contrast with TF-IDF

The same message was represented via TF-IDF for direct comparison:

| Representation | Structure | Word Order | Token Relationships |
|---|---|---|---|
| TF-IDF | Bag-of-words vector | Not captured | None — each word is independent |
| Self-Attention | Token-to-token weight matrix | Captured via positional encoding | Full pairwise relationships between all tokens |

TF-IDF assigned every token in this message an identical weight (0.333), since it does not use stopword removal and treats all remaining words as equally important, independent, and order-less. This is the fundamental limitation attention addresses: TF-IDF cannot represent that "dropped" relates more directly to "parcel" than to "clear," nor can it distinguish this message from a random reordering of the same words. Self-attention explicitly computes a relationship score between every pair of tokens and preserves positional information, providing a structurally richer representation, the foundation that enables transformers to capture context, word order, and long-range dependencies that bag-of-words methods fundamentally cannot represent.