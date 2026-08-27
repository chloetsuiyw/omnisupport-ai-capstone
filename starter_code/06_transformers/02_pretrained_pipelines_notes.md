# Phase 8d — Pretrained Transformer Pipelines

## Setup

Two pretrained Hugging Face pipelines were applied to a sample of real support ticket text, without any fine-tuning or domain-specific training:

- **Sentiment analysis:** distilbert-base-uncased-finetuned-sst-2-english (default sentiment pipeline model)
- **Zero-shot classification:** facebook/bart-large-mnli (default zero-shot pipeline model)

## Results

| Ticket (abbreviated) | Sentiment | Confidence | Zero-shot Top Label | Confidence |
|---|---|---|---|---|
| "product is damaged... confirm next step? ASAP" | NEGATIVE | 0.9997 | damaged item | 0.888 |
| "box was crushed and the item is broken" | NEGATIVE | 0.9964 | damaged item | 0.930 |
| "box was crushed and the item is broken" (variant) | NEGATIVE | 0.9991 | damaged item | 0.881 |

## Interpretation

Both pretrained pipelines performed strongly without any task-specific training. Sentiment analysis correctly identified all three damage-related tickets as strongly negative (>99.6% confidence), consistent with the genuinely negative nature of product-damage complaints. Zero-shot classification, given only a list of candidate category labels, with no training examples, correctly identified "damaged item" as the top label for all three tickets, with 88-93% confidence, despite the model having no prior exposure to this specific dataset or its category taxonomy.

## Business Relevance

Sentiment analysis offers a lightweight, zero-training-cost way to flag high-distress tickets for priority handling, complementing the escalation classifier built in Phase 4. Zero-shot classification is particularly valuable for categories or edge cases not well-represented in training data (e.g. a new product category or an emerging issue type), since it requires no retraining, only an updated candidate label list, making it a practical fallback where the supervised TF-IDF classifier (Phase 8a) would need retraining to adapt.