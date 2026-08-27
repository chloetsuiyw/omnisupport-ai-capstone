# Phase 8a — Classical NLP: TF-IDF Ticket Classification

## Setup

10,000-row subset of ticket descriptions, classifying into 10 issue_category values. Split 80/20 train/validation, stratified to preserve class proportions.

## BoW vs. TF-IDF Comparison

Bag-of-words and TF-IDF vectorization (both capped at 5,000 features) were compared using a LogisticRegression classifier:

| Vectorizer | Macro F1 |
|---|---|
| BoW | 0.9925 |
| TF-IDF | 0.9925 |

Both methods produced identical scores. Investigation into the underlying text (see Text Diversity Check below) explains this: the dataset uses a small, fixed set of templated phrases per category, layered with random order values and greeting/closing variations. Since a small number of category-defining words appear consistently and near-exclusively within each class, raw word-count (BoW) and frequency-weighted (TF-IDF) representations carry equivalent discriminative signal. The templated structure removes any advantage TF-IDF's weighting would typically provide over simpler counting.

## Text Diversity Check

91.48% of descriptions are unique strings (due to randomized order values and phrase combinations), but inspection of same-category examples confirmed a small, repeating set of template phrases per class (e.g. "the box was crushed and the item is broken", "there is visible damage", both mapping to damaged_item). This is consistent with the synthetic generation process producing highly separable text data, mirroring the near-perfect separability already observed in Phase 4 (classical ML) and Phase 7 (computer vision).

## Final Classifier and Error Analysis

The TF-IDF classifier achieved 99% accuracy and 0.99-1.00 macro F1 across all 10 classes. All 16 misclassifications in the validation set were fully explained by one cause: blank issue_description text. Of 21 blank-text tickets in the validation set, 16 were misclassified, the remaining 5 happened to have damaged_item as their true label, which coincided with the model's learned default prediction for zero-information inputs (a TF-IDF vector of all zeros produces a prediction driven entirely by the classifier's intercept term).

This is a complete and well-evidenced explanation for the model's only failure mode: no text classifier, regardless of architecture, could correctly classify a ticket with no description text. This connects directly to the Phase 2 data audit, which identified 9,971 blank issue_description values (1.0%) across the full 1,000,000-row dataset, a pre-existing data quality issue, not a modeling shortcoming.

## Business Implication

Since classification performance is effectively perfect for any ticket containing real text, and the only failure mode is a known, documented data quality gap, the practical recommendation is to address blank-description tickets upstream (e.g. flagging them at intake for the customer to provide detail) rather than attempting to improve classifier architecture, which would not meaningfully change performance given the errors are entirely explained by missing input, not model limitations.