# ML Evaluation Requirements

Students must select evaluation metrics that match the business risk of each task rather than reporting accuracy alone.

Minimum evidence:
- Baseline result before tuning.
- Train/validation/test or equivalent holdout reasoning.
- Cross-validation for at least one classical ML comparison.
- Confusion matrix and precision/recall/F1 for escalation or another classification target.
- MAE and RMSE for the regression task.
- Threshold analysis where the classification model exposes probabilities.
- Error analysis with at least three meaningful failure patterns.
- Leakage audit explaining which raw fields were excluded and why.
- Subgroup comparison using at least one synthetic demographic/support attribute.
