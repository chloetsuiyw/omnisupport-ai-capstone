# Phase 4 — Regression: resolution_time_hours

## Setup

Feature matrix built from the leakage-safe feature set defined in Phase 3 (19 features, 996,000 rows post-cleaning). Data split 70/15/15 into train/validation/test using a random split (no temporal ordering requirement identified in the brief, and no meaningful seasonality expected in this synthetic dataset, so a random split was used rather than a more complex time-based split).

## Baseline

A DummyRegressor predicting the mean of resolution_time_hours for every ticket was used as the baseline, per the recommendation to always compare real models against a naive reference point.

- MAE: 6.77 hours
- RMSE: 14.31 hours

## Model: Random Forest

A RandomForestRegressor (100 trees, max depth 12) was trained on the same feature set, using one-hot encoding for categorical features via a ColumnTransformer pipeline.

- MAE: 3.76 hours (a 44% reduction vs. baseline)
- RMSE: 12.42 hours (a 13% reduction vs. baseline)

## Interpretation

The substantial MAE improvement alongside the much smaller RMSE improvement indicates the model has learned meaningful signal for typical tickets, but continues to struggle on extreme long-duration cases. This is consistent with the data audit finding that a small cluster of tickets (348 rows, resolution time > 300 hours) correspond exactly to open_after_7d cases, rare events that are inherently harder to predict from ticket-creation-time features alone, and which disproportionately inflate RMSE relative to MAE due to RMSE's squared-error penalty on large misses.

## Feature Importances

The Random Forest's feature importances show issue_category dominating prediction, particularly the lost_parcel category (23.8% importance), the single strongest predictor by a wide margin. order_value_capped is the strongest numeric feature (12.6%). Collectively, issue_category dummy variables account for more predictive weight than any other feature group, indicating that what kind of issue a ticket represents matters more to resolution time than continuous variables like tenure or delivery delay.

## Error Analysis by Segment

**By issue_category:** error is fairly consistent across categories, ranging narrowly from 3.63 to 4.02 hours MAE. No category shows severe, isolated failure. Notably, lost_parcel, despite being the dominant feature driving predictions overall, has the lowest segment error (3.64 hours). Suggesting this category follows a strong, learnable resolution-time pattern. The highest-error categories, account_access (4.02) and product_question (3.85), likely depend on factors not captured in the current feature set (e.g. account complexity, agent expertise required), which the model cannot yet account for.

**By priority:** counter-intuitively, high (3.85) and urgent (3.80) priority tickets show higher error than low (3.77) and medium (3.74) priority tickets. This suggests that urgent/high-priority resolution time may be driven more by situational factors, specialist availability, case complexity, than by patterns the model can learn from the available ticket-creation-time features. This is a genuine limitation worth flagging to stakeholders: the model is less reliable exactly where accurate prediction matters most for staffing decisions.

## Interpretation Summary

Overall segment-level error is narrow (roughly 3.6-4.0 hours across all categories and priorities), indicating the model does not exhibit severe bias toward or against any particular ticket type. The two counter-intuitive findings above (lost_parcel's low error despite high importance; high/urgent priority's elevated error) are worth surfacing explicitly in the business write-up, since they run against naive expectations and point to specific, actionable limitations rather than a uniform "model works well" claim.

## Controlled Tuning: max_depth

A controlled experiment varying only max_depth (6, 12, 20, unlimited), holding all other hyperparameters fixed, produced:

| max_depth | MAE | RMSE |
|---|---|---|
| 6 | 4.18 | 12.30 |
| 12 | 3.76 | 12.42 |
| 20 | 3.87 | 12.99 |
| None | 3.94 | 13.08 |

MAE and RMSE disagree on the optimal depth: depth=6 minimizes RMSE (by making more conservative, average-leaning predictions that avoid large misses on rare extreme cases), while depth=12 minimizes MAE (by capturing more nuance for typical tickets, at some cost on extreme cases). Beyond depth 12, both metrics degrade monotonically, indicating overfitting as tree depth increases.

**Decision:** max_depth=12 was retained as the final model. Since the business use case is proactive staffing for the general ticket population rather than minimizing worst-case error on rare extreme-duration tickets, MAE, the more interpretable, business-aligned metric, was prioritized over RMSE in this trade-off.