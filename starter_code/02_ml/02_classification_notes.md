# Phase 4 — Classification: escalated

## Setup

Same leakage-safe feature set as regression (19 features), with resolution_time_hours excluded per the Phase 1 leakage audit, since it is an outcome of the same underlying process as escalation rather than a valid predictor of it. Data split 70/15/15 using a stratified split (stratify=y) to preserve the 69/31 class balance across train, validation, and test sets, confirmed identical escalation rates (0.3096) in both train and validation after splitting.

## Baseline

A DummyClassifier predicting the majority class ("not escalated") for every ticket scored F1 = 0.0, as expected, it never predicts a true escalation, illustrating precisely why plain accuracy (which this baseline would score ~69% on) is a misleading metric for this imbalanced target.

## Model: Random Forest

A RandomForestClassifier (100 trees, max depth 12, class_weight="balanced" to compensate for class imbalance without resampling) achieved:

- F1: 0.557
- PR-AUC: 0.573

This represents genuine, moderate predictive signal, not a dramatic result, but a defensible one, consistent with this being a real (if synthetic) business dataset rather than an artificially easy classification task.

## Threshold Analysis

The model's predicted probabilities ranged from 0.236 to 0.968 (never below ~0.24), a direct effect of class_weight="balanced" shifting the model's confidence floor upward. As a result, thresholds below ~0.3 provide no meaningful discrimination, at threshold 0.2, 100% of tickets are flagged and 100% of escalations are caught trivially, since no tickets are filtered out at all. The meaningful decision range is approximately 0.3-0.7:

| Threshold | % Flagged | % Escalations Caught | Precision |
|---|---|---|---|
| 0.3 | 89.6% | 96.7% | 33.4% |
| 0.4 | 60.8% | 81.2% | 41.3% |
| 0.5 | 38.9% | 62.8% | 50.0% |
| 0.6 | 21.3% | 41.2% | 60.0% |
| 0.7 | 9.1% | 20.9% | 70.8% |

Precision tracks the threshold value almost linearly (e.g. ~50% precision at threshold 0.5, ~71% at threshold 0.7), indicating the model is reasonably well-calibrated within its effective range. This table allows the business to select an operating point based on senior-agent review capacity: for example, flagging ~21% of tickets (threshold 0.6) catches 41% of true escalations at 60% precision.

## Feature Importances

previous_ticket_count dominates (33.0% importance), a customer's history of prior support contact is the strongest signal for escalation risk, markedly different from the regression model, where this feature barely registered. order_value_capped (13.7%) and delivery_delay_days (10.0%) are the next strongest numeric features. issue_category_lost_parcel remains present (6.4%) but is far less dominant here than in the regression model (23.8%), indicating that resolution time and escalation likelihood are driven by substantially different underlying factors.

## Error Analysis by Segment

**By issue_category:** lost_parcel is a clear standout (F1 = 0.79), consistent with its high escalation rate (64.7%) identified both here and in the Phase 2 audit. All other categories cluster more tightly between F1 0.46-0.59, with no other category approaching lost_parcel's performance. The model excels specifically where escalation is both common and evidently predictable, and performs only moderately elsewhere.

**By priority:** high (F1 = 0.68) and urgent (F1 = 0.68) priority tickets are predicted substantially better than low (0.52) and medium (0.49). This is the inverse of the regression model's pattern, where high/urgent priority tickets had the highest error. Together, these findings suggest priority is a strong, learnable signal for whether a ticket will escalate, but a weak signal for how long it will take to resolve, a genuinely useful distinction for the business to understand, since it implies the two prediction tasks should not be assumed to share the same drivers.

## Summary

The classification model provides real, actionable value for escalation triage, performing particularly well on lost-parcel and high/urgent-priority tickets, with a documented, well-calibrated threshold table enabling the business to choose an operating point matched to available senior-agent capacity. The model's limitations are equally well understood: performance is moderate (not dramatic) overall, and weakest on lower-priority, lower-escalation-rate categories where the available features carry less predictive signal.

# Supplementary Analysis: Cross-Validation and Demographic Subgroup Fairness

## 5-Fold Cross-Validation

To verify the model's performance is stable rather than dependent on a single lucky/unlucky train-test split, 5-fold stratified cross-validation was run using macro F1 (averaging across both classes):

| Fold | Macro F1 |
|---|---|
| 1 | 0.6613 |
| 2 | 0.6628 |
| 3 | 0.6585 |
| 4 | 0.6580 |
| 5 | 0.6598 |

Mean macro F1: 0.6601, standard deviation: 0.0018, a very tight spread across folds, indicating the model's performance is stable and not an artifact of a particular data split. Note: macro F1 (0.66) differs from the headline escalated-class F1 (0.557) reported earlier, since macro F1 averages across both classes and is pulled upward by the majority class's strong performance; both metrics are legitimate but measure different things.

## Demographic Subgroup Fairness (customer_region)

F1 was computed separately for each of the 10 UK regions plus the "Unknown" region (customers with missing customer_region, identified during the Phase 2 audit):

| Region | F1 | Count | Escalation Rate |
|---|---|---|---|
| West Midlands | 0.5644 | 14,862 | 31.0% |
| Northern Ireland | 0.5620 | 5,795 | 32.0% |
| Scotland | 0.5609 | 11,655 | 31.0% |
| Yorkshire | 0.5595 | 14,644 | 31.9% |
| South East | 0.5583 | 22,155 | 31.0% |
| East Midlands | 0.5561 | 11,971 | 30.0% |
| South West | 0.5546 | 13,345 | 30.7% |
| London | 0.5542 | 26,359 | 30.7% |
| North West | 0.5527 | 17,663 | 31.1% |
| Wales | 0.5527 | 8,669 | 31.0% |
| Unknown | 0.5505 | 2,282 | 30.2% |

Performance is remarkably consistent across all regions, spanning a narrow range (F1: 0.5505–0.5644, a spread of just 0.014). No region is systematically underserved, and notably, customers with missing region data ("Unknown") perform comparably to fully-identified regions, indicating the model does not penalize customers with incomplete profile information. This is a positive equity finding, the escalation classifier's predictive quality does not vary meaningfully by customer geography.

## Final Held-Out Test Set Confirmation

After all validation-set analysis (threshold selection, feature importances, segment/demographic analysis, cross-validation), the model was evaluated once on the previously untouched test set:

| Metric | Validation Set | Test Set |
|---|---|---|
| F1 (escalated class) | 0.5570 | 0.5577 |
| PR-AUC | 0.5733 | 0.5739 |

Test-set results are consistent with validation-set results to within 0.001, confirming the model was not inadvertently overfit to the validation set through the extensive analysis performed against it, and that the reported performance is honest and reproducible on genuinely unseen data.