# Phase 2 — Data Audit Summary
## Dataset Overview

The primary dataset comprises 1,000,000 support ticket records across 10 Parquet shards (100,000 rows each), confirmed complete via row-count validation. After cleaning, 996,000 rows remain.

## Missingness

Two distinct forms of missingness were identified. csat_score contains 56,852 true NaN values (5.69%). This is expected, since satisfaction scores are only collected for tickets where the customer responded to a follow-up survey. Separately, three text columns contained missingness encoded as blank strings rather than NaN, which a standard .isna() check alone would have missed: customer_region (15,286 blank, 1.5%) and issue_description (9,971 blank, 1.0%), both flagged in the data dictionary as columns that "may contain missing values." A third, image_id (999,120 blank, 99.9%), was investigated separately below and found to be structural rather than a quality issue.

## Image Linkage

Cross-tabulating image_id against attachment_available showed that while 193,468 tickets have attachment_available = 1, only 880 of these actually link to one of the 480 supplied synthetic images. All 480 images are referenced, at an average of ~1.8 tickets per image. The remaining ~192,588 "has attachment" tickets reference attachment types outside the supplied image set (e.g. documents). This defines the true usable subset for the Phase 7 computer vision task as the 880 image-linked rows, not the full attachment-flagged population.

## Duplicates

4,000 exact duplicate rows were identified (0.4% of the dataset). Full-row duplicate count and ticket_id-only duplicate count were exactly equal, confirming these are simple re-ingestion duplicates (identical data under a repeated ID) rather than conflicting records sharing an ID. All 4,000 were removed via drop_duplicates().

## Category Consistency

Three columns contained casing inconsistencies representing the same underlying category: support_channel ("Web Chat" vs "web_chat", 1,961 rows), customer_region ("london" vs "London", 1,099 rows), and product_category ("Electronics" vs "electronics", 913 rows). All were merged into a single consistent casing per category.

## Outliers

Three numeric columns were assessed using the IQR method, with column-specific interpretation rather than a uniform rule:

- order_value (6.68% flagged, max £21,990.51 against a typical £58-£185 range): treated as genuine extreme values rather than errors, consistent with the data dictionary's note on "rare extreme values." Rather than dropping these rows, order_value was capped at the 99th percentile (£768.15) for modelling, with a companion binary flag (order_value_was_capped) retained so models can still learn from the fact that a value was extreme, without extreme values dominating splits or coefficients.
- delivery_delay_days (9.21% flagged): IQR produced a negative lower bound (-3.5 days), which is meaningless for a floor-bounded count variable. This confirmed that IQR is not a diagnostically meaningful method for this column. The flagged rows reflect the expected right-skew of a delay variable (most orders on time, a long tail of delayed ones), not data errors. No transformation was applied. This is treated as genuine business signal.
- resolution_time_hours (3.43% flagged, max 899 hours): this is a model target, so no capping or removal was applied to avoid biasing what the model is meant to predict. Further investigation (below) explained the extreme tail directly.

## Long-Tail Resolution Time Investigation

A log-scale histogram of resolution_time_hours revealed a distinct, separate cluster of values above ~300 hours, rather than a smooth long tail. Cross-tabulation confirmed all 348 rows in this cluster have resolution_status_after_7d = "open_after_7d", a definitional relationship (tickets still open after 7 days necessarily show long resolution times), not a data quality artifact. Notably, only 27.9% of these long-open tickets were also escalated, indicating that prolonged resolution time and escalation are not as tightly linked as might be assumed.

## Class Balance

The escalated target shows a 69.04% / 30.96% split (not escalated / escalated). This is a moderate, not severe, imbalance. Sufficient to make plain accuracy a misleading metric (a model predicting the majority class alone would score 69%), but not severe enough to require resampling techniques such as SMOTE. PR-AUC and F1 on the escalated class, alongside a probability-threshold table, were selected as the primary evaluation metrics for Phase 4 (see Phase 1 framing).

## Notable Business Finding

Escalation rate varies substantially by issue_category: lost_parcel tickets escalate at ~64%, nearly double the next-highest category (delivery_late, ~37%), and roughly 2.7x the rate of the lowest categories (~24%). This is a strong candidate feature for the Phase 4 classification model and a standalone finding worth flagging to the business. Lost-parcel tickets may warrant direct routing to specialist queues rather than standard triage.