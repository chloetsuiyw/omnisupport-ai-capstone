# Phase 5 — Customer Segmentation (Clustering)

## Setup

Ticket-level data was aggregated to one row per customer (198,643 unique customers, median 5 tickets each) across seven features: total ticket count, average order value, average delivery delay, escalation rate, average resolution time, average tenure, and refund rate. Features were standardized via StandardScaler prior to clustering, since KMeans is distance-based and features here vary widely in scale (e.g. tenure in months vs. rates between 0-1).

## Choosing k

KMeans was compared across k = 2 to 6, using silhouette score (sampled at 20,000 customers for compute efficiency) as the primary selection criterion, since inertia mechanically decreases with more clusters and is not a reliable standalone signal. Silhouette scores were uniformly low across all k tested (0.11-0.16), indicating that customer behavior in this dataset varies along a continuum rather than falling into sharply separated groups, wwhich is a realistic property of real-world (or realistic synthetic) customer data, not a modeling failure.

k=4 was selected as a balance between interpretability and cluster quality. k=2 scored marginally higher on silhouette but produced an oversimplified two-way split; k=4 provides a genuinely usable number of distinct, describable personas while remaining close to the observed silhouette ceiling.

## Cluster Profiles

| Cluster | Size | Tickets/Customer | Avg Order Value | Avg Delivery Delay | Escalation Rate | Avg Resolution Time | Refund Rate |
|---|---|---|---|---|---|---|---|
| 0 | 68,996 | 7.30 | £144 | 2.80 days | 30.5% | 23.7 hrs | 32.0% |
| 1 | 53,547 | 3.93 | £186 | 3.92 days | 55.3% | 27.1 hrs | 31.4% |
| 2 | 75,850 | 3.70 | £122 | 2.25 days | 14.0% | 21.5 hrs | 32.4% |
| 3 | 250 | 5.23 | £140 | 3.12 days | 32.0% | 170.5 hrs | 31.0% |

## Cluster Validation

Cluster 3, though small (250 customers), showed a dramatically elevated average resolution time (170.5 hours vs. 21-27 hours in other clusters). Cross-referencing confirmed 100% of cluster 3 customers had at least one ticket with resolution_status_after_7d = "open_after_7d", the same rare data pattern identified independently during the Phase 2 audit. This is a meaningful validation signal: the clustering algorithm, working purely from aggregated numeric features, rediscovered a structural pattern already known from the manual audit, supporting confidence that the clustering reflects genuine structure rather than arbitrary partitioning.

## Cautious Personas

Note: the following personas are descriptive summaries of customers who share similar patterns in the observed data. They do not claim that any feature causes another, for example, higher order values and higher escalation rates co-occurring in Cluster 1 does not establish that larger orders cause escalations, only that customers with these characteristics tend to appear together in this segment.

- **Cluster 0 — "Frequent Contacters":** customers who raise tickets more often than average (7.3 vs. the ~5.0 dataset mean) but show broadly typical escalation and resolution outcomes. This group may represent customers who reach out proactively or have ongoing needs, rather than customers experiencing unusually poor service.
- **Cluster 1 — "High-Value, High-Friction":** the highest average order value (£186) paired with the highest escalation rate (55.3%) and longest average delivery delay (3.9 days). This segment may warrant closer monitoring, though the data does not establish whether higher spend leads to different handling, or whether these customers simply purchase categories more prone to delivery issues.
- **Cluster 2 — "Low-Friction":** the largest segment (75,850 customers), with the lowest escalation rate (14.0%) and shortest delivery delays. These appear to be the most straightforward customers to serve, though this may partly reflect lower-value or simpler order types rather than differences in customer behavior itself.
- **Cluster 3 — "Stuck Tickets" (rare, high-priority):
** a very small group (250 customers, 0.13% of the base) defined almost entirely by extreme resolution times, directly linked to the open_after_7d cases found in the Phase 2 audit. Given the exact overlap with a known data issue, this cluster may be better understood as an operational edge case requiring investigation (e.g. tickets stuck in a workflow bottleneck) rather than a distinct customer type.

## Summary

Clustering surfaced four customer segments of genuinely different scale and character, ranging from a large, low-friction majority to a tiny, high-severity group matching a known data anomaly. Silhouette scores indicate these boundaries are soft rather than sharp, appropriate given the moderate correlation-based signal seen throughout Phase 4, and personas are presented descriptively, without causal claims, per the phase's explicit guidance.