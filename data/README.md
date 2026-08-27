# Data Assets

## Primary support dataset
`raw/support_records_part_001.parquet` through `support_records_part_010.parquet` are the only primary tabular shards. Each contains 100,000 rows; together they contain exactly **1,000,000 records**.

`raw/dataset_preview.csv` is a small convenience preview and is not additional primary data.

The raw data deliberately contains controlled quality problems: blank/missing text, numeric NaNs, exact duplicates, outliers, inconsistent categories/casing, class imbalance, noisy ticket text, subgroup imbalance, predictive signal and post-outcome leakage fields. Your task is to detect and handle these appropriately rather than assume the data is clean.

## Transformer fine-tuning subset
`subsets/transformer_finetune_10000.parquet` contains 10,000 ticket text/label examples selected from the supplied primary data for the compulsory small Hugging Face fine-tuning task. The preview CSV is only for inspection.

## Agent operational store
`agent_store/orders.csv`, `customers.csv`, and `returns.csv` provide a small local operational layer for tool-calling. Use these files rather than scanning the million-row analytical extract on every agent tool call.

## Images
`images/` contains 480 synthetic return images in four classes. `image_labels.csv` maps image IDs to labels and relative paths.

All records and identifiers are synthetic. No real customer personal data is included.
