# Phase 6 — Deep Learning: Tabular Neural Network

## Setup

A small feedforward neural network (Linear(61→64) → ReLU → Linear(64→32) → ReLU → Linear(32→1)) was trained on the same leakage-safe feature set as Phase 4's regression task, predicting resolution_time_hours. Features were standardized (numeric) and one-hot encoded (categorical) via a ColumnTransformer, producing a 61-dimensional input. The same 70/15/15 train/val/test split was used for direct comparability with Phase 4. L1Loss (MAE) was used as the training objective to match Phase 4's primary evaluation metric.

## Training and Overfitting Discussion

Across all runs, training and validation loss tracked closely together throughout training, with no meaningful divergence, the classic signature of overfitting (train loss continuing to drop while validation loss rises) was not observed. This suggests the small network architecture and dataset size did not lead to overfitting within 15 epochs, though it also means the model was not aggressively fitting to the training data at all.

## Learning Rate Comparison

A controlled experiment varying only the learning rate (0.01, 0.001, 0.0001), with all other settings fixed, was run for 15 epochs each:

| Learning Rate | Final Val Loss (MAE) | Convergence Speed |
|---|---|---|
| 0.01 | 3.5005 | Fast (plateaus by epoch 8-9) |
| 0.001 | 3.5127 | Fast (plateaus by epoch 2-3) |
| 0.0001 | 3.5037 | Slow (still descending at epoch 15) |

All three learning rates converged to a very similar final loss (~3.50), despite reaching it via different trajectories. This is a meaningful finding: it indicates the ~3.50 MAE plateau is not an artifact of a poorly chosen learning rate, but reflects a genuine ceiling for this model architecture and feature set, further optimization tuning is unlikely to substantially improve results without changing the model capacity or feature engineering.

## Comparison to Phase 4 (Random Forest)

The neural network's converged MAE (~3.50) is comparable to, and marginally better than, the Random Forest's MAE of 3.76 from Phase 4. However, this comparison should be read cautiously: the neural network plateaued almost immediately (by epoch 2-3 in most runs) rather than showing a clear learning curve of substantial, sustained improvement, suggesting it converged to a relatively simple solution rather than capturing meaningfully more complex patterns than the tree-based model. Given the added complexity and training time of the neural network for a marginal (and possibly not meaningfully different, given the model's simplicity) improvement, the Random Forest remains the more practical choice for this specific business use case, with the neural network serving primarily as a useful comparison point and methodology exercise.

## Checkpoint

The final model (trained with the best-performing learning rate, 0.01) was saved to outputs/03_deep_learning/tabular_nn_checkpoint.pt via torch.save(model.state_dict(), ...), enabling reproducible reloading without retraining.

## Reproducibility

torch.manual_seed(42) and np.random.seed(42) were set at the top of the script to ensure consistent data splits and initialization across runs.