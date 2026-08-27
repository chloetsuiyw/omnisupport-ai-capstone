# Phase 7 — Computer Vision: Return Image Classification

## Setup

480 synthetic return images across 4 balanced classes (120 each: damaged, normal, packaging_damage, wrong_item), split 70/15/15 (train/val/test). Training data used augmentation (horizontal flip, rotation, color jitter); validation and test sets used no augmentation, to reflect genuine, unmodified evaluation conditions.

## Basic CNN (Trained from Scratch)

A small 3-layer convolutional network (Conv→ReLU→MaxPool ×3, followed by a dense classifier with dropout) was trained for 20 epochs.

Validation accuracy reached 100% by epoch 4 and remained at or near 100% for the remainder of training. Test set evaluation confirmed this: perfect precision, recall, and F1 across all four classes, with zero misclassifications in the confusion matrix.

This result was scrutinized rather than accepted at face value. Given the small dataset size and the synthetic nature of the images, this near-perfect performance most likely reflects a genuinely easy, highly visually separable classification task, consistent with even the lowest-confidence correct predictions still scoring 97.7%+ confidence, rather than a modeling error. This should be treated as a property of this specific synthetic dataset, not evidence the model would generalize equally well to real-world return photos, which would likely include more variation in lighting, angle, and background.

## Transfer Learning (ResNet18, Frozen Backbone)

A pretrained ResNet18 (ImageNet weights, all convolutional layers frozen, only the final classification layer retrained) was trained for 10 epochs on the same data.

Test accuracy: 96%, precision/recall/F1 broken down as follows:

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| damaged | 0.82 | 1.00 | 0.90 |
| normal | 1.00 | 0.81 | 0.90 |
| packaging_damage | 1.00 | 1.00 | 1.00 |
| wrong_item | 1.00 | 1.00 | 1.00 |

## Comparison and Interpretation

Counter-intuitively, the transfer learning model performed worse than the basic CNN trained from scratch (96% vs. 100% accuracy). All errors were concentrated in a single confusion pair: some normal images were misclassified as damaged, while packaging_damage and wrong_item remained perfectly classified by both models.

This is explained by two factors specific to this dataset, firstly, ResNet18's pretrained features come from natural real-world ImageNet photos, and with the backbone frozen, only the final layer could adapt to the synthetic damage/normal distinction, a mismatch between pretrained feature space and this task's visual cues; and secondly given the basic CNN already achieved perfect performance, the underlying classification task is evidently simple enough that a small, task-specific network can fully solve it, leaving little room for transfer learning's typical advantage (leveraging rich pretrained features when task-specific data is limited) to provide additional benefit.

**Business takeaway:** transfer learning is not automatically superior, its value depends on how well the pretrained model's learned features align with the specific task and how much complexity the task actually requires. For this particular, highly separable synthetic classification problem, the simpler from-scratch CNN was both faster to train and more accurate.

## Reproducibility

torch.manual_seed(42) set for consistent initialization and data splits across runs. Training/validation loss curves for both models saved to outputs/04_computer_vision/.