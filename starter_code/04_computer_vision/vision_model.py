"""Computer-vision starter.
TODO: load the supplied synthetic images, train a basic CNN, then compare with transfer learning. Include augmentation, confusion matrix, and error analysis.
"""

from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

torch.manual_seed(42)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def load_labels():
    labels_path = PROJECT_ROOT / "data" / "image_labels.csv"
    df = pd.read_csv(labels_path)
    return df

class ReturnImageDataset(Dataset):
    def __init__(self, df, project_root, transform=None):
        self.df = df.reset_index(drop=True)
        self.project_root = project_root
        self.transform = transform
        self.classes = sorted(df["label"].unique())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.project_root / row["relative_path"]
        image = Image.open(img_path).convert("RGB")
        label = self.class_to_idx[row["label"]]
        if self.transform:
            image = self.transform(image)
        return image, label

def get_transforms(augment=False):
    if augment:
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def create_dataloaders(labels_df, project_root, batch_size=16):
    train_df = labels_df.sample(frac=0.7, random_state=42)
    remaining = labels_df.drop(train_df.index)
    val_df = remaining.sample(frac=0.5, random_state=42)
    test_df = remaining.drop(val_df.index)

    train_ds = ReturnImageDataset(train_df, project_root, transform=get_transforms(augment=True))
    val_ds = ReturnImageDataset(val_df, project_root, transform=get_transforms(augment=False))
    test_ds = ReturnImageDataset(test_df, project_root, transform=get_transforms(augment=False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_ds.classes

class BasicCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128 -> 64
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64 -> 32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32 -> 16
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_cnn(model, train_loader, val_loader, epochs=20, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_losses.append(loss.item())
                preds = outputs.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        train_loss = sum(train_losses) / len(train_losses)
        val_loss = sum(val_losses) / len(val_losses)
        val_acc = correct / total
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"Epoch {epoch+1}/{epochs} — train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")

    return model, history

def evaluate_model(model, test_loader, classes):
    model.eval()
    all_preds = []
    all_labels = []
    all_images = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(y_batch.tolist())
            all_images.extend(X_batch)
    return all_preds, all_labels, all_images

def find_low_confidence_predictions(model, test_loader, classes, top_n=5):
    model.eval()
    results = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            probs = torch.softmax(outputs, dim=1)
            confidences, preds = probs.max(dim=1)
            for i in range(len(y_batch)):
                results.append({
                    "true_label": classes[y_batch[i].item()],
                    "predicted_label": classes[preds[i].item()],
                    "confidence": confidences[i].item(),
                })
    results_df = pd.DataFrame(results).sort_values("confidence")
    return results_df.head(top_n)

import matplotlib.pyplot as plt

def plot_training_curve(history, save_path="outputs/04_computer_vision/basic_cnn_training_curve.png"):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(history["train_loss"], label="train_loss")
    axes[0].plot(history["val_loss"], label="val_loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(history["val_acc"], label="val_acc")
    axes[1].set_title("Validation Accuracy")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved {save_path}")

from torchvision import models

def build_transfer_model(num_classes=4):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

if __name__ == "__main__":
    labels_df = load_labels()
    print(labels_df.shape)
    print(labels_df["label"].value_counts())

    train_loader, val_loader, test_loader, classes = create_dataloaders(labels_df, PROJECT_ROOT)
    print("\nClasses:", classes)
    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))
    print("Test batches:", len(test_loader))

    X_batch, y_batch = next(iter(train_loader))
    print("\nBatch shape:", X_batch.shape)
    print("Labels sample:", y_batch[:5])

    print("\n--- Training Basic CNN ---")
    model = BasicCNN(num_classes=len(classes))
    model, history = train_cnn(model, train_loader, val_loader, epochs=20, lr=0.001)

    print("\n--- Evaluating on Test Set ---")
    test_preds, test_labels, test_images = evaluate_model(model, test_loader, classes)

    from sklearn.metrics import classification_report, confusion_matrix
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=classes))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(test_labels, test_preds)
    print(pd.DataFrame(cm, index=classes, columns=classes))

    print("\n--- Lowest-Confidence Correct Predictions ---")
    print(find_low_confidence_predictions(model, test_loader, classes, top_n=5))

    plot_training_curve(history)

    print("\n--- Training Transfer Learning Model (ResNet18) ---")
    transfer_model = build_transfer_model(num_classes=len(classes))
    transfer_model, transfer_history = train_cnn(transfer_model, train_loader, val_loader, epochs=10, lr=0.001)
    plot_training_curve(transfer_history, save_path="outputs/04_computer_vision/transfer_model_training_curve.png")

    print("\n--- Evaluating Transfer Model on Test Set ---")
    transfer_preds, transfer_labels, _ = evaluate_model(transfer_model, test_loader, classes)
    print("\nClassification Report (Transfer Learning):")
    print(classification_report(transfer_labels, transfer_preds, target_names=classes))