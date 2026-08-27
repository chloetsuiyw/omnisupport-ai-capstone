"""Deep-learning starter.
TODO: create dataset/dataloader logic, a small neural network, training loop, validation tracking, loss/optimizer experiments, checkpointing, and reproducibility controls.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01_data"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "02_ml"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from data_audit import load_full_dataset, clean_and_prepare
from feature_engineering import build_features, get_feature_columns

torch.manual_seed(42)
np.random.seed(42)

class TicketDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def prepare_data():
    df = load_full_dataset()
    df = clean_and_prepare(df)
    df = build_features(df)

    X_raw = df[get_feature_columns()]
    y = df["resolution_time_hours"].values

    cat_cols = X_raw.select_dtypes(include=["str", "object"]).columns.tolist()
    num_cols = [c for c in X_raw.columns if c not in cat_cols]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])

    X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(X_raw, y, test_size=0.3, random_state=42)
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(X_temp_raw, y_temp, test_size=0.5, random_state=42)

    X_train = preprocessor.fit_transform(X_train_raw)
    X_train = X_train.toarray() if hasattr(X_train, "toarray") else X_train

    X_val = preprocessor.transform(X_val_raw)
    X_val = X_val.toarray() if hasattr(X_val, "toarray") else X_val

    X_test = preprocessor.transform(X_test_raw)
    X_test = X_test.toarray() if hasattr(X_test, "toarray") else X_test

    return X_train, X_val, X_test, y_train, y_val, y_test, X_train.shape[1]

def build_model(input_dim, output_dim=1):
    return nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, output_dim),
    )

def train_model(model, train_loader, val_loader, epochs=15, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()  # MAE, matches Phase 4's metric for direct comparison

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch+1}/{epochs} — train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")

    return model, history

def compare_learning_rates(input_dim, train_loader, val_loader, lrs=[0.01, 0.001, 0.0001], epochs=15):
    results = {}
    for lr in lrs:
        print(f"\n=== Training with lr={lr} ===")
        model = build_model(input_dim)
        model, history = train_model(model, train_loader, val_loader, epochs=epochs, lr=lr)
        results[lr] = history
    return results

def save_checkpoint(model, path="outputs/03_deep_learning/tabular_nn_checkpoint.pt"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Checkpoint saved to {path}")

if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, input_dim = prepare_data()
    print("Input dim:", input_dim)
    print("Train/Val/Test sizes:", len(X_train), len(X_val), len(X_test))

    train_ds = TicketDataset(X_train, y_train)
    val_ds = TicketDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    lr_results = compare_learning_rates(input_dim, train_loader, val_loader, lrs=[0.01, 0.001, 0.0001], epochs=15)

    print("\n--- Final val_loss by learning rate ---")
    for lr, history in lr_results.items():
        print(f"lr={lr}: final val_loss={history['val_loss'][-1]:.4f}")

        best_lr = min(lr_results, key=lambda lr: lr_results[lr]["val_loss"][-1])
    print(f"\nBest learning rate by final val_loss: {best_lr}")

    final_model = build_model(input_dim)
    final_model, final_history = train_model(final_model, train_loader, val_loader, epochs=15, lr=best_lr)
    save_checkpoint(final_model)