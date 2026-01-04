from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


class _TabularDataset(Dataset):
    def __init__(self, X2d: np.ndarray, y1d: np.ndarray):
        self.X = X2d.astype(np.float32)
        self.y = y1d.astype(np.int64)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx])
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int],
        num_classes: int,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()

        act: nn.Module
        act_name = str(activation).lower()
        if act_name == "tanh":
            act = nn.Tanh()
        elif act_name == "gelu":
            act = nn.GELU()
        else:
            act = nn.ReLU()

        layers: list[nn.Module] = []
        prev = int(input_dim)
        for h in hidden_layers:
            h = int(h)
            layers.append(nn.Linear(prev, h))
            layers.append(act)
            if dropout and float(dropout) > 0:
                layers.append(nn.Dropout(p=float(dropout)))
            prev = h
        layers.append(nn.Linear(prev, int(num_classes)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _optimizer_from_cfg(model: nn.Module, model_cfg: dict, train_cfg: dict):
    opt_name = str(model_cfg.get("optimizer", train_cfg.get("optimizer", "adam"))).lower()
    lr = float(model_cfg.get("lr", train_cfg.get("lr", train_cfg.get("learning_rate", 1e-3))))
    weight_decay = float(model_cfg.get("weight_decay", train_cfg.get("weight_decay", 0.0)))

    if opt_name == "sgd":
        momentum = float(model_cfg.get("momentum", train_cfg.get("momentum", 0.9)))
        nesterov = bool(model_cfg.get("nesterov", train_cfg.get("nesterov", False)))
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )

    if opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def run_mlp(
    feature_df: pd.DataFrame,
    y: np.ndarray,
    train_cfg: dict,
    model_cfg: dict,
    standardize: bool,
    random_state: int,
    stratify,
    full_counts: pd.Series,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Train a torch MLP and return (y_test, y_pred, y_proba)."""

    X = feature_df.to_numpy(dtype=np.float32)

    # Label encoding to contiguous indices for CrossEntropyLoss
    classes_sorted = sorted(pd.Series(y).dropna().unique().tolist())
    num_classes = len(classes_sorted)
    label_to_idx = {lbl: i for i, lbl in enumerate(classes_sorted)}
    idx_to_label = {i: lbl for lbl, i in label_to_idx.items()}
    y_idx = np.array([label_to_idx[v] for v in y], dtype=np.int64)

    test_size = float(train_cfg.get("test_size", 0.2))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_idx, test_size=test_size, random_state=random_state, stratify=stratify
    )

    # Optional standardization (fit on train only)
    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_test = scaler.transform(X_test).astype(np.float32)

    train_counts = pd.Series(y_train).value_counts().sort_index()
    test_counts = pd.Series(y_test).value_counts().sort_index()
    print(f"[mi-race][mlp] Total rows: {feature_df.shape[0]}")
    print(f"[mi-race][mlp] Class distribution (train): {train_counts.to_dict()}")
    print(f"[mi-race][mlp] Class distribution (test):  {test_counts.to_dict()}")

    # Warn if some classes missing in training set
    missing_classes = [c for c in full_counts.index if c not in train_counts.index]
    if missing_classes:
        print(f"[mi-race][WARN][mlp] Classes missing from training set: {missing_classes} -> cannot be predicted.")

    hidden_layers = list(model_cfg.get("hidden_layers", [128, 128]))
    activation = str(model_cfg.get("activation", "relu"))
    dropout = float(model_cfg.get("dropout", 0.0))

    batch_size_raw = model_cfg.get("batch_size", train_cfg.get("batch_size", 64))
    if isinstance(batch_size_raw, str) and batch_size_raw.lower() == "auto":
        batch_size = 64
    else:
        batch_size = int(batch_size_raw)
    epochs = int(model_cfg.get("epochs", train_cfg.get("epochs", 15)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(random_state))

    model = MLP(
        input_dim=int(X_train.shape[1]),
        hidden_layers=hidden_layers,
        num_classes=int(num_classes),
        activation=activation,
        dropout=dropout,
    ).to(device)

    optimizer = _optimizer_from_cfg(model, model_cfg, train_cfg)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(_TabularDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(_TabularDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    log_every_epochs = int(model_cfg.get("log_every_epochs", 5))

    try:
        from tqdm.auto import tqdm
    except Exception:  # pragma: no cover
        tqdm = None

    def _eval() -> tuple[float, float]:
        model.eval()
        total_loss = 0.0
        total = 0
        correct = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                bs = int(yb.size(0))
                total_loss += float(loss.item()) * bs
                total += bs
                preds = torch.argmax(logits, dim=1)
                correct += int((preds == yb).sum().item())
        avg_loss = total_loss / max(total, 1)
        acc = correct / max(total, 1)
        model.train()
        return avg_loss, acc

    model.train()
    for ep in range(1, epochs + 1):
        total_loss = 0.0
        total = 0
        correct = 0
        iterator = train_loader
        show_bar = (tqdm is not None) and (log_every_epochs > 0) and (ep % log_every_epochs == 0)
        if show_bar:
            iterator = tqdm(train_loader, desc=f"[mlp] epoch {ep}/{epochs}", leave=False)
        for xb, yb in iterator:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            bs = int(yb.size(0))
            total_loss += float(loss.item()) * bs
            total += bs
            preds = torch.argmax(logits, dim=1)
            correct += int((preds == yb).sum().item())

        avg_loss = total_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        if log_every_epochs > 0 and (ep % log_every_epochs == 0):
            test_loss, test_acc = _eval()
            print(
                f"[mi-race][mlp] epoch {ep}/{epochs}  "
                f"train_loss={avg_loss:.4f} train_acc={train_acc:.4f}  "
                f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
            )

    model.eval()
    y_pred_idx: list[int] = []
    y_proba_list: list[np.ndarray] = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            y_pred_idx.extend(preds.cpu().numpy().tolist())
            y_proba_list.append(probs.cpu().numpy())

    y_pred = np.array([idx_to_label[i] for i in y_pred_idx])
    y_test_labels = np.array([idx_to_label[i] for i in y_test])
    y_proba = np.vstack(y_proba_list) if y_proba_list else None

    return y_test_labels, y_pred, y_proba
