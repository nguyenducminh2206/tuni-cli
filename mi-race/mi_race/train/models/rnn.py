from __future__ import annotations

from typing import Tuple, Optional
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


def _to_1d_array(x) -> np.ndarray:
    if isinstance(x, (list, tuple, np.ndarray)):
        arr = np.asarray(x, dtype=float)
        return arr.reshape(-1)
    return np.array([], dtype=float)


def _standardize_sequence(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    m = float(np.mean(arr))
    s = float(np.std(arr))
    if s == 0.0:
        return np.zeros_like(arr)
    return (arr - m) / s


def _pad_or_truncate(arr: np.ndarray, length: int, pad_value: float = 0.0) -> np.ndarray:
    if arr.size >= length:
        return arr[:length]
    out = np.full((length,), pad_value, dtype=float)
    out[: arr.size] = arr
    return out


class SeqDataset(Dataset):
    def __init__(self, X2d: np.ndarray, y1d: np.ndarray):
        self.X = X2d.astype(np.float32)
        self.y = y1d.astype(np.int64)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx][None, :])  # (1, T)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y


class RNNClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int, bidirectional: bool, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=(dropout if num_layers > 1 else 0.0),  # LSTM inter-layer dropout only applies when num_layers>1
        )
        feat = hidden_size * (2 if bidirectional else 1)
        self.post_dropout = nn.Dropout(p=float(dropout)) if dropout and dropout > 0 else nn.Identity()
        self.fc = nn.Linear(feat, num_classes)

    def forward(self, x):
        # x: (B, 1, T); transpose to (B, T, 1)
        x = x.transpose(1, 2)
        out, (h_n, c_n) = self.lstm(x)
        # Use last layer's hidden state(s)
        if self.lstm.bidirectional:
            last_fwd = h_n[-2]
            last_bwd = h_n[-1]
            h = torch.cat([last_fwd, last_bwd], dim=1)
        else:
            h = h_n[-1]
        h = self.post_dropout(h)
        logits = self.fc(h)
        return logits


def run_rnn(
    raw_df: pd.DataFrame,
    y: np.ndarray,
    train_cfg: dict,
    model_cfg: dict,
    random_state: int,
    stratify,
) -> Tuple[np.ndarray, np.ndarray]:
    """Train a simple LSTM/GRU-like RNN on a kept sequence column and return (y_test, y_pred).

    Expects the original dataframe with a sequence column (default 'time_trace').
    The sequence is standardized per sample (optional) and padded/truncated to max_len.
    """
    sequence_col = model_cfg.get("sequence_col", "time_trace")
    if sequence_col not in raw_df.columns:
        raise SystemExit(f"[mi-race] RNN requires a sequence column '{sequence_col}' in the dataset.")

    # Prepare sequences
    seq_series = raw_df[sequence_col].apply(_to_1d_array)
    lengths = seq_series.apply(lambda a: a.size)
    if lengths.max() == 0:
        raise SystemExit("[mi-race] RNN: all sequences appear empty.")
    # Log basic sequence stats so users can verify raw sequences are being used
    try:
        print(
            "[mi-race][rnn] Sequence stats:",
            {
                "col": sequence_col,
                "count": int(lengths.shape[0]),
                "len_min": int(lengths.min()),
                "len_max": int(lengths.max()),
                "len_mean": float(lengths.mean()),
            },
        )
    except Exception:
        pass

    # Train/test split on indices to keep alignment
    idx = np.arange(len(raw_df))
    test_size = float(train_cfg.get("test_size", 0.2))
    idx_train, idx_test, y_train, y_test = train_test_split(
        idx, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    # Preprocessing options to speed up training
    downsample_every = int(model_cfg.get("downsample_every", 1))  # keep every k-th sample
    window_len = int(model_cfg.get("window_len", 0))  # 0 disables window cropping
    window_pos = str(model_cfg.get("window_pos", "head")).lower()  # head|center|tail|random

    def _apply_preprocess(arr: np.ndarray, j: int) -> np.ndarray:
        # Downsample first to reduce compute
        if downsample_every and downsample_every > 1 and arr.size > 0:
            arr = arr[::downsample_every]
        # Window crop if requested
        if window_len and window_len > 0 and arr.size > window_len:
            if window_pos == "head":
                start = 0
            elif window_pos == "tail":
                start = arr.size - window_len
            elif window_pos == "center":
                start = max(0, (arr.size - window_len) // 2)
            else:  # random but deterministic per sample index
                rng = np.random.RandomState(int(random_state) + int(j))
                start = int(rng.randint(0, arr.size - window_len + 1))
            arr = arr[start:start + window_len]
        return arr

    # Determine max_len: prefer provided; else derive from TRAIN set after preprocess
    max_len_cfg = model_cfg.get("max_len")
    if max_len_cfg is not None:
        max_len = int(max_len_cfg)
    else:
        # derive from preprocessed training sequences
        def _proc_len(j: int) -> int:
            return int(_apply_preprocess(seq_series.iloc[j], j).size)
        if len(idx_train) == 0:
            max_len = int(lengths.max())
        else:
            max_len = int(max(_proc_len(int(j)) for j in idx_train))
    pad_value = float(model_cfg.get("pad_value", 0.0))
    standardize = bool(model_cfg.get("standardize", True))
    optimizer_name = str(model_cfg.get("optimizer", "adam")).lower()
    weight_decay = float(model_cfg.get("weight_decay", 0.0))
    momentum = float(model_cfg.get("momentum", 0.9)) if optimizer_name == "sgd" else None
    dropout = float(model_cfg.get("dropout", 0.0))
    print(
        f"[mi-race][rnn] Settings: max_len={max_len}, pad_value={pad_value}, standardize={standardize}, "
        f"downsample_every={downsample_every}, window_len={window_len}, window_pos={window_pos}, "
        f"optimizer={optimizer_name}, lr={model_cfg.get('lr', 1e-3)}, weight_decay={weight_decay}, "
        f"momentum={(momentum if momentum is not None else 'n/a')}, dropout={dropout}"
    )

    # Build fixed-length arrays
    def build_matrix(indices: np.ndarray) -> np.ndarray:
        out = np.zeros((len(indices), max_len), dtype=float)
        for i, j in enumerate(indices):
            arr = seq_series.iloc[j]
            arr = _apply_preprocess(arr, int(j))
            if standardize:
                arr = _standardize_sequence(arr)
            out[i] = _pad_or_truncate(arr, max_len, pad_value=pad_value)
        return out

    X_train = build_matrix(idx_train)
    X_test = build_matrix(idx_test)

    # Labels → indices
    classes_sorted = sorted(pd.Series(y).dropna().unique().tolist())
    num_classes = len(classes_sorted)
    label_to_idx = {lbl: i for i, lbl in enumerate(classes_sorted)}
    idx_to_label = {i: lbl for lbl, i in label_to_idx.items()}
    y_train_idx = np.array([label_to_idx[v] for v in y_train], dtype=np.int64)
    y_test_idx = np.array([label_to_idx[v] for v in y_test], dtype=np.int64)

    # DataLoaders
    batch_size = int(model_cfg.get("batch_size", 64))
    train_loader = DataLoader(SeqDataset(X_train, y_train_idx), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(SeqDataset(X_test, y_test_idx), batch_size=batch_size, shuffle=False)

    # Model
    hidden_size = int(model_cfg.get("hidden_size", 128))
    num_layers = int(model_cfg.get("num_layers", 1))
    bidir = bool(model_cfg.get("bidirectional", False))
    lr = float(model_cfg.get("lr", 1e-3))
    epochs = int(model_cfg.get("epochs", 5))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RNNClassifier(input_size=1, hidden_size=hidden_size, num_layers=num_layers,
                          num_classes=num_classes, bidirectional=bidir, dropout=dropout).to(device)
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=(momentum or 0.0), weight_decay=weight_decay, nesterov=bool(model_cfg.get("nesterov", False)))
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    try:
        from tqdm.auto import tqdm
    except Exception:  # pragma: no cover
        tqdm = None

    model.train()
    for ep in range(1, epochs + 1):
        total_loss = 0.0
        total = 0
        correct = 0
        iterator = train_loader
        if tqdm is not None:
            iterator = tqdm(train_loader, desc=f"[rnn] epoch {ep}/{epochs}", leave=False)
        for xb, yb in iterator:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            bs = yb.size(0)
            total_loss += loss.item() * bs
            total += bs
            preds = torch.argmax(logits, dim=1)
            correct += (preds == yb).sum().item()
        avg_loss = total_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        print(f"[mi-race][rnn] epoch {ep}/{epochs}  loss={avg_loss:.4f}  acc={train_acc:.4f}")

    model.eval()
    y_pred_idx = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1)
            y_pred_idx.extend(preds.cpu().numpy().tolist())
    y_pred = np.array([idx_to_label[i] for i in y_pred_idx])
    return y_test, y_pred
