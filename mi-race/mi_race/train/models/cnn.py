from __future__ import annotations

from typing import Tuple
import re
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def run_cnn(feature_df: pd.DataFrame,
            y: np.ndarray,
            train_cfg: dict,
            model_cfg: dict,
            standardize: bool,
            random_state: int,
            stratify,
            quiet: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Train a simple 1D CNN on split sequence features and return (y_test, y_pred)."""
    # Detect groups of columns named like '<prefix>_<idx>' and pick the desired prefix
    pat = re.compile(r"^(?P<prefix>.+)_(?P<idx>\d+)$")
    groups: dict[str, list[tuple[int, str]]] = {}
    for c in feature_df.columns:
        m = pat.match(c)
        if not m:
            continue
        pref = m.group("prefix")
        idx = int(m.group("idx"))
        groups.setdefault(pref, []).append((idx, c))

    if not groups:
        raise SystemExit(
            "[mi-race] CNN requires split sequence columns. Set data.sequence_mode='split' and include at least one sequence column in data.x_cols."
        )

    def _sorted_cols(prefix: str) -> list[str]:
        return [name for _idx, name in sorted(groups[prefix], key=lambda t: t[0])]

    # Decide which group(s) to use and build X_seq with shape
    # (samples, n_channels, n_steps). Single-group runs use n_channels=1.
    seq_prefix = model_cfg.get("sequence_prefix")
    multi_channel = bool(model_cfg.get("multi_channel", False))

    if seq_prefix is not None:
        # Explicit single group (backward compatible).
        if seq_prefix not in groups:
            raise SystemExit(
                f"[mi-race] sequence_prefix='{seq_prefix}' not found. Available split groups: {sorted(groups.keys())}"
            )
        channel_prefixes = [seq_prefix]
    elif multi_channel:
        # Stack every detected group as one input channel (Task B).
        channel_prefixes = sorted(groups.keys())
    elif len(groups) == 1:
        channel_prefixes = [next(iter(groups))]
    else:
        raise SystemExit(
            f"[mi-race] Multiple split sequence groups found {sorted(groups.keys())}. "
            "Set model.cnn.multi_channel=true to use all as channels, or "
            "model.cnn.sequence_prefix to choose one."
        )

    # Build a (samples, n_steps) array per channel, then stack on axis 1.
    per_channel = [feature_df[_sorted_cols(p)].to_numpy(dtype=float) for p in channel_prefixes]
    step_counts = {arr.shape[1] for arr in per_channel}
    if len(step_counts) != 1:
        raise SystemExit(
            f"[mi-race] CNN channels have unequal step counts {sorted(step_counts)} "
            f"for groups {channel_prefixes}; cannot stack."
        )
    X_seq = np.stack(per_channel, axis=1)  # (samples, C, n_steps)
    n_channels = X_seq.shape[1]
    n_steps = X_seq.shape[2]

    if not quiet:
        if n_channels == 1:
            print(f"[cnn] using split sequence group '{channel_prefixes[0]}' ({n_steps} steps)")
        else:
            print(f"[cnn] using {n_channels} channels {channel_prefixes} ({n_steps} steps each)")

    if standardize:
        # Fit one scaler across all channels (flatten C*n_steps), then restore shape.
        scaler = StandardScaler()
        s = X_seq.shape[0]
        X_seq = scaler.fit_transform(X_seq.reshape(s, -1)).reshape(s, n_channels, n_steps)

    test_size = float(train_cfg.get("test_size", 0.2))
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    if not quiet:
        train_counts = pd.Series(y_train).value_counts().sort_index()
        test_counts = pd.Series(y_test).value_counts().sort_index()
        print(f"[cnn] {feature_df.shape[0]} rows  ·  train {dict((str(k), int(v)) for k, v in train_counts.items())}")
        print(f"[cnn] test  {dict((str(k), int(v)) for k, v in test_counts.items())}")

    classes_sorted = sorted(pd.Series(y).dropna().unique().tolist())
    num_classes = len(classes_sorted)
    label_to_idx = {lbl: i for i, lbl in enumerate(classes_sorted)}
    idx_to_label = {i: lbl for lbl, i in label_to_idx.items()}

    y_train_idx = np.array([label_to_idx[v] for v in y_train], dtype=np.int64)
    y_test_idx  = np.array([label_to_idx[v] for v in y_test], dtype=np.int64)

    class SeqDataset(Dataset):
        def __init__(self, X3d: np.ndarray, y1d: np.ndarray):
            # X3d: (samples, n_channels, n_steps)
            self.X = X3d.astype(np.float32)
            self.y = y1d.astype(np.int64)
        def __len__(self):
            return self.X.shape[0]
        def __getitem__(self, idx: int):
            x = torch.from_numpy(self.X[idx])  # (n_channels, n_steps)
            y = torch.tensor(self.y[idx], dtype=torch.long)
            return x, y

    batch_size = int(model_cfg.get("batch_size", 64))
    sampler_mode = str(model_cfg.get("sampler", "none")).lower()
    if sampler_mode == "weighted":
        class_counts = np.bincount(y_train_idx, minlength=num_classes).astype(float)
        class_counts[class_counts == 0.0] = 1.0
        sample_weights = 1.0 / class_counts[y_train_idx]
        sampler = WeightedRandomSampler(weights=torch.tensor(sample_weights, dtype=torch.float32),
                                        num_samples=len(sample_weights),
                                        replacement=True)
        if not quiet:
            print("[cnn] using WeightedRandomSampler for class balancing")
        train_loader = DataLoader(SeqDataset(X_train, y_train_idx), batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(SeqDataset(X_train, y_train_idx), batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(SeqDataset(X_test,  y_test_idx),  batch_size=batch_size, shuffle=False)

    in_channels = n_channels
    channels = list(model_cfg.get("channels", [16, 32]))
    kernel_size = int(model_cfg.get("kernel_size", 5))
    pool_size = int(model_cfg.get("pool", 2))
    fc_hidden = int(model_cfg.get("fc", 128))

    class CNN1D(nn.Module):
        def __init__(self, L: int):
            super().__init__()
            pad = kernel_size // 2
            self.conv1 = nn.Conv1d(in_channels, channels[0], kernel_size, padding=pad)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool1d(pool_size)
            self.conv2 = nn.Conv1d(channels[0], channels[1], kernel_size, padding=pad)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool1d(pool_size)
            with torch.no_grad():
                dummy = torch.zeros(1, in_channels, L)
                h = self._forward_features(dummy)
                flat = h.view(1, -1).shape[1]
            self.fc1 = nn.Linear(flat, fc_hidden)
            self.relu_fc = nn.ReLU()
            self.out = nn.Linear(fc_hidden, num_classes)

        def _forward_features(self, x):
            x = self.pool1(self.relu1(self.conv1(x)))
            x = self.pool2(self.relu2(self.conv2(x)))
            return x

        def forward(self, x):
            x = self._forward_features(x)
            x = torch.flatten(x, 1)
            x = self.relu_fc(self.fc1(x))
            return self.out(x)

    L = n_steps
    device_cfg = str(model_cfg.get("device", "auto")).lower()
    if device_cfg == "cpu":
        device = torch.device("cpu")
    elif device_cfg in ("cuda", "gpu"):
        if not torch.cuda.is_available():
            raise SystemExit("[mi-race][cnn] device='cuda' requested but CUDA is not available.")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not quiet:
        print(f"[cnn] device: {device}")
    model = CNN1D(L).to(device)
    epochs = int(model_cfg.get("epochs", 5))
    # Hardcoded cadence: show tqdm/progress + epoch logs every 5 epochs
    log_every_epochs = 5
    lr = float(model_cfg.get("lr", 1e-3))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    criterion: nn.Module
    cw_mode = str(model_cfg.get("class_weight", "none")).lower()
    if cw_mode == "balanced":
        counts = np.bincount(y_train_idx, minlength=num_classes).astype(float)
        counts[counts == 0.0] = 1.0
        inv_freq = 1.0 / counts
        weights = inv_freq * (num_classes / inv_freq.sum())
        weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
        if not quiet:
            print(f"[cnn] class weights (balanced): {weights.tolist()}")
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    else:
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
        # Only show tqdm progress on selected epochs to reduce noise
        from mi_race.cli.ui import progress_disabled
        show_bar = (not quiet) and (tqdm is not None) and (ep % log_every_epochs == 0) and (not progress_disabled())
        if show_bar:
            iterator = tqdm(train_loader, desc=f"[cnn] epoch {ep}/{epochs}", leave=False)
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
        # Print epoch summary strictly every N epochs
        if ep % log_every_epochs == 0:
            # Evaluate on test set (loss & accuracy)
            model.eval()
            with torch.no_grad():
                t_total = 0
                t_correct = 0
                t_loss_sum = 0.0
                for xbt, ybt in test_loader:
                    xbt = xbt.to(device)
                    ybt = ybt.to(device)
                    logits_t = model(xbt)
                    loss_t = criterion(logits_t, ybt)
                    bs_t = ybt.size(0)
                    t_loss_sum += loss_t.item() * bs_t
                    t_total += bs_t
                    preds_t = torch.argmax(logits_t, dim=1)
                    t_correct += (preds_t == ybt).sum().item()
                test_loss = t_loss_sum / max(t_total, 1)
                test_acc = t_correct / max(t_total, 1)
            model.train()
            if not quiet:
                print(
                    f"[cnn] epoch {ep}/{epochs}  "
                    f"train_loss={avg_loss:.4f} train_acc={train_acc:.4f}  "
                    f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
                )

    model.eval()
    y_pred_idx = []
    final_total = 0
    final_correct = 0
    final_loss_sum = 0.0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            bs = yb.size(0)
            final_loss_sum += loss.item() * bs
            final_total += bs
            preds = torch.argmax(logits, dim=1)
            final_correct += (preds == yb).sum().item()
            y_pred_idx.extend(preds.cpu().numpy().tolist())
    if (not quiet) and final_total > 0:
        final_test_loss = final_loss_sum / final_total
        final_test_acc = final_correct / final_total
        print(f"[cnn] final test: loss={final_test_loss:.4f} acc={final_test_acc:.4f}")
    y_pred = np.array([idx_to_label[i] for i in y_pred_idx])
    return y_test, y_pred
