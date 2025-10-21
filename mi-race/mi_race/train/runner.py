"""Runner function for CLI tool"""

from pathlib import Path
import json
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from ..data.extract_data import build_df
from ..analysis import info_from_confusion_matrix

# Optional CNN support (PyTorch)
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import re



def _load_df_from_cfg(data_cfg: dict) -> pd.DataFrame:
    """
    Rules:
      - If data.path given:
          * If path exists and is a file: load by extension
          * If path exists and is a directory OR has no extension: treat as dataset id -> build_df(path name)
          * If path does not exist: treat raw string as dataset id -> build_df(path)
      - Else if data.id given: build_df(id)
      - Else: error
    """
    path = data_cfg.get("path")
    ds_id = data_cfg.get("id")
    if path:
        p = Path(path)
        if p.exists():
            if p.is_file():
                suf = p.suffix.lower()
                if suf == ".csv":
                    return pd.read_csv(p)
                if suf == ".tsv":
                    return pd.read_csv(p, sep="\t")
                if suf == ".parquet":
                    return pd.read_parquet(p)
                raise SystemExit(f"[mi-race] Unsupported file extension: {suf}")
            # Directory OR something like 'data_7x7' (no extension) -> dataset id
            return build_df(p.name)
        # Not existing path: treat string as dataset id
        return build_df(path)
    if ds_id:
        return build_df(ds_id)
    raise SystemExit("[mi-race] Provide data.path or data.id in config.")


def _summarize_sequence_col(series: pd.Series, prefix: str) -> pd.DataFrame:
    """
    Convert a column of sequences (list/array) into statistical features.
    Returns a DataFrame with derived columns.
    """
    def safe_array(x):
        if isinstance(x, (list, tuple, np.ndarray)):
            arr = np.asarray(x, dtype=float)
            return arr
        return np.array([], dtype=float)

    arrays = series.apply(safe_array)
    lengths = arrays.apply(len).to_numpy()
    # Avoid warnings on empty
    means = np.array([a.mean() if a.size else np.nan for a in arrays])
    stds  = np.array([a.std(ddof=0) if a.size else np.nan for a in arrays])
    mins  = np.array([a.min() if a.size else np.nan for a in arrays])
    maxs  = np.array([a.max() if a.size else np.nan for a in arrays])
    
    # Add after _summarize_sequence_col stats (extend dict)
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    qs = {f"{prefix}_q{int(q*100)}": np.array([np.quantile(a, q) if a.size else np.nan for a in arrays])
          for q in quantiles}
    # merge into returned DataFrame
    return pd.DataFrame({ **{
        f"{prefix}_len": lengths,
        f"{prefix}_mean": means,
        f"{prefix}_std": stds,
        f"{prefix}_min": mins,
        f"{prefix}_max": maxs,
    }, **qs })


def _split_sequence_col(series: pd.Series, prefix: str) -> pd.DataFrame:
    """
    Split a column of sequences (list/array) into multiple columns.
    Each time point becomes a separate column with the pattern: prefix_0, prefix_1, etc.
    Returns a DataFrame with individual time point columns.
    """
    def safe_array(x):
        if isinstance(x, (list, tuple, np.ndarray)):
            arr = np.asarray(x, dtype=float)
            return arr
        return np.array([], dtype=float)

    arrays = series.apply(safe_array)
    
    # Find the maximum length across all sequences to determine number of columns
    max_length = arrays.apply(len).max()
    
    if max_length == 0:
        # All sequences are empty, return empty DataFrame
        return pd.DataFrame()
    
    # Create columns for each time point
    split_data = {}
    for i in range(max_length):
        col_name = f"{prefix}_{i}"
        # Extract the i-th element from each sequence, fill with NaN if sequence is too short
        split_data[col_name] = arrays.apply(lambda arr: arr[i] if i < len(arr) else np.nan)
    
    return pd.DataFrame(split_data)


def _extract_sequence_range(series: pd.Series, prefix: str, start_idx: int, end_idx: int) -> pd.DataFrame:
    """
    Extract a specific range from a sequence column.
    Returns a DataFrame with columns like prefix_start, prefix_start+1, ..., prefix_end.
    """
    def safe_array(x):
        if isinstance(x, (list, tuple, np.ndarray)):
            arr = np.asarray(x, dtype=float)
            return arr
        return np.array([], dtype=float)

    arrays = series.apply(safe_array)
    
    # Create columns for the specified range
    split_data = {}
    for i in range(start_idx, end_idx + 1):
        col_name = f"{prefix}_{i}"
        # Extract the i-th element from each sequence, fill with NaN if sequence is too short
        split_data[col_name] = arrays.apply(lambda arr: arr[i] if i < len(arr) else np.nan)
    
    return pd.DataFrame(split_data)


def _summarize_cell(x, n_head=3, n_tail=3):
    # treat NaNs/None nicely
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""

    # summarize arrays/lists/tuples
    if isinstance(x, (np.ndarray, list, tuple)) and isinstance(x, Sequence):
        arr = np.asarray(x)
        if arr.size <= n_head + n_tail + 1:
            body = ", ".join(map(str, arr.tolist()))
        else:
            head = ", ".join(map(str, arr[:n_head].tolist()))
            tail = ", ".join(map(str, arr[-n_tail:].tolist()))
            body = f"{head}, …, {tail}"
        return f"[{body}] (len={arr.size})"

    # IMPORTANT: return a string for everything else
    return str(x)


def _preview_block(df: pd.DataFrame, colwidth: int = 80) -> str:
    fmt = {c: (lambda v, f=_summarize_cell: f(v)) for c in df.columns}
    with pd.option_context(
        "display.large_repr", "truncate",
        "display.max_colwidth", colwidth,
    ):
        return " \n=== Preview (first rows) ===\n" + df.head().to_string(index=False, formatters=fmt) + "\n"


def _ensure_outdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _cm_table(cm: np.ndarray, labels) -> str:
    """Create a clean confusion matrix table without internal vertical bars."""
    # Create header row
    header_labels = [str(label) for label in labels]
    header = "     " + "".join(f"{label:>6}" for label in header_labels)
    
    # Create separator line
    separator = "    " + "-" * (6 * len(labels))
    
    # Create data rows
    rows = []
    for i, label in enumerate(labels):
        row_values = "".join(f"{int(cm[i][j]):>6}" for j in range(len(labels)))
        rows.append(f"{str(label):>3} |{row_values}")
    
    # Combine all parts
    return "\n".join([header, separator] + rows)  


def run_cmd(args):
    # Load config
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise SystemExit(f"[mi-race] Config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Model override
    if args.model:
        cfg.setdefault("model", {})["type"] = args.model

    # Data section
    if "data" not in cfg:
        raise SystemExit("[mi-race] Missing 'data' section in config.")
    data_cfg = cfg["data"]

    df = _load_df_from_cfg(data_cfg)
    print(f"[mi-race] Loaded dataset with columns: {df.columns.tolist()}")

    # Target
    y_col = data_cfg.get("y_col")
    if not y_col:
        raise SystemExit("[mi-race] data.y_col must be specified.")
    if y_col not in df.columns:
        raise SystemExit(f"[mi-race] y_col '{y_col}' not found. Available: {df.columns.tolist()}")

    # Feature columns
    x_cols_raw = data_cfg.get("x_cols")
    single = data_cfg.get("x_col")
    if x_cols_raw and single:
        raise SystemExit("[mi-race] Use only one of data.x_cols or data.x_col.")
    if x_cols_raw is None and single is not None:
        x_cols = [single]
    elif isinstance(x_cols_raw, str):
        x_cols = [x_cols_raw]
    elif isinstance(x_cols_raw, (list, tuple)):
        x_cols = list(x_cols_raw)
    elif x_cols_raw is None:
        # Infer numeric
        x_cols = [c for c in df.columns if c != y_col and pd.api.types.is_numeric_dtype(df[c])]
    else:
        raise SystemExit(f"[mi-race] Invalid data.x_cols type: {type(x_cols_raw).__name__}")

    # Deduplicate preserving order
    seen = set()
    x_cols = [c for c in x_cols if not (c in seen or seen.add(c))]
    if not x_cols:
        raise SystemExit("[mi-race] No feature columns resolved.")

    # Handle sequence columns and time trace ranges
    seq_mode = data_cfg.get("sequence_mode", "stats")  # 'stats' | 'ignore' | 'split'
    new_feature_frames = []
    keep_cols = []
    
    # Check if any x_cols are sequence ranges in the form "<seqcol>_<start>:<seqcol>_<end>"
    range_specs = []
    regular_cols = []
    
    for c in x_cols:
        if ":" in c:
            # This is a sequence range specification
            range_specs.append(c)
        else:
            regular_cols.append(c)
    
    # Validate existence of regular columns only (time trace ranges are handled separately)
    missing = [c for c in regular_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"[mi-race] Missing feature columns: {missing}. Available: {df.columns.tolist()}")
    if y_col in regular_cols:
        raise SystemExit(f"[mi-race] y_col '{y_col}' cannot be among x_cols.")
    
    # Process sequence ranges first
    for range_spec in range_specs:
        try:
            # Parse range like "seqcol_1:seqcol_50" -> left and right must share same base prefix before final underscore
            parts = range_spec.split(":")
            if len(parts) != 2:
                raise ValueError("Invalid range format")

            def parse_part(token: str):
                if "_" not in token:
                    raise ValueError("Range token must be of the form '<col>_<index>'")
                base, idx_str = token.rsplit("_", 1)
                return base, int(idx_str)

            base_l, start_idx = parse_part(parts[0])
            base_r, end_idx   = parse_part(parts[1])
            if base_l != base_r:
                raise ValueError("Range endpoints must refer to the same sequence column")
            col_name = base_l
            if col_name not in df.columns:
                raise ValueError(f"sequence column '{col_name}' not found in dataset")
            if start_idx > end_idx:
                raise ValueError("Start index must be <= end index")

            series = df[col_name]
            range_df = _extract_sequence_range(series, col_name, start_idx, end_idx)
            if not range_df.empty:
                new_feature_frames.append(range_df)
                print(f"[mi-race] Extracted sequence range {range_spec}: {len(range_df.columns)} columns ({range_df.columns[0]} to {range_df.columns[-1]})")
            else:
                print(f"[mi-race] Sequence range {range_spec} was empty, skipping")

        except (ValueError, IndexError) as e:
            raise SystemExit(f"[mi-race] Invalid sequence range specification '{range_spec}': {e}")
    
    # Process regular columns (including any declared sequence columns if present)
    for c in regular_cols:
        if c not in df.columns:
            continue  # Will be caught later in validation
            
        s = df[c]
        if s.dtype == object and s.head(5).apply(lambda v: isinstance(v, (list, tuple, np.ndarray))).all():
            if seq_mode == "stats":
                stats_df = _summarize_sequence_col(s, c)
                new_feature_frames.append(stats_df)
                print(f"[mi-race] Sequence column '{c}' converted to stats: {stats_df.columns.tolist()}")
            elif seq_mode == "split":
                split_df = _split_sequence_col(s, c)
                if not split_df.empty:
                    new_feature_frames.append(split_df)
                    print(f"[mi-race] Sequence column '{c}' split into {len(split_df.columns)} columns: {split_df.columns.tolist()}")
                else:
                    print(f"[mi-race] Sequence column '{c}' was empty, skipping split")
            elif seq_mode == "ignore":
                print(f"[mi-race] Ignoring sequence column '{c}'")
                continue
            else:
                raise SystemExit(f"[mi-race] Unknown sequence_mode '{seq_mode}'. Supported: 'stats', 'split', 'ignore'")
        else:
            keep_cols.append(c)

    feature_df = df[keep_cols].copy()
    if new_feature_frames:
        for fdf in new_feature_frames:
            feature_df = pd.concat([feature_df, fdf], axis=1)

    # Additionally, split any sequence columns explicitly listed in config under data.sequence_cols
    seq_declared = data_cfg.get("sequence_cols", [])
    if isinstance(seq_declared, (list, tuple)):
        for sc in seq_declared:
            if sc in df.columns:
                s = df[sc]
                if s.dtype == object and s.head(5).apply(lambda v: isinstance(v, (list, tuple, np.ndarray))).all():
                    split_df = _split_sequence_col(s, sc)
                    if not split_df.empty:
                        feature_df = pd.concat([feature_df, split_df], axis=1)
                        print(f"[mi-race] (declared) sequence column '{sc}' split into {len(split_df.columns)} columns")
                else:
                    print(f"[mi-race][WARN] Declared sequence column '{sc}' is not a sequence-like object column; skipping split")
    
    pd.DataFrame(feature_df)

    resolved_feature_cols = feature_df.columns.tolist()
    print(f"[mi-race] Final feature columns (n={len(resolved_feature_cols)}): {resolved_feature_cols}")

    # Save feature_df to CSV file
    out_cfg = cfg.get("output", {})
    out_dir = Path(out_cfg.get("dir", "outputs"))
    _ensure_outdir(out_dir)
    feature_df_path = out_dir / "processed_features.csv"
    feature_df.to_csv(feature_df_path, index=False)
    print(f"[mi-race] Saved processed features to: {feature_df_path}")

    # Build arrays for general stats and counts
    y = df[y_col].to_numpy()
    full_counts = pd.Series(y).value_counts().sort_index()
    print(f"[mi-race] Class distribution (full): {full_counts.to_dict()}")

    # Train settings
    train_cfg = cfg.get("train", {})
    test_size = float(train_cfg.get("test_size", 0.2))
    random_state = int(train_cfg.get("random_state", 42))
    standardize = bool(train_cfg.get("standardize", True))
    stratify = y if train_cfg.get("stratify", True) else None

    # Resolve model selection and its specific config (support nested model configs)
    model_section = cfg.get("model", {})
    mtype = model_section.get("type", "mlp").lower()
    selected_cfg = model_section.get(mtype, model_section)

    if mtype == "mlp":
        # Classic sklearn MLP on tabular features
        X = feature_df.to_numpy()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )

        train_counts = pd.Series(y_train).value_counts().sort_index()
        test_counts  = pd.Series(y_test).value_counts().sort_index()
        print(f"[mi-race] Class distribution (train): {train_counts.to_dict()}")
        print(f"[mi-race] Class distribution (test):  {test_counts.to_dict()}")

        hidden_layers = tuple(selected_cfg.get("hidden_layers", [128, 128]))
        activation = selected_cfg.get("activation", "relu")
        alpha = float(selected_cfg.get("alpha", 1e-4))
        max_iter = int(train_cfg.get("max_iter", 200))

        steps = []
        if standardize:
            steps.append(("scaler", StandardScaler()))
        steps.append((
            "clf",
            MLPClassifier(
                hidden_layer_sizes=hidden_layers,
                activation=activation,
                alpha=alpha,
                max_iter=max_iter,
                random_state=random_state
            )
        ))
        pipe = Pipeline(steps)

        pipe.fit(X_train, y_train)

        learned_classes = pipe.named_steps["clf"].classes_.tolist()
        print(f"[mi-race] Model learned classes: {learned_classes}")
        missing_classes = [c for c in full_counts.index if c not in learned_classes]
        if missing_classes:
            print(f"[mi-race][WARN] Classes missing from training set: {missing_classes} "
                  f"-> cannot be predicted. Consider reducing test_size or using stratified CV.")

        y_pred = pipe.predict(X_test)

    elif mtype == "cnn":
        # 1D CNN on split sequence columns generated by _split_sequence_col
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
                "[mi-race] CNN requires split sequence columns. Set data.sequence_mode='split' "
                "and include at least one sequence column (e.g., 'time_trace') in data.x_cols."
            )

        # Choose which split sequence to use
        seq_prefix = selected_cfg.get("sequence_prefix")
        chosen_prefix = None
        if seq_prefix is not None:
            if seq_prefix not in groups:
                raise SystemExit(
                    f"[mi-race] sequence_prefix='{seq_prefix}' not found. Available split groups: {sorted(groups.keys())}"
                )
            chosen_prefix = seq_prefix
        else:
            if len(groups) == 1:
                chosen_prefix = next(iter(groups))
            else:
                raise SystemExit(
                    f"[mi-race] Multiple split sequence groups found {sorted(groups.keys())}. "
                    f"Specify model.cnn.sequence_prefix to choose one."
                )

        seq_cols_sorted = [name for idx, name in sorted(groups[chosen_prefix], key=lambda t: t[0])]
        print(f"[mi-race][cnn] Using split sequence group '{chosen_prefix}' with {len(seq_cols_sorted)} steps")
        X_seq = feature_df[seq_cols_sorted].to_numpy(dtype=float)

        if standardize:
            scaler = StandardScaler()
            X_seq = scaler.fit_transform(X_seq)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_seq, y, test_size=test_size, random_state=random_state, stratify=stratify
        )

        train_counts = pd.Series(y_train).value_counts().sort_index()
        test_counts  = pd.Series(y_test).value_counts().sort_index()
        print(f"[mi-race] Class distribution (train): {train_counts.to_dict()}")
        print(f"[mi-race] Class distribution (test):  {test_counts.to_dict()}")

        # Label mapping to integers
        classes_sorted = sorted(pd.Series(y).dropna().unique().tolist())
        num_classes = len(classes_sorted)
        label_to_idx = {lbl: i for i, lbl in enumerate(classes_sorted)}
        idx_to_label = {i: lbl for lbl, i in label_to_idx.items()}

        y_train_idx = np.array([label_to_idx[v] for v in y_train], dtype=np.int64)
        y_test_idx  = np.array([label_to_idx[v] for v in y_test], dtype=np.int64)

        # Torch dataset
        class SeqDataset(Dataset):
            def __init__(self, X2d: np.ndarray, y1d: np.ndarray):
                self.X = X2d.astype(np.float32)
                self.y = y1d.astype(np.int64)
            def __len__(self):
                return self.X.shape[0]
            def __getitem__(self, idx: int):
                # shape (1, L)
                x = torch.from_numpy(self.X[idx][None, :])
                y = torch.tensor(self.y[idx], dtype=torch.long)
                return x, y

        batch_size = int(selected_cfg.get("batch_size", 64))
        # Optional weighted sampler to handle imbalance
        sampler_mode = str(selected_cfg.get("sampler", "none")).lower()
        if sampler_mode == "weighted":
            class_counts = np.bincount(y_train_idx, minlength=num_classes).astype(float)
            class_counts[class_counts == 0.0] = 1.0
            sample_weights = 1.0 / class_counts[y_train_idx]
            sampler = WeightedRandomSampler(weights=torch.tensor(sample_weights, dtype=torch.float32),
                                            num_samples=len(sample_weights),
                                            replacement=True)
            print("[mi-race][cnn] Using WeightedRandomSampler for class balancing")
            train_loader = DataLoader(SeqDataset(X_train, y_train_idx), batch_size=batch_size, sampler=sampler)
        else:
            train_loader = DataLoader(SeqDataset(X_train, y_train_idx), batch_size=batch_size, shuffle=True)
        test_loader  = DataLoader(SeqDataset(X_test,  y_test_idx),  batch_size=batch_size, shuffle=False)

        # Model definition
        in_channels = 1
        channels = list(selected_cfg.get("channels", [16, 32]))
        kernel_size = int(selected_cfg.get("kernel_size", 5))
        pool_size = int(selected_cfg.get("pool", 2))
        fc_hidden = int(selected_cfg.get("fc", 128))

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
                # Determine flattened size dynamically
                with torch.no_grad():
                    dummy = torch.zeros(1, 1, L)
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

        L = X_seq.shape[1]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CNN1D(L).to(device)
        epochs = int(selected_cfg.get("epochs", 5))
        lr = float(selected_cfg.get("lr", 1e-3))
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # Optional class weights for imbalance handling
        criterion: nn.Module
        cw_mode = str(selected_cfg.get("class_weight", "none")).lower()
        if cw_mode == "balanced":
            counts = np.bincount(y_train_idx, minlength=num_classes).astype(float)
            # Avoid division by zero (shouldn't happen with stratify=True)
            counts[counts == 0.0] = 1.0
            inv_freq = 1.0 / counts
            weights = inv_freq * (num_classes / inv_freq.sum())  # normalize-ish
            weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
            print(f"[mi-race][cnn] Using class weights (balanced): {weights.tolist()}")
            criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            criterion = nn.CrossEntropyLoss()

        # Train with tqdm progress
        try:
            from tqdm.auto import tqdm  # lazy import to keep MLP path lightweight
        except Exception:  # pragma: no cover
            tqdm = None

        model.train()
        for ep in range(1, epochs + 1):
            total_loss = 0.0
            total = 0
            correct = 0
            iterator = train_loader
            if tqdm is not None:
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
            print(f"[mi-race][cnn] epoch {ep}/{epochs}  loss={avg_loss:.4f}  acc={train_acc:.4f}")

        # Predict on test
        model.eval()
        y_pred_idx = []
        with torch.no_grad():
            for xb, _ in test_loader:
                xb = xb.to(device)
                logits = model(xb)
                preds = torch.argmax(logits, dim=1)
                y_pred_idx.extend(preds.cpu().numpy().tolist())
        y_pred = np.array([idx_to_label[i] for i in y_pred_idx])

    else:
        raise SystemExit("[mi-race] Unknown model type. Supported: 'mlp', 'cnn'.")
    acc = accuracy_score(y_test, y_pred)

    n_classes = sorted(pd.Series(y).dropna().unique().tolist())
    cm = confusion_matrix(y_test, y_pred, labels=n_classes)

    out_cfg = cfg.get("output", {})
    out_dir = Path(out_cfg.get("dir", "outputs"))
    _ensure_outdir(out_dir)
    cm_path = out_dir / "confusion_matrix_global.csv"
    pd.DataFrame(cm).to_csv(cm_path, header=False, index=False)

    # Compute information-theoretic metrics from confusion matrix
    info = info_from_confusion_matrix(cm, labels=n_classes)
    mi_path = out_dir / "confusion_matrix_info.json"
    with mi_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    from sklearn.metrics import f1_score
    # Identify labels with zero predicted samples for clearer messaging
    preds_unique = sorted(pd.Series(y_pred).dropna().unique().tolist())
    missing_predicted = [c for c in n_classes if c not in preds_unique]
    if missing_predicted:
        print(f"[mi-race][WARN] No predicted samples for classes: {missing_predicted}. "
              f"Metrics for these labels will use zero_division=0.")

    # Use zero_division=0 to avoid sklearn UndefinedMetricWarning
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    preview = _preview_block(df)
    print(preview)
    print(f"Total rows: {len(df):,}  •  Classes: {n_classes}\n")

    print("\n=== Metrics (test) ===")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Macro F1: {macro_f1*100:.2f}%")

    print("\n=== Confusion Matrix (test) ===")
    print(_cm_table(cm, n_classes))

    # Print MI/entropy summary
    print("\n=== Information (from confusion matrix) ===")
    print(f"H(true): {info['H_true']:.4f} bits  |  H(pred): {info['H_pred']:.4f} bits  |  Hjoint: {info['H_joint']:.4f} bits")
    print(f"I(true;pred): {info['I']:.4f} bits  |  NMI_sqrt: {info['NMI_sqrt']:.4f}  |  NMI_min: {info['NMI_min']:.4f}  |  NMI_max: {info['NMI_max']:.4f}")
    print(f"H(true|pred): {info['H_true_given_pred']:.4f} bits  |  H(pred|true): {info['H_pred_given_true']:.4f} bits")

    # Classification report
    if cfg.get("output", {}).get("show_report", True):
        print("\n=== Classification Report (test) ===")
        print(classification_report(y_test, y_pred, labels=n_classes, digits=4, zero_division=0))

    print(f"\nSaved: {cm_path}")
    print(f"Saved: {mi_path}")
