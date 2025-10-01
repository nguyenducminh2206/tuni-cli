from pathlib import Path
import json
from collections.abc import Sequence

import numpy as np
import pandas as pd
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from ..data.extract_data import build_df


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
    headers = [""] + [str(_) for _ in labels]
    rows = [[str(labels[i])] + [int(v) for v in cm[i]] for i in range(len(labels))]
    return tabulate(rows, headers=headers, tablefmt="github")  


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

    # Validate existence
    missing = [c for c in x_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"[mi-race] Missing feature columns: {missing}. Available: {df.columns.tolist()}")
    if y_col in x_cols:
        raise SystemExit(f"[mi-race] y_col '{y_col}' cannot be among x_cols.")

    # Handle sequence columns (like 'time_trace')
    # If a column has Python objects that are list/array, convert to stats
    seq_mode = data_cfg.get("sequence_mode", "stats")  # 'stats' | 'ignore'
    new_feature_frames = []
    keep_cols = []
    for c in x_cols:
        s = df[c]
        if s.dtype == object and s.head(5).apply(lambda v: isinstance(v, (list, tuple, np.ndarray))).all():
            if seq_mode == "stats":
                stats_df = _summarize_sequence_col(s, c)
                new_feature_frames.append(stats_df)
                print(f"[mi-race] Sequence column '{c}' converted to stats: {stats_df.columns.tolist()}")
            elif seq_mode == "ignore":
                print(f"[mi-race] Ignoring sequence column '{c}'")
                continue
            else:
                raise SystemExit(f"[mi-race] Unknown sequence_mode '{seq_mode}'")
        else:
            keep_cols.append(c)

    feature_df = df[keep_cols].copy()
    if new_feature_frames:
        for fdf in new_feature_frames:
            feature_df = pd.concat([feature_df, fdf], axis=1)

    resolved_feature_cols = feature_df.columns.tolist()
    print(f"[mi-race] Final feature columns (n={len(resolved_feature_cols)}): {resolved_feature_cols}")

    # Build arrays
    X = feature_df.to_numpy()
    y = df[y_col].to_numpy()

    full_counts = pd.Series(y).value_counts().sort_index()
    print(f"[mi-race] Class distribution (full): {full_counts.to_dict()}")

    # Train/test split
    train_cfg = cfg.get("train", {})
    test_size = float(train_cfg.get("test_size", 0.2))
    random_state = int(train_cfg.get("random_state", 42))
    stratify = y if train_cfg.get("stratify", True) else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    train_counts = pd.Series(y_train).value_counts().sort_index()
    test_counts  = pd.Series(y_test).value_counts().sort_index()
    print(f"[mi-race] Class distribution (train): {train_counts.to_dict()}")
    print(f"[mi-race] Class distribution (test):  {test_counts.to_dict()}")

    # Model
    model_cfg = cfg.get("model", {})
    mtype = model_cfg.get("type", "mlp").lower()
    if mtype != "mlp":
        raise SystemExit("[mi-race] Only 'mlp' currently supported.")

    hidden_layers = tuple(model_cfg.get("hidden_layers", [128, 128]))
    activation = model_cfg.get("activation", "relu")
    alpha = float(model_cfg.get("alpha", 1e-4))
    max_iter = int(train_cfg.get("max_iter", 200))
    standardize = bool(train_cfg.get("standardize", True))

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
    acc = accuracy_score(y_test, y_pred)

    n_classes = sorted(pd.Series(y).dropna().unique().tolist())
    cm = confusion_matrix(y_test, y_pred, labels=n_classes)

    out_cfg = cfg.get("output", {})
    out_dir = Path(out_cfg.get("dir", "outputs"))
    _ensure_outdir(out_dir)
    cm_path = out_dir / "confusion_matrix_global.csv"
    pd.DataFrame(cm).to_csv(cm_path, header=False, index=False)

    from sklearn.metrics import f1_score
    macro_f1 = f1_score(y_test, y_pred, average='macro')

    preview = _preview_block(df)
    print(preview)
    print(f"Total rows: {len(df):,}  •  Classes: {n_classes}\n")

    print("\n=== Metrics (test) ===")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Macro F1: {macro_f1*100:.2f}%")

    print("\n=== Confusion Matrix (test) ===")
    print(_cm_table(cm, n_classes))

    # Optional classification report
    if cfg.get("output", {}).get("show_report", True):
        print("\n=== Classification Report (test) ===")
        print(classification_report(y_test, y_pred, digits=4))

    print(f"\nSaved: {cm_path}")
