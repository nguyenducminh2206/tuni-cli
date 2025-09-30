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
from sklearn.metrics import confusion_matrix, accuracy_score

from ..data.extract_data import build_df


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
        return " === Preview (first rows) ===\n\n" + df.head().to_string(index=False, formatters=fmt) + "\n"


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
        raise SystemExit(f"[mi-race] Config not found: {cfg_path.resolve()}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Override model type from CLI
    if args.model:
        cfg.setdefault("model", {})["type"] = args.model

    # Data section
    if "data" not in cfg:
        raise SystemExit("[mi-race] Missing 'data' section in config.")
    data_cfg = cfg["data"]

    def _load_df(dc: dict) -> pd.DataFrame:
        path = dc.get("path")
        ds_id = dc.get("id")
        if path:
            p = Path(path)
            if p.exists():
                suf = p.suffix.lower()
                if suf == ".csv":
                    return pd.read_csv(p)
                if suf == ".tsv":
                    return pd.read_csv(p, sep="\t")
                if suf == ".parquet":
                    return pd.read_parquet(p)
                raise SystemExit(f"[mi-race] Unsupported file type: {suf}")
            # treat as dataset id
            return build_df(path)
        if ds_id:
            return build_df(ds_id)
        raise SystemExit("[mi-race] Provide data.path or data.id in config.")

    df = _load_df(data_cfg)

    # Target column
    y_col = data_cfg.get("y_col")
    if not y_col:
        raise SystemExit("[mi-race] data.y_col must be specified.")
    if y_col not in df.columns:
        raise SystemExit(f"[mi-race] y_col '{y_col}' not in dataset. Available: {df.columns.tolist()}")

    # Feature columns resolution (scalable for 1 or many)
    x_cols_raw = data_cfg.get("x_cols")
    single = data_cfg.get("x_col")  # alternate key
    if x_cols_raw and single:
        raise SystemExit("[mi-race] Use only one of data.x_cols or data.x_col, not both.")

    if x_cols_raw is None and single is not None:
        x_cols = [single]
    elif isinstance(x_cols_raw, str):
        x_cols = [x_cols_raw]
    elif isinstance(x_cols_raw, (list, tuple)):
        x_cols = list(x_cols_raw)
    elif x_cols_raw is None:
        # Infer numeric (excluding target)
        x_cols = [
            c for c in df.columns
            if c != y_col and pd.api.types.is_numeric_dtype(df[c])
        ]
    else:
        raise SystemExit(f"[mi-race] data.x_cols must be string/list/tuple. Got {type(x_cols_raw).__name__}")

    # Deduplicate while preserving order
    seen = set()
    x_cols = [c for c in x_cols if not (c in seen or seen.add(c))]

    if not x_cols:
        raise SystemExit("[mi-race] No feature columns resolved (x_cols).")

    # Validate existence
    missing = [c for c in x_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"[mi-race] Missing feature columns: {missing}. Available: {df.columns.tolist()}")

    if y_col in x_cols:
        raise SystemExit(f"[mi-race] y_col '{y_col}' cannot also be in x_cols.")

    print(f"[mi-race] Using feature columns (n={len(x_cols)}): {x_cols}")

    # Arrays (ensure 2D even if single column)
    X = df[x_cols].to_numpy()
    y = df[y_col].to_numpy()

    # Train/test split
    train_cfg = cfg.get("train", {})
    test_size = float(train_cfg.get("test_size", 0.2))
    random_state = int(train_cfg.get("random_state", 42))
    stratify = y if train_cfg.get("stratify", True) else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

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

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    n_classes = sorted(pd.Series(y).dropna().unique().tolist())
    cm = confusion_matrix(y_test, y_pred, labels=n_classes)

    out_cfg = cfg.get("output", {})
    out_dir = Path(out_cfg.get("dir", "outputs"))
    _ensure_outdir(out_dir)
    cm_path = out_dir / "confusion_matrix_global.csv"
    pd.DataFrame(cm).to_csv(cm_path, header=False, index=False)

    preview = _preview_block(df)
    print(preview)
    print(f"Total rows: {len(df):,}  •  Classes: {n_classes}\n")
    print("===  Test Accuracy ===")
    print(f"{acc*100:.2f}%\n")
    print("=== Confusion Matrix (test) ===")
    print(_cm_table(cm, n_classes))
    print(f"\nSaved: {cm_path}")
