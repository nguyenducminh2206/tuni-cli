from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence
from typing import Tuple

import numpy as np
import pandas as pd

from ..data.extract_data import build_df


def ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_df_from_cfg(data_cfg: dict) -> pd.DataFrame:
    """
    Load dataframe according to config.

    Rules:
      - If data.path given:
          * If path exists and is a file: load by extension
          * If path exists and is a directory OR has no extension: treat as dataset id -> build_df(path name)
          * If path does not exist: treat raw string as dataset id -> build_df(path)
      - Else if data.id given: build_df(id)
      - Else: error

    Additionally, when data.balance is enabled, we load more samples per HDF5 file (10
    instead of 1) to avoid ending up with too small a dataset after balancing.
    """
    path = data_cfg.get("path")
    ds_id = data_cfg.get("id")

    balance_cfg = data_cfg.get("balance", False)
    if isinstance(balance_cfg, bool):
        balance_enabled = balance_cfg
    elif isinstance(balance_cfg, dict):
        balance_enabled = balance_cfg.get("enabled", True)
    else:
        balance_enabled = False
    n_samples_per_file = 10 if balance_enabled else 1

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
            return build_df(p.name, n_samples_per_file=n_samples_per_file)
        # Not existing path: treat string as dataset id
        return build_df(path, n_samples_per_file=n_samples_per_file)
    if ds_id:
        return build_df(ds_id, n_samples_per_file=n_samples_per_file)
    raise SystemExit("[mi-race] Provide data.path or data.id in config.")


def _safe_array(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        arr = np.asarray(x, dtype=float)
        return arr
    return np.array([], dtype=float)


def summarize_sequence_col(series: pd.Series, prefix: str) -> pd.DataFrame:
    """Convert a sequence column into statistical features."""
    arrays = series.apply(_safe_array)
    lengths = arrays.apply(len).to_numpy()
    means = np.array([a.mean() if a.size else np.nan for a in arrays])
    stds  = np.array([a.std(ddof=0) if a.size else np.nan for a in arrays])
    mins  = np.array([a.min() if a.size else np.nan for a in arrays])
    maxs  = np.array([a.max() if a.size else np.nan for a in arrays])

    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    qs = {f"{prefix}_q{int(q*100)}": np.array([np.quantile(a, q) if a.size else np.nan for a in arrays])
          for q in quantiles}
    return pd.DataFrame({ **{
        f"{prefix}_len": lengths,
        f"{prefix}_mean": means,
        f"{prefix}_std": stds,
        f"{prefix}_min": mins,
        f"{prefix}_max": maxs,
    }, **qs })


def split_sequence_col(series: pd.Series, prefix: str) -> pd.DataFrame:
    """Split a sequence column into multiple time-step columns."""
    arrays = series.apply(_safe_array)
    max_length = arrays.apply(len).max()
    if max_length == 0:
        return pd.DataFrame()
    split_data = {}
    for i in range(max_length):
        col_name = f"{prefix}_{i}"
        split_data[col_name] = arrays.apply(lambda arr: arr[i] if i < len(arr) else np.nan)
    return pd.DataFrame(split_data)


def extract_sequence_range(series: pd.Series, prefix: str, start_idx: int, end_idx: int) -> pd.DataFrame:
    arrays = series.apply(_safe_array)
    split_data = {}
    for i in range(start_idx, end_idx + 1):
        col_name = f"{prefix}_{i}"
        split_data[col_name] = arrays.apply(lambda arr: arr[i] if i < len(arr) else np.nan)
    return pd.DataFrame(split_data)


def summarize_cell(x, n_head=3, n_tail=3):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    if isinstance(x, (np.ndarray, list, tuple)) and isinstance(x, Sequence):
        arr = np.asarray(x)
        if arr.size <= n_head + n_tail + 1:
            body = ", ".join(map(str, arr.tolist()))
        else:
            head = ", ".join(map(str, arr[:n_head].tolist()))
            tail = ", ".join(map(str, arr[-n_tail:].tolist()))
            body = f"{head}, …, {tail}"
        return f"[{body}] (len={arr.size})"
    return str(x)


def build_features_from_config(df: pd.DataFrame, cfg: dict) -> Tuple[pd.DataFrame, list[str]]:
    """Create the feature dataframe according to `data` config.
    Returns: (feature_df, resolved_feature_columns)
    """
    data_cfg = cfg["data"]
    y_col = data_cfg.get("y_col")

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
        x_cols = [c for c in df.columns if c != y_col and pd.api.types.is_numeric_dtype(df[c])]
    else:
        raise SystemExit(f"[mi-race] Invalid data.x_cols type: {type(x_cols_raw).__name__}")

    seen = set()
    x_cols = [c for c in x_cols if not (c in seen or seen.add(c))]
    if not x_cols:
        raise SystemExit("[mi-race] No feature columns resolved.")

    seq_mode = data_cfg.get("sequence_mode", "stats")  # 'stats' | 'ignore' | 'split'
    new_feature_frames = []
    keep_cols = []

    range_specs = []
    regular_cols = []
    for c in x_cols:
        if ":" in c:
            range_specs.append(c)
        else:
            regular_cols.append(c)

    missing = [c for c in regular_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"[mi-race] Missing feature columns: {missing}. Available: {df.columns.tolist()}")
    if y_col in regular_cols:
        raise SystemExit(f"[mi-race] y_col '{y_col}' cannot be among x_cols.")

    for range_spec in range_specs:
        try:
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
            range_df = extract_sequence_range(series, col_name, start_idx, end_idx)
            if not range_df.empty:
                new_feature_frames.append(range_df)
            else:
                pass
        except (ValueError, IndexError) as e:
            raise SystemExit(f"[mi-race] Invalid sequence range specification '{range_spec}': {e}")

    for c in regular_cols:
        if c not in df.columns:
            continue
        s = df[c]
        if s.dtype == object and s.head(5).apply(lambda v: isinstance(v, (list, tuple, np.ndarray))).all():
            if seq_mode == "stats":
                stats_df = summarize_sequence_col(s, c)
                new_feature_frames.append(stats_df)
            elif seq_mode == "split":
                split_df = split_sequence_col(s, c)
                if not split_df.empty:
                    new_feature_frames.append(split_df)
            elif seq_mode == "ignore":
                continue
            else:
                raise SystemExit(f"[mi-race] Unknown sequence_mode '{seq_mode}'. Supported: 'stats', 'split', 'ignore'")
        else:
            keep_cols.append(c)

    feature_df = df[keep_cols].copy()
    for fdf in new_feature_frames:
        feature_df = pd.concat([feature_df, fdf], axis=1)

    # additionally, split any declared sequence columns
    seq_declared = data_cfg.get("sequence_cols", [])
    if isinstance(seq_declared, (list, tuple)):
        for sc in seq_declared:
            if sc in df.columns:
                s = df[sc]
                if s.dtype == object and s.head(5).apply(lambda v: isinstance(v, (list, tuple, np.ndarray))).all():
                    split_df = split_sequence_col(s, sc)
                    if not split_df.empty:
                        feature_df = pd.concat([feature_df, split_df], axis=1)

    resolved_feature_cols = feature_df.columns.tolist()
    return feature_df, resolved_feature_cols

