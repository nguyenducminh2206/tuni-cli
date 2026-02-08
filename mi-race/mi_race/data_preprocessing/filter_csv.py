from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Tuple

import numpy as np
import pandas as pd

from ..train.data_prep import extract_sequence_range


def _safe_filename_part(text: str) -> str:
    s = str(text).strip()
    if not s:
        return "empty"

    s = re.sub(r"[\\/:*?\"<>|]+", "_", s)
    s = s.strip(" .")    
    if len(s) > 64:
        s = s[:64]
    return s or "value"


def _infer_input_csv_from_config(config_path: Path) -> Path:
    if not config_path.exists():
        raise SystemExit(f"[mi-race] Config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg: dict[str, Any] = json.load(f)

    data_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else {}
    path_raw = data_cfg.get("path")
    if not path_raw:
        raise SystemExit("[mi-race] filter requires data.path in config or use --file")

    p = Path(str(path_raw))
    if not p.exists():
        raise SystemExit(f"[mi-race] data.path does not exist: {p}")

    # Match concat's default output naming (now uses workspace temp/).
    if p.is_dir():
        candidate = Path("temp") / f"{p.name}_concat.csv"
        return candidate

    # If data.path points at a CSV, use that directly.
    if p.is_file() and p.suffix.lower() == ".csv":
        return p

    raise SystemExit(f"[mi-race] data.path must be a folder or .csv file: {p}")


def _pick_column(columns: Iterable[str]) -> str:
    cols = list(columns)
    if not cols:
        raise SystemExit("[mi-race] CSV has no columns")

    max_show = 30
    print("[mi-race] Available columns:")
    for i, c in enumerate(cols[:max_show], start=1):
        print(f"  {i:>2}. {c}")
    if len(cols) > max_show:
        print(f"  ... ({len(cols) - max_show} more)")

    while True:
        raw = input("[mi-race] Choose a column (name or number): ").strip()
        if not raw:
            continue

        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(cols):
                return cols[idx - 1]
            print(f"[mi-race] Invalid number. Enter 1..{len(cols)}")
            continue

        # Case-insensitive exact match if unique
        low = raw.lower()
        exact = [c for c in cols if c == raw]
        if exact:
            return exact[0]
        ci = [c for c in cols if c.lower() == low]
        if len(ci) == 1:
            return ci[0]

        print("[mi-race] Column not found. Try again.")


def _parse_value_for_series(raw: str, s: pd.Series) -> Tuple[object, str]:
    raw = raw.strip()
    dtype = str(s.dtype)

    if pd.api.types.is_bool_dtype(s):
        low = raw.lower()
        if low in {"1", "true", "t", "yes", "y"}:
            return True, dtype
        if low in {"0", "false", "f", "no", "n"}:
            return False, dtype
        return raw, dtype

    if pd.api.types.is_numeric_dtype(s):
        try:
            return float(raw), dtype
        except ValueError:
            return raw, dtype

    if pd.api.types.is_datetime64_any_dtype(s):
        try:
            return pd.to_datetime(raw), dtype
        except Exception:
            return raw, dtype

    return raw, dtype


def _apply_filter(df: pd.DataFrame, column: str, value: object) -> pd.DataFrame:
    s = df[column]

    if pd.api.types.is_numeric_dtype(s) and isinstance(value, (int, float, np.floating)):
        if pd.api.types.is_float_dtype(s):
            mask = np.isclose(s.to_numpy(dtype=float, copy=False), float(value), rtol=0.0, atol=1e-12)
            mask = pd.Series(mask, index=df.index)
        else:
            mask = s == value
    else:
        mask = s == value

    return df.loc[mask]


def _safe_expr_filename(expr: str) -> str:
    # compact + filesystem safe
    expr = expr.strip().replace(" ", "")
    expr = expr.replace("&&", "&")
    return _safe_filename_part(expr)


def default_filtered_temp_path(source_stem: str, expr: str) -> Path:
    out_dir = Path("temp")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{source_stem}__filtered__{_safe_expr_filename(expr)}.csv"


_CLAUSE_RE = re.compile(
    r"^\s*(?P<col>[^<>=!]+?)\s*(?P<op>==|!=|>=|<=|=|>|<)\s*(?P<val>.+?)\s*$"
)


def _split_filter_expr(expr: str) -> list[str]:
    expr = str(expr or "").strip()
    if not expr:
        return []
    expr = expr.replace("&&", "&")
    return [p.strip() for p in expr.split("&") if p.strip()]


def _strip_quotes(v: str) -> str:
    v = v.strip()
    if len(v) >= 2 and ((v[0] == v[-1] == '"') or (v[0] == v[-1] == "'")):
        return v[1:-1]
    return v


def _coerce_value(raw: str, series: pd.Series) -> object:
    txt = _strip_quotes(str(raw))
    low = txt.lower()
    if low in {"nan", "none", "null"}:
        return np.nan

    if pd.api.types.is_bool_dtype(series):
        if low in {"1", "true", "t", "yes", "y"}:
            return True
        if low in {"0", "false", "f", "no", "n"}:
            return False

    if pd.api.types.is_numeric_dtype(series):
        try:
            return float(txt)
        except ValueError:
            return txt

    if pd.api.types.is_datetime64_any_dtype(series):
        try:
            return pd.to_datetime(txt)
        except Exception:
            return txt

    return txt


def _compare(series: pd.Series, op: str, value: object) -> pd.Series:
    if op == "=":
        op = "=="

    if isinstance(value, float) and np.isnan(value):
        if op == "==":
            return series.isna()
        if op == "!=":
            return ~series.isna()
        raise SystemExit("[mi-race] NaN comparisons only support == or !=")

    if pd.api.types.is_numeric_dtype(series) and isinstance(value, (int, float, np.floating)):
        if op == "==":
            if pd.api.types.is_float_dtype(series):
                return pd.Series(
                    np.isclose(series.to_numpy(dtype=float, copy=False), float(value), rtol=0.0, atol=1e-12),
                    index=series.index,
                )
            return series == value
        if op == "!=":
            if pd.api.types.is_float_dtype(series):
                return pd.Series(
                    ~np.isclose(series.to_numpy(dtype=float, copy=False), float(value), rtol=0.0, atol=1e-12),
                    index=series.index,
                )
            return series != value
        if op == ">":
            return series > value
        if op == ">=":
            return series >= value
        if op == "<":
            return series < value
        if op == "<=":
            return series <= value

    if op == "==":
        return series == value
    if op == "!=":
        return series != value
    if op == ">":
        return series > value
    if op == ">=":
        return series >= value
    if op == "<":
        return series < value
    if op == "<=":
        return series <= value

    raise SystemExit(f"[mi-race] Unsupported operator: {op}")


def _parse_range_token(token: str) -> tuple[str, int, int]:
    parts = token.split(":")
    if len(parts) != 2:
        raise ValueError("Invalid range format")

    def parse_part(t: str) -> tuple[str, int]:
        t = t.strip()
        if "_" not in t:
            raise ValueError("Range token must be of the form '<col>_<index>'")
        base, idx_str = t.rsplit("_", 1)
        return base, int(idx_str)

    base_l, start_idx = parse_part(parts[0])
    base_r, end_idx = parse_part(parts[1])
    if base_l != base_r:
        raise ValueError("Range endpoints must refer to the same base column")
    if start_idx > end_idx:
        raise ValueError("Start index must be <= end index")
    return base_l, start_idx, end_idx


def apply_filter_expression(df: pd.DataFrame, expr: str) -> pd.DataFrame:
    """Apply a simple AND-only filter expression to a dataframe.

    Examples:
      - "sigma==0.5&&mu==0.5"
      - "mu=0.5&sigma=1"

    Operators: ==, =, !=, >=, <=, >, <
    AND: '&' or '&&'

    Also supports sequence range tokens on the LHS like:
      - "time_point_1:time_point_11==0" (applies to ALL columns in the range)
    """
    clauses = _split_filter_expr(expr)
    if not clauses:
        return df

    mask = pd.Series(True, index=df.index)
    for clause in clauses:
        m = _CLAUSE_RE.match(clause)
        if not m:
            raise SystemExit(f"[mi-race] Invalid filter clause: '{clause}'. Expected e.g. mu==0.5")

        col_token = m.group("col").strip()
        op = m.group("op").strip()
        val_raw = m.group("val").strip()

        # Range token
        if ":" in col_token:
            try:
                base, start_idx, end_idx = _parse_range_token(col_token)
            except Exception as e:
                raise SystemExit(f"[mi-race] Invalid range token in filter '{col_token}': {e}")

            range_cols = [f"{base}_{i}" for i in range(start_idx, end_idx + 1)]

            # Case 1: base sequence column exists
            if base in df.columns:
                derived = extract_sequence_range(df[base], base, start_idx, end_idx)
                local = pd.Series(True, index=df.index)
                for c in range_cols:
                    if c not in derived.columns:
                        raise SystemExit(f"[mi-race] Derived filter column missing: {c}")
                    v = _coerce_value(val_raw, derived[c])
                    local &= _compare(derived[c], op, v)
                mask &= local
                continue

            # Case 2: split columns exist already
            missing = [c for c in range_cols if c not in df.columns]
            if missing:
                available = [c for c in df.columns if c.startswith(f"{base}_")]
                hint = f" Available split cols: {available[:25]}" + (" ..." if len(available) > 25 else "")
                raise SystemExit(f"[mi-race] Filter references missing columns: {missing}.{hint}")

            local = pd.Series(True, index=df.index)
            for c in range_cols:
                v = _coerce_value(val_raw, df[c])
                local &= _compare(df[c], op, v)
            mask &= local
            continue

        # Single column
        if col_token not in df.columns:
            raise SystemExit(f"[mi-race] Filter column '{col_token}' not found. Available: {df.columns.tolist()}")

        series = df[col_token]
        value = _coerce_value(val_raw, series)
        mask &= _compare(series, op, value)

    out = df.loc[mask].reset_index(drop=True)
    if out.empty:
        raise SystemExit(f"[mi-race] Filter produced 0 rows (expr='{expr}')")
    return out


def filter_csv_interactive(
    *,
    input_csv: Path,
    out_csv: Path | None = None,
    column: str | None = None,
    value_raw: str | None = None,
) -> Path:
    if not input_csv.exists():
        raise SystemExit(
            f"[mi-race] Input CSV not found: {input_csv}. "
            "If you used concat, run 'concat csv-files' first or pass --file."
        )

    df = pd.read_csv(input_csv)
    if df.empty:
        raise SystemExit(f"[mi-race] CSV is empty: {input_csv}")

    chosen_column = column or _pick_column(df.columns)

    # Loop so we can handle 'value not present'
    while True:
        s = df[chosen_column]

        if value_raw is None:
            uniques = s.dropna().unique()
            sample = uniques[:15]
            dtype = str(s.dtype)
            print(f"[mi-race] Column '{chosen_column}' dtype={dtype} (unique≈{len(uniques):,})")
            if len(sample):
                print("[mi-race] Example values:")
                for v in sample:
                    print(f"  - {v}")
            value_raw = input(f"[mi-race] Enter value for {chosen_column}: ").strip()
            if not value_raw:
                value_raw = None
                continue

        value, dtype = _parse_value_for_series(value_raw, s)
        filtered = _apply_filter(df, chosen_column, value)

        if len(filtered) == 0:
            print(f"[mi-race] No rows match: {chosen_column} == {value_raw}")
            again = input("[mi-race] Try a different value? (y/n): ").strip().lower()
            if again in {"y", "yes"}:
                value_raw = None
                continue
            raise SystemExit("[mi-race] Aborted (no matching rows)")

        # Optionally chain additional filters
        while True:
            more = input("[mi-race] Add another filter condition? (y/n): ").strip().lower()
            if more in {"n", "no", ""}:
                break
            if more in {"y", "yes"}:
                next_col = _pick_column(filtered.columns)
                next_val_raw = input(f"[mi-race] Enter value for {next_col}: ").strip()
                if not next_val_raw:
                    print("[mi-race] Empty value; skipping.")
                    continue
                next_val, _ = _parse_value_for_series(next_val_raw, filtered[next_col])
                next_filtered = _apply_filter(filtered, next_col, next_val)
                if len(next_filtered) == 0:
                    print(f"[mi-race] No rows match: {next_col} == {next_val_raw}")
                    continue
                filtered = next_filtered
                continue

            print("[mi-race] Please answer y/n")

        if out_csv is None:
            col_part = _safe_filename_part(chosen_column)
            val_part = _safe_filename_part(value_raw)
            out_csv = input_csv.parent / f"{input_csv.stem}__filtered__{col_part}={val_part}{input_csv.suffix}"

        filtered.to_csv(out_csv, index=False)
        print(f"[mi-race] Wrote: {out_csv}")
        print(f"[mi-race] Rows: {len(filtered):,} (from {len(df):,})  Columns: {len(filtered.columns):,}")
        return out_csv


def run_filter_csv(args) -> None:
    cfg_path = Path(getattr(args, "config", "config.json"))
    file_arg = getattr(args, "file", None)
    out_arg = getattr(args, "out", None)
    column = getattr(args, "column", None)
    value = getattr(args, "value", None)

    input_csv = Path(file_arg) if file_arg else _infer_input_csv_from_config(cfg_path)
    out_csv = Path(out_arg) if out_arg else None

    filter_csv_interactive(input_csv=input_csv, out_csv=out_csv, column=column, value_raw=value)
