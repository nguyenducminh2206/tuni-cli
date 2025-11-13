from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import shutil

import pandas as pd


# ---------- IO ----------

def _read_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"[mi-race][compare] summary not found: {path}")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise SystemExit(f"[mi-race][compare] failed to read {path}: {e}")
    if "model" not in df.columns:
        raise SystemExit(f"[mi-race][compare] invalid summary (missing 'model' column): {path}")
    if "accuracy" not in df.columns:
        df["accuracy"] = pd.NA
    return df


def _noise_columns(df: pd.DataFrame) -> List[str]:
    # Deprecated thin wrapper: prefer using _split_prefixes to discover arbitrary split_* columns.
    return _split_prefixes(df).get("noise", [])


def _split_prefixes(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Discover columns of the form '<prefix>_<suffix>' and group them by prefix.

    Returns a dict prefix -> sorted list of matching columns. Suffixes are sorted numerically when
    possible, otherwise lexicographically.
    """
    cols = [c for c in df.columns if c not in ("model", "accuracy")]
    groups: Dict[str, List[str]] = {}
    for c in cols:
        if "_" not in c:
            continue
        pref, suf = c.split("_", 1)
        groups.setdefault(pref, []).append(c)

    def _sort_cols(col_list: List[str]) -> List[str]:
        try:
            # try numeric sort on suffix
            return sorted(col_list, key=lambda s: float(s.split("_", 1)[1]))
        except Exception:
            return sorted(col_list)

    return {p: _sort_cols(v) for p, v in groups.items()}


# ---------- Terminal helpers ----------

def _term_width(default: int = 80) -> int:
    try:
        return shutil.get_terminal_size((default, 20)).columns
    except Exception:
        return default


def _bar(value: float, width: int, fill_char: str = "█") -> str:
    if pd.isna(value):
        value = 0.0
    v = max(0.0, min(1.0, float(value)))
    n = int(round(v * width))
    return fill_char * n + " " * (width - n)


ANSI_RESET = "\x1b[0m"
MODEL_COLORS: Dict[str, str] = {
    # basic ANSI foreground colors
    "mlp": "\x1b[38;5;114m",   # light green
    "cnn": "\x1b[38;5;177m",   # light magenta
    "rnn": "\x1b[38;5;81m",    # light cyan
    "rf": "\x1b[38;5;220m",     # yellow
}


def _color_for(model: str) -> str:
    return MODEL_COLORS.get(model, "\x1b[38;5;214m")  # default orange


# ---------- Plots ----------

def print_model_accuracy_bar(df: pd.DataFrame) -> None:
    print("\n=== Model Accuracy (Overall) ===")
    plot_df = df[["model", "accuracy"]].copy()
    plot_df["accuracy"] = pd.to_numeric(plot_df["accuracy"], errors="coerce")
    plot_df = plot_df.sort_values("accuracy", ascending=False)
    width = max(10, _term_width() - 30)
    for _, r in plot_df.iterrows():
        model = str(r.get("model"))
        acc = float(r.get("accuracy")) if pd.notna(r.get("accuracy")) else float("nan")
        color = _color_for(model)
        bar = _bar(acc if pd.notna(acc) else 0.0, width)
        acc_txt = f"{acc:.4f}" if pd.notna(acc) else "nan"
        print(f"{model:>6} |{color}{bar}{ANSI_RESET}| {acc_txt}")


def print_accuracy_vs_noise_grouped_bars(df: pd.DataFrame, split_prefix: Optional[str] = None) -> None:
    prefixes = _split_prefixes(df)
    if split_prefix:
        if split_prefix not in prefixes:
            print(f"\n[mi-race][compare] No columns found for split prefix '{split_prefix}'; available: {sorted(prefixes.keys())}")
            return
        prefixes = {split_prefix: prefixes[split_prefix]}
    if not prefixes:
        print("\n[mi-race][compare] No per-split columns (like noise_* or kcross_*) found in summary; skipping grouped comparisons.")
        return

    # Models in the order they appear
    models = [str(m) for m in df["model"].tolist()]
    width = max(10, _term_width() - 24)

    print("\n=== Accuracy vs splits (grouped bars) ===")
    # Legend
    legend = "  Legend: " + "  ".join(f"{_color_for(m)}■{ANSI_RESET} {m}" for m in models)
    print(legend)

    # For each prefix (e.g., noise, kcross) print grouped bars per suffix column
    for prefix, cols in prefixes.items():
        print(f"\n--- {prefix} ---")
        for c in cols:
            # heading per split column
            try:
                suf = c.split("_", 1)[1]
                # try numeric label for nicer heading
                try:
                    suf_val = float(suf)
                    label = f"{prefix} {suf_val:g}"
                except Exception:
                    label = f"{prefix} {suf}"
            except Exception:
                label = c
            print(f"\n{label}:")

            for _, row in df.iterrows():
                model = str(row.get("model"))
                acc = pd.to_numeric(row.get(c), errors="coerce")
                color = _color_for(model)
                bar = _bar(acc if pd.notna(acc) else 0.0, width)
                acc_txt = f"{float(acc):.4f}" if pd.notna(acc) else "nan"
                print(f"  {model:>6} |{color}{bar}{ANSI_RESET}| {acc_txt}")


def run_compare(args=None) -> None:
    """CLI entry: read outputs/summary_models.csv and print two terminal plots."""
    base = Path.cwd() / "outputs"
    path = base / "summary_models.csv"
    df = _read_summary(path)
    print_model_accuracy_bar(df)
    # If args provided, allow restricting to a single split prefix via args.split
    split_pref = None
    if args is not None and hasattr(args, "split"):
        split_pref = getattr(args, "split")
    print_accuracy_vs_noise_grouped_bars(df, split_pref)
