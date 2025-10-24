from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
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
    cols = [c for c in df.columns if c.startswith("noise_")]
    try:
        cols = sorted(cols, key=lambda s: float(s.split("_", 1)[1]))
    except Exception:
        cols = sorted(cols)
    return cols


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


def print_accuracy_vs_noise_grouped_bars(df: pd.DataFrame) -> None:
    noise_cols = _noise_columns(df)
    if not noise_cols:
        print("\n[mi-race][compare] No noise_* columns found in summary; skipping 'accuracy vs noise'.")
        return

    # Models in the order they appear
    models = [str(m) for m in df["model"].tolist()]
    width = max(10, _term_width() - 24)

    print("\n=== Accuracy vs Noise (grouped bars) ===")
    # Legend
    legend = "  Legend: " + "  ".join(f"{_color_for(m)}■{ANSI_RESET} {m}" for m in models)
    print(legend)

    for c in noise_cols:
        # heading per noise
        try:
            noise_val = float(c.split("_", 1)[1])
            noise_label = f"noise {noise_val:g}"
        except Exception:
            noise_label = c
        print(f"\n{noise_label}:")

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
    print_accuracy_vs_noise_grouped_bars(df)
