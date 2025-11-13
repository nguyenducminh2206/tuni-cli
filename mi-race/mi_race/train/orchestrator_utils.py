"""Utility helpers for the training orchestrator.

This module holds pure helper functions used by `mi_race.train.orchestrator` to
keep the orchestrator focused on high-level flow.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import List

import pandas as pd


def _summarize_feature_columns(cols: List[str], *, threshold: int = 5, max_show_other: int = 20) -> str:
    """Summarize feature column names compactly.

    - Detect split sequence groups of the form '<prefix>_<idx>' where count >= threshold.
    - Print each group as: '<prefix>_*: <prefix>_<min>.. <prefix>_<max> (<count> cols)'
    - Then list a shortened set of non-grouped feature names up to max_show_other.
    """
    pat = re.compile(r"^(?P<prefix>.+)_(?P<idx>\d+)$")
    groups: dict[str, list[tuple[int, str]]] = {}
    grouped_cols: set[str] = set()

    for c in cols:
        m = pat.match(c)
        if not m:
            continue
        pref = m.group("prefix")
        idx = int(m.group("idx"))
        groups.setdefault(pref, []).append((idx, c))

    # Decide which prefixes qualify as split groups
    split_summaries = []
    for pref, items in sorted(groups.items()):
        if len(items) >= threshold:
            items_sorted = sorted(items, key=lambda t: t[0])
            min_idx = items_sorted[0][0]
            max_idx = items_sorted[-1][0]
            count = len(items_sorted)
            split_summaries.append(f"- {pref}_*: {pref}_{min_idx}..{pref}_{max_idx} ({count} cols)")
            for _, name in items_sorted:
                grouped_cols.add(name)

    # Non-grouped columns
    other_cols = [c for c in cols if c not in grouped_cols]
    other_display = other_cols[:max_show_other]
    more = len(other_cols) - len(other_display)
    other_line = "- Other features ({}): {}{}".format(
        len(other_cols),
        ", ".join(other_display),
        f" … (+{more} more)" if more > 0 else "",
    )

    lines = []
    if split_summaries:
        lines.append("Split groups:")
        lines.extend(split_summaries)
    lines.append(other_line)
    return "\n".join(lines)


def _update_accuracy_summary(base_out: Path, model_name: str, accuracy: float) -> Path:
    """Update outputs/summary_models.csv with overall accuracy per model (row-per-model)."""
    path = base_out / "summary_models.csv"
    # Load or initialize
    if path.exists():
        try:
            df = pd.read_csv(path)
        except Exception:
            df = pd.DataFrame(columns=["model", "accuracy"])
    else:
        df = pd.DataFrame(columns=["model", "accuracy"])

    # Ensure required columns
    if "model" not in df.columns:
        df["model"] = pd.Series(dtype=str)
    if "accuracy" not in df.columns:
        df["accuracy"] = pd.Series(dtype=float)

    # Work with 'model' as index for clean updates (avoids concat warnings)
    df = df.set_index("model", drop=False)
    if model_name not in df.index:
        df.loc[model_name, "model"] = model_name
    df.loc[model_name, "accuracy"] = float(accuracy)

    # Order columns: model, accuracy, then any other columns sorted alphabetically
    other_cols = [c for c in df.columns if c not in ("model", "accuracy")]
    other_cols = sorted(other_cols)
    ordered_cols = ["model", "accuracy"] + other_cols
    df = df[ordered_cols]
    df.to_csv(path, index=False)
    return path



def _sanitize_summary_models(base_out: Path) -> Path:
    """Clean up legacy rows/columns in outputs/summary_models.csv.

    - Drop rows with missing model or model starting with 'noise_'
    - Drop legacy columns: 'noise', '*_accuracy', '*_epochs'
    - Keep row-per-model with columns: model, accuracy, noise_*
    """
    path = base_out / "summary_models.csv"
    if not path.exists():
        return path
    try:
        df = pd.read_csv(path)
    except Exception:
        return path

    if "model" not in df.columns:
        return path

    # Drop legacy rows (where model is NaN or startswith 'noise_')
    df = df.dropna(subset=["model"]).copy()
    df = df[~df["model"].astype(str).str.startswith("noise_")]

    # Drop legacy columns: 'noise' and any '*_accuracy', '*_epochs' except 'accuracy'
    legacy_cols = []
    if "noise" in df.columns:
        legacy_cols.append("noise")
    for c in list(df.columns):
        if c.endswith("_accuracy") or c.endswith("_epochs"):
            if c != "accuracy":
                legacy_cols.append(c)
    legacy_cols = list(dict.fromkeys(legacy_cols))
    if legacy_cols:
        df = df.drop(columns=legacy_cols, errors="ignore")

    # Ensure required columns
    if "accuracy" not in df.columns:
        df["accuracy"] = pd.Series(dtype=float)

    # Order columns: model, accuracy, then other columns alphabetically
    other_cols = [c for c in df.columns if c not in ("model", "accuracy")]
    other_cols = sorted(other_cols)
    ordered_cols = ["model", "accuracy"] + other_cols
    df = df[[c for c in ordered_cols if c in df.columns]]
    df.to_csv(path, index=False)
    return path


def _update_split_accuracy_summary(base_out: Path, model_name: str, split_name: str, split_value, accuracy: float) -> Path:
    """Update outputs/summary_models.csv with per-split accuracy for a given model (row-per-model).

    split_name: column name used for splitting (e.g., 'noise' or 'kcross')
    split_value: value of that column for this subset
    The column written will be '<split_name>_<val>' where formatting depends on value type.
    """
    path = base_out / "summary_models.csv"
    # Load or initialize
    if path.exists():
        try:
            df = pd.read_csv(path)
        except Exception:
            df = pd.DataFrame(columns=["model", "accuracy"])
    else:
        df = pd.DataFrame(columns=["model", "accuracy"])

    if "model" not in df.columns:
        df["model"] = pd.Series(dtype=str)
    if "accuracy" not in df.columns:
        df["accuracy"] = pd.Series(dtype=float)

    # Format column name
    col_suffix = None
    try:
        # Prefer numeric formatting when possible for consistent ordering
        if isinstance(split_value, int):
            col_suffix = f"{int(split_value)}"
        elif isinstance(split_value, float):
            col_suffix = f"{float(split_value):.2f}"
        else:
            col_suffix = str(split_value).replace(" ", "_")
    except Exception:
        col_suffix = str(split_value)

    col_name = f"{split_name}_{col_suffix}"
    if col_name not in df.columns:
        df[col_name] = pd.Series(dtype=float)

    df = df.set_index("model", drop=False)
    if model_name not in df.index:
        df.loc[model_name, "model"] = model_name
    df.loc[model_name, col_name] = float(accuracy)

    # Order columns: model, accuracy, other columns alphabetically
    other_cols = [c for c in df.columns if c not in ("model", "accuracy")]
    other_cols = sorted(other_cols)
    ordered_cols = ["model", "accuracy"] + other_cols
    df = df[[c for c in ordered_cols if c in df.columns]]
    df.to_csv(path, index=False)
    return path
