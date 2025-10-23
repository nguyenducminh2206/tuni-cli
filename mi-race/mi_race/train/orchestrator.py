"""Runner orchestrator for mi-race CLI.

This refactored module delegates data preparation and model training to
mi_race.train.data_prep and mi_race.train.models.*, and uses
mi_race.reporting.report to write per-model reports.
"""

from __future__ import annotations

from pathlib import Path
import json

import re
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

from ..analysis import info_from_confusion_matrix
from .data_prep import (
    load_df_from_cfg as dp_load_df_from_cfg,
    build_features_from_config,
    ensure_outdir as dp_ensure_outdir,
)
from .models.mlp import run_mlp
from ..reporting.report import build_report_text, save_per_model_artifacts


def _summarize_feature_columns(cols: list[str], *, threshold: int = 5, max_show_other: int = 20) -> str:
    """Summarize feature columns compactly.

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
    """Create or update outputs/summary_models.csv with the latest accuracy per model."""
    path = base_out / "summary_models.csv"
    if path.exists():
        try:
            df = pd.read_csv(path)
        except Exception:
            df = pd.DataFrame(columns=["model", "accuracy"])
    else:
        df = pd.DataFrame(columns=["model", "accuracy"])
    if "model" not in df.columns or "accuracy" not in df.columns:
        df = pd.DataFrame(columns=["model", "accuracy"])
    if (df["model"] == model_name).any():
        df.loc[df["model"] == model_name, "accuracy"] = accuracy
    else:
        df = pd.concat([df, pd.DataFrame([{"model": model_name, "accuracy": accuracy}])], ignore_index=True)
    df.to_csv(path, index=False)
    return path


def run_cmd(args):
    # Load config
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise SystemExit(f"[mi-race] Config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Data section
    if "data" not in cfg:
        raise SystemExit("[mi-race] Missing 'data' section in config.")
    data_cfg = cfg["data"]

    # Load dataset
    df = dp_load_df_from_cfg(data_cfg)
    print(f"[mi-race] Loaded dataset with columns: {df.columns.tolist()}")

    # Target
    y_col = data_cfg.get("y_col")
    if not y_col:
        raise SystemExit("[mi-race] data.y_col must be specified.")
    if y_col not in df.columns:
        raise SystemExit(f"[mi-race] y_col '{y_col}' not found. Available: {df.columns.tolist()}")

    # Build features
    feature_df, resolved_feature_cols = build_features_from_config(df, cfg)
    summary_cols_text = _summarize_feature_columns(resolved_feature_cols)
    print(f"[mi-race] Final feature columns (n={len(resolved_feature_cols)}):\n{summary_cols_text}")

    # Save processed features
    out_cfg = cfg.get("output", {})
    base_out = Path(out_cfg.get("dir", "outputs"))
    dp_ensure_outdir(base_out)
    feature_df_path = base_out / "processed_features.csv"
    feature_df.to_csv(feature_df_path, index=False)
    print(f"[mi-race] Saved processed features to: {feature_df_path}")
    

    # Build arrays for general stats and counts
    y = df[y_col].to_numpy()
    full_counts = pd.Series(y).value_counts().sort_index()
    print(f"[mi-race] Class distribution (full): {full_counts.to_dict()}")

    # Train settings
    train_cfg = cfg.get("train", {})
    random_state = int(train_cfg.get("random_state", 42))
    standardize = bool(train_cfg.get("standardize", True))
    stratify = y if train_cfg.get("stratify", True) else None

    # Determine which model to run
    model_section = cfg.get("model", {})
    selection = getattr(args, "model", None)
    if selection not in (None, "mlp", "cnn"):
        raise SystemExit("[mi-race] Unsupported --model value. Use 'mlp' or 'cnn'.")
    if selection in ("mlp", "cnn"):
        mtype = selection
        selected_cfg = model_section.get(mtype, model_section.get(mtype, {})) if isinstance(model_section, dict) else {}
    else:
        # default by config
        if isinstance(model_section, dict) and any(k in ("mlp", "cnn") for k in model_section.keys()):
            if "mlp" in model_section:
                mtype = "mlp"
                selected_cfg = model_section["mlp"]
            elif "cnn" in model_section:
                mtype = "cnn"
                selected_cfg = model_section["cnn"]
            else:
                mtype = "mlp"
                selected_cfg = {}
        else:
            mtype = str(model_section.get("type", "mlp")).lower() if isinstance(model_section, dict) else "mlp"
            selected_cfg = model_section if isinstance(model_section, dict) else {}
    print(f"\n[mi-race] ===== Running model: {mtype} =====")
    if mtype == "mlp":
        y_test, y_pred = run_mlp(feature_df, y, train_cfg, selected_cfg, standardize, random_state, stratify, full_counts)
    elif mtype == "cnn":
        # Lazy import to avoid requiring torch unless needed
        from .models.cnn import run_cnn
        y_test, y_pred = run_cnn(feature_df, y, train_cfg, selected_cfg, standardize, random_state, stratify)
    else:
        raise SystemExit("[mi-race] Unknown model type. Supported: 'mlp', 'cnn'.")

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    n_classes = sorted(pd.Series(y).dropna().unique().tolist())
    cm = confusion_matrix(y_test, y_pred, labels=n_classes)
    info = info_from_confusion_matrix(cm, labels=n_classes)
    preds_unique = sorted(pd.Series(y_pred).dropna().unique().tolist())
    missing_predicted = [c for c in n_classes if c not in preds_unique]
    if missing_predicted:
        print(f"[mi-race][WARN][{mtype}] No predicted samples for classes: {missing_predicted}. Metrics with zero_division=0.")
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    # Report
    report_txt = build_report_text(mtype, n_classes, acc, macro_f1, cm, info, y_test, y_pred,
                                   show_clf_report=cfg.get("output", {}).get("show_report", True))
    saved = save_per_model_artifacts(base_out, mtype, cm, info, report_txt)
    print(report_txt)
    print(f"Saved: {saved['cm']}")
    print(f"Saved: {saved['info']}")
    print(f"Saved: {saved['report']}")

    # Update simple accuracy summary across runs
    summary_path = _update_accuracy_summary(base_out, mtype, float(acc))
    print(f"[mi-race] Updated accuracy summary: {summary_path}")

    # No multi-model loop here; summary is updated incrementally per run.
