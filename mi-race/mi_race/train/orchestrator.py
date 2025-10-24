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

    # Order columns: model, accuracy, then noise_* columns (sorted by numeric noise)
    noise_cols = [c for c in df.columns if c.startswith("noise_")]
    try:
        noise_cols = sorted(noise_cols, key=lambda s: float(s.split("_", 1)[1]))
    except Exception:
        noise_cols = sorted(noise_cols)
    ordered_cols = ["model", "accuracy"] + noise_cols
    df = df[ordered_cols]
    df.to_csv(path, index=False)
    return path


def _update_noise_accuracy_summary(base_out: Path, model_name: str, noise: float, accuracy: float) -> Path:
    """Update outputs/summary_models.csv with per-noise accuracy for a given model (row-per-model)."""
    path = base_out / "summary_models.csv"
    # Load or initialize
    if path.exists():
        try:
            df = pd.read_csv(path)
        except Exception:
            df = pd.DataFrame(columns=["model", "accuracy"])
    else:
        df = pd.DataFrame(columns=["model", "accuracy"])

    # Ensure 'model' column
    if "model" not in df.columns:
        df["model"] = pd.Series(dtype=str)
    if "accuracy" not in df.columns:
        df["accuracy"] = pd.Series(dtype=float)

    # Column name for this noise (stable two-decimal format)
    noise_col = f"noise_{noise:.2f}"
    if noise_col not in df.columns:
        df[noise_col] = pd.Series(dtype=float)

    # Index and update
    df = df.set_index("model", drop=False)
    if model_name not in df.index:
        df.loc[model_name, "model"] = model_name
    df.loc[model_name, noise_col] = float(accuracy)

    # Order columns
    noise_cols = [c for c in df.columns if c.startswith("noise_")]
    try:
        noise_cols = sorted(noise_cols, key=lambda s: float(s.split("_", 1)[1]))
    except Exception:
        noise_cols = sorted(noise_cols)
    ordered_cols = ["model", "accuracy"] + noise_cols
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

    # Order columns: model, accuracy, noise_*
    noise_cols = [c for c in df.columns if c.startswith("noise_")]
    try:
        noise_cols = sorted(noise_cols, key=lambda s: float(s.split("_", 1)[1]))
    except Exception:
        noise_cols = sorted(noise_cols)
    ordered_cols = ["model", "accuracy"] + noise_cols
    df = df[[c for c in ordered_cols if c in df.columns]]
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

    # Optional: balance by min count of label if enabled in config
    train_cfg = cfg.get("train", {})
    random_state = int(train_cfg.get("random_state", 42))
    balance_cfg = data_cfg.get("balance", False)
    enabled_balance = False
    if isinstance(balance_cfg, bool):
        enabled_balance = balance_cfg
    elif isinstance(balance_cfg, dict):
        enabled_balance = balance_cfg.get("enabled", True)
    if enabled_balance:
        before_counts = pd.Series(df[y_col]).value_counts().sort_index()
        min_count = int(before_counts.min()) if not before_counts.empty else 0
        if min_count > 0:
            df = (
                df.dropna(subset=[y_col])
                  .groupby(y_col, group_keys=False)
                  .sample(n=min_count, random_state=random_state)
                  .reset_index(drop=True)
            )
            after_counts = pd.Series(df[y_col]).value_counts().sort_index()
            print(f"[mi-race] Balancing enabled: undersampled each class to min_count={min_count}")
            print(f"[mi-race] Class distribution (before): {before_counts.to_dict()}")
            print(f"[mi-race] Class distribution (after):  {after_counts.to_dict()}")
        else:
            print("[mi-race] Balancing requested but no valid label counts; proceeding without balancing.")

    # Build features on the (optionally balanced) dataframe
    feature_df, resolved_feature_cols = build_features_from_config(df, cfg)
    summary_cols_text = _summarize_feature_columns(resolved_feature_cols)
    print(f"[mi-race] Final feature columns (n={len(resolved_feature_cols)}):\n{summary_cols_text}")

    # Save processed features
    out_cfg = cfg.get("output", {})
    base_out = Path(out_cfg.get("dir", "outputs"))
    dp_ensure_outdir(base_out)
    # Sanitize legacy summary file once per run
    _sanitize_summary_models(base_out)
    feature_df_path = base_out / "processed_features.csv"
    feature_df.to_csv(feature_df_path, index=False)
    print(f"[mi-race] Saved processed features to: {feature_df_path}")
    

    # Build arrays for general stats and counts (from the data actually used for training)
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
        # Regular overall training run
        y_test, y_pred = run_mlp(feature_df, y, train_cfg, selected_cfg, standardize, random_state, stratify, full_counts)

        # Additionally: train per noise level if a 'noise' column exists
        if "noise" in df.columns:
            try:
                noise_levels = sorted(pd.Series(df["noise"]).dropna().unique().tolist())
            except Exception:
                noise_levels = []
            for noise_level in noise_levels:
                print(f"\n=== Training for noise {noise_level} ===")
                mask = df["noise"] == noise_level
                # Subset features and labels using the same row order
                feature_df_sub = feature_df.loc[mask].reset_index(drop=True)
                y_sub = y[mask.to_numpy()] if hasattr(mask, "to_numpy") else y[mask]
                if feature_df_sub.empty or len(y_sub) == 0:
                    print(f"[mi-race][mlp] Skipping noise {noise_level}: no rows after filtering")
                    continue
                sub_counts = pd.Series(y_sub).value_counts().sort_index()
                sub_stratify = y_sub if train_cfg.get("stratify", True) else None
                epochs_like = int(train_cfg.get("max_iter", 200))  # use max_iter as epochs proxy for MLP
                y_t_sub, y_p_sub = run_mlp(
                    feature_df_sub,
                    y_sub,
                    train_cfg,
                    selected_cfg,
                    standardize,
                    random_state,
                    sub_stratify,
                    sub_counts,
                )
                # Per-noise metrics & summary update
                acc_sub = accuracy_score(y_t_sub, y_p_sub)
                labels_sub = sorted(pd.Series(y_t_sub).dropna().unique().tolist())
                cm_sub = confusion_matrix(y_t_sub, y_p_sub, labels=labels_sub)
                info_sub = info_from_confusion_matrix(cm_sub, labels=labels_sub)
                # Print noise-level summary with epochs, CM, and MI
                print(
                    f"[mi-race][mlp] noise={noise_level}  accuracy={acc_sub:.4f}  epochs={epochs_like}"
                )
                print("Confusion Matrix (noise={}):".format(noise_level))
                print(pd.DataFrame(cm_sub, index=[f"true_{l}" for l in labels_sub], columns=[f"pred_{l}" for l in labels_sub]))
                print(
                    "MI: I(true;pred)={:.4f}  NMI_sqrt={:.4f}  NMI_min={:.4f}  NMI_max={:.4f}".format(
                        info_sub.get("I", float("nan")),
                        info_sub.get("NMI_sqrt", float("nan")),
                        info_sub.get("NMI_min", float("nan")),
                        info_sub.get("NMI_max", float("nan")),
                    )
                )
                _update_noise_accuracy_summary(base_out, "mlp", float(noise_level), float(acc_sub))
    elif mtype == "cnn":
        # Lazy import to avoid requiring torch unless needed
        from .models.cnn import run_cnn
        y_test, y_pred = run_cnn(feature_df, y, train_cfg, selected_cfg, standardize, random_state, stratify)

        # Per-noise loop for CNN as well
        if "noise" in df.columns:
            try:
                noise_levels = sorted(pd.Series(df["noise"]).dropna().unique().tolist())
            except Exception:
                noise_levels = []
            for noise_level in noise_levels:
                print(f"\n=== Training for noise {noise_level} (cnn) ===")
                mask = df["noise"] == noise_level
                feature_df_sub = feature_df.loc[mask].reset_index(drop=True)
                y_sub = y[mask.to_numpy()] if hasattr(mask, "to_numpy") else y[mask]
                if feature_df_sub.empty or len(y_sub) == 0:
                    print(f"[mi-race][cnn] Skipping noise {noise_level}: no rows after filtering")
                    continue
                sub_stratify = y_sub if train_cfg.get("stratify", True) else None
                y_t_sub, y_p_sub = run_cnn(
                    feature_df_sub,
                    y_sub,
                    train_cfg,
                    selected_cfg,
                    standardize,
                    random_state,
                    sub_stratify,
                )
                acc_sub = accuracy_score(y_t_sub, y_p_sub)
                labels_sub = sorted(pd.Series(y_t_sub).dropna().unique().tolist())
                cm_sub = confusion_matrix(y_t_sub, y_p_sub, labels=labels_sub)
                info_sub = info_from_confusion_matrix(cm_sub, labels=labels_sub)
                # Epochs from CNN config
                epochs_cnn = int(selected_cfg.get("epochs", 5)) if isinstance(selected_cfg, dict) else 5
                print(
                    f"[mi-race][cnn] noise={noise_level}  accuracy={acc_sub:.4f}  epochs={epochs_cnn}"
                )
                print("Confusion Matrix (noise={}):".format(noise_level))
                print(pd.DataFrame(cm_sub, index=[f"true_{l}" for l in labels_sub], columns=[f"pred_{l}" for l in labels_sub]))
                print(
                    "MI: I(true;pred)={:.4f}  NMI_sqrt={:.4f}  NMI_min={:.4f}  NMI_max={:.4f}".format(
                        info_sub.get("I", float("nan")),
                        info_sub.get("NMI_sqrt", float("nan")),
                        info_sub.get("NMI_min", float("nan")),
                        info_sub.get("NMI_max", float("nan")),
                    )
                )
                _update_noise_accuracy_summary(base_out, "cnn", float(noise_level), float(acc_sub))
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
