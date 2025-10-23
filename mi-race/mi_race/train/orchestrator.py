"""Runner orchestrator for mi-race CLI.

This refactored module delegates data preparation and model training to
mi_race.train.data_prep and mi_race.train.models.*, and uses
mi_race.reporting.report to write per-model reports.
"""

from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

from ..analysis import info_from_confusion_matrix
from .data_prep import (
    load_df_from_cfg as dp_load_df_from_cfg,
    build_features_from_config,
    preview_block as dp_preview_block,
    ensure_outdir as dp_ensure_outdir,
)
from .models.mlp import run_mlp
from ..reporting.report import build_report_text, save_per_model_artifacts


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
    print(f"[mi-race] Final feature columns (n={len(resolved_feature_cols)}): {resolved_feature_cols}")

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

    # Determine which models to run
    model_section = cfg.get("model", {})
    selection = getattr(args, "model", None)
    run_specs: list[tuple[str, dict]] = []
    if selection == "all":
        for key, val in model_section.items():
            if isinstance(val, dict) and key in ("mlp", "cnn"):
                run_specs.append((key, val))
        if not run_specs:
            # fallback to single
            mtype = model_section.get("type", "mlp").lower()
            run_specs.append((mtype, model_section))
    else:
        if selection in ("mlp", "cnn"):
            mtype = selection
            selected_cfg = model_section.get(mtype, model_section)
        else:
            # default
            # if nested keys exist, default to 'mlp' if present, else first available
            if any(k in ("mlp", "cnn") for k in model_section.keys()):
                if "mlp" in model_section:
                    mtype = "mlp"
                    selected_cfg = model_section["mlp"]
                else:
                    mtype = "cnn"
                    selected_cfg = model_section["cnn"]
            else:
                mtype = model_section.get("type", "mlp").lower()
                selected_cfg = model_section
        run_specs.append((mtype, selected_cfg))

    # Print preview once
    print(dp_preview_block(df))
    print(f"Total rows: {len(df):,}  •  Classes: {sorted(pd.Series(y).dropna().unique().tolist())}\n")

    results = []
    for mtype, mcfg in run_specs:
        print(f"\n[mi-race] ===== Running model: {mtype} =====")
        if mtype == "mlp":
            y_test, y_pred = run_mlp(feature_df, y, train_cfg, mcfg, standardize, random_state, stratify, full_counts)
        elif mtype == "cnn":
            # Lazy import to avoid requiring torch unless needed
            from .models.cnn import run_cnn
            y_test, y_pred = run_cnn(feature_df, y, train_cfg, mcfg, standardize, random_state, stratify)
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

        results.append({"model": mtype, "accuracy": acc, "macro_f1": macro_f1})

    if len(results) > 1:
        summary_path = base_out / "summary_models.csv"
        pd.DataFrame(results).to_csv(summary_path, index=False)
        print(f"\n[mi-race] Wrote summary: {summary_path}")
