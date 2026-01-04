"""Runner orchestrator for mi-race CLI.

This refactored module delegates data preparation and model training to
mi_race.train.data_prep and mi_race.train.models.*, and uses
mi_race.reporting.report to write per-model reports.
"""

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

from ..analysis import info_from_confusion_matrix
from ..analysis.metrics import ksg_mi_labels_probs
from .data_prep import (
    load_df_from_cfg as dp_load_df_from_cfg,
    build_features_from_config,
    ensure_outdir as dp_ensure_outdir,
)
from .models.mlp import run_mlp
from datetime import datetime
from ..reporting.report import (
    build_report_text,
    save_per_model_artifacts,
    build_detailed_run_report,
)
from .orchestrator_utils import (
    _summarize_feature_columns,
    _update_accuracy_summary,
    _sanitize_summary_models,
    _update_split_accuracy_summary,
)



def run_cmd(args):
    start_time = datetime.now()
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

    # Determine which model to run (before building features so we can skip splits for RNN)
    # Build arrays for general stats and counts (from the data actually used for training)
    y = df[y_col].to_numpy()
    full_counts = pd.Series(y).value_counts().sort_index()
    print(f"[mi-race] Class distribution (full): {full_counts.to_dict()}")

    # Train settings
    train_cfg = cfg.get("train", {})
    random_state = int(train_cfg.get("random_state", 42))
    standardize = bool(train_cfg.get("standardize", True))
    stratify = y if train_cfg.get("stratify", True) else None

    # Which column to split per-feature (e.g., 'noise' or 'kcross') for per-slice runs
    # Prefer explicit setting in data section, else train section, default to 'noise'
    split_feature = None
    if isinstance(data_cfg, dict):
        split_feature = data_cfg.get("split_by")
    if not split_feature:
        split_feature = train_cfg.get("split_by", "noise")

    model_section = cfg.get("model", {})
    selection = getattr(args, "model", None)
    if selection not in (None, "mlp", "cnn", "rnn", "rf"):
        raise SystemExit("[mi-race] Unsupported --model value. Use 'mlp', 'cnn', 'rnn', or 'rf'.")
    if selection in ("mlp", "cnn", "rnn", "rf"):
        mtype = selection
        selected_cfg = model_section.get(mtype, model_section.get(mtype, {})) if isinstance(model_section, dict) else {}
    else:
        # default by config
        if isinstance(model_section, dict) and any(k in ("mlp", "cnn", "rnn", "rf") for k in model_section.keys()):
            if "mlp" in model_section:
                mtype = "mlp"
                selected_cfg = model_section["mlp"]
            elif "cnn" in model_section:
                mtype = "cnn"
                selected_cfg = model_section["cnn"]
            elif "rnn" in model_section:
                mtype = "rnn"
                selected_cfg = model_section["rnn"]
            elif "rf" in model_section:
                mtype = "rf"
                selected_cfg = model_section["rf"]
            else:
                mtype = "mlp"
                selected_cfg = {}
        else:
            mtype = str(model_section.get("type", "mlp")).lower() if isinstance(model_section, dict) else "mlp"
            selected_cfg = model_section if isinstance(model_section, dict) else {}

    # For non-RNN models, build features (which may include split groups) and save
    if mtype in ("mlp", "cnn", "rf"):
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
    else:
        # For RNN, we keep raw sequences and skip split feature preview/export
        out_cfg = cfg.get("output", {})
        base_out = Path(out_cfg.get("dir", "outputs"))
        dp_ensure_outdir(base_out)
        _sanitize_summary_models(base_out)
    print(f"\n[mi-race] ===== Running model: {mtype} =====")
    if mtype == "mlp":
        # Regular overall training run
        y_test, y_pred, y_proba = run_mlp(feature_df, y, train_cfg, selected_cfg, standardize, random_state, stratify, full_counts)

        # Additionally: train per-split feature (e.g., noise or kcross) if that column exists
        if split_feature in df.columns:
            try:
                split_levels = sorted(pd.Series(df[split_feature]).dropna().unique().tolist())
            except Exception:
                split_levels = []
            for split_value in split_levels:
                print(f"\n=== Training for {split_feature} {split_value} ===")
                mask = df[split_feature] == split_value
                # Subset features and labels using the same row order
                feature_df_sub = feature_df.loc[mask].reset_index(drop=True)
                y_sub = y[mask.to_numpy()] if hasattr(mask, "to_numpy") else y[mask]
                if feature_df_sub.empty or len(y_sub) == 0:
                    print(f"[mi-race][mlp] Skipping {split_feature} {split_value}: no rows after filtering")
                    continue
                sub_counts = pd.Series(y_sub).value_counts().sort_index()
                sub_stratify = y_sub if train_cfg.get("stratify", True) else None
                epochs_like = int(selected_cfg.get("epochs", train_cfg.get("epochs", 15)))
                y_t_sub, y_p_sub, _ = run_mlp(
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
                # Print split-level summary with epochs, CM, and MI
                print(
                    f"[mi-race][mlp] {split_feature}={split_value}  accuracy={acc_sub:.4f}  epochs={epochs_like}"
                )
                print("Confusion Matrix ({}={}):".format(split_feature, split_value))
                print(pd.DataFrame(cm_sub, index=[f"true_{l}" for l in labels_sub], columns=[f"pred_{l}" for l in labels_sub]))
                print(
                    "MI: I(true;pred)={:.4f}  NMI_sqrt={:.4f}  NMI_min={:.4f}  NMI_max={:.4f}".format(
                        info_sub.get("I", float("nan")),
                        info_sub.get("NMI_sqrt", float("nan")),
                        info_sub.get("NMI_min", float("nan")),
                        info_sub.get("NMI_max", float("nan")),
                    )
                )
                _update_split_accuracy_summary(base_out, "mlp", split_feature, split_value, float(acc_sub))
    elif mtype == "cnn":
        # Lazy import to avoid requiring torch unless needed
        from .models.cnn import run_cnn
        y_test, y_pred = run_cnn(feature_df, y, train_cfg, selected_cfg, standardize, random_state, stratify)
        y_proba = None

        # Per-split-feature loop for CNN as well
        if split_feature in df.columns:
            try:
                split_levels = sorted(pd.Series(df[split_feature]).dropna().unique().tolist())
            except Exception:
                split_levels = []
            for split_value in split_levels:
                print(f"\n=== Training for {split_feature} {split_value} (cnn) ===")
                mask = df[split_feature] == split_value
                feature_df_sub = feature_df.loc[mask].reset_index(drop=True)
                y_sub = y[mask.to_numpy()] if hasattr(mask, "to_numpy") else y[mask]
                if feature_df_sub.empty or len(y_sub) == 0:
                    print(f"[mi-race][cnn] Skipping {split_feature} {split_value}: no rows after filtering")
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
                    f"[mi-race][cnn] {split_feature}={split_value}  accuracy={acc_sub:.4f}  epochs={epochs_cnn}"
                )
                print("Confusion Matrix ({}={}):".format(split_feature, split_value))
                print(pd.DataFrame(cm_sub, index=[f"true_{l}" for l in labels_sub], columns=[f"pred_{l}" for l in labels_sub]))
                print(
                    "MI: I(true;pred)={:.4f}  NMI_sqrt={:.4f}  NMI_min={:.4f}  NMI_max={:.4f}".format(
                        info_sub.get("I", float("nan")),
                        info_sub.get("NMI_sqrt", float("nan")),
                        info_sub.get("NMI_min", float("nan")),
                        info_sub.get("NMI_max", float("nan")),
                    )
                )
                _update_split_accuracy_summary(base_out, "cnn", split_feature, split_value, float(acc_sub))
    elif mtype == "rf":
        from .models.random_forest import run_random_forest
        y_test, y_pred, y_proba = run_random_forest(feature_df, y, train_cfg, selected_cfg, standardize, random_state, stratify)

        # Per-split-feature loop for RF
        if split_feature in df.columns:
            try:
                split_levels = sorted(pd.Series(df[split_feature]).dropna().unique().tolist())
            except Exception:
                split_levels = []
            for split_value in split_levels:
                print(f"\n=== Training for {split_feature} {split_value} (rf) ===")
                mask = df[split_feature] == split_value
                feature_df_sub = feature_df.loc[mask].reset_index(drop=True)
                y_sub = y[mask.to_numpy()] if hasattr(mask, "to_numpy") else y[mask]
                if feature_df_sub.empty or len(y_sub) == 0:
                    print(f"[mi-race][rf] Skipping {split_feature} {split_value}: no rows after filtering")
                    continue
                sub_stratify = y_sub if train_cfg.get("stratify", True) else None
                y_t_sub, y_p_sub = run_random_forest(
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
                print(
                    f"[mi-race][rf] {split_feature}={split_value}  accuracy={acc_sub:.4f}"
                )
                print("Confusion Matrix ({}={}):".format(split_feature, split_value))
                print(pd.DataFrame(cm_sub, index=[f"true_{l}" for l in labels_sub], columns=[f"pred_{l}" for l in labels_sub]))
                print(
                    "MI: I(true;pred)={:.4f}  NMI_sqrt={:.4f}  NMI_min={:.4f}  NMI_max={:.4f}".format(
                        info_sub.get("I", float("nan")),
                        info_sub.get("NMI_sqrt", float("nan")),
                        info_sub.get("NMI_min", float("nan")),
                        info_sub.get("NMI_max", float("nan")),
                    )
                )
                _update_split_accuracy_summary(base_out, "rf", split_feature, split_value, float(acc_sub))
    elif mtype == "rnn":
        # Use original df to keep raw sequences for RNN
        from .models.rnn import run_rnn
        seq_col_for_log = None
        try:
            # selected_cfg may be dict or something falsy
            if isinstance(selected_cfg, dict):
                seq_col_for_log = selected_cfg.get("sequence_col", "time_trace")
        except Exception:
            seq_col_for_log = "time_trace"
        if seq_col_for_log is None:
            seq_col_for_log = "time_trace"
        print(
            f"[mi-race][rnn] Using raw sequence column '{seq_col_for_log}' from the original dataframe. "
            "Split feature columns (data.sequence_mode='split') are ignored for RNN."
        )
        # Also export a processed_features_rnn.csv with padded/truncated sequences for inspection
        try:
            if seq_col_for_log in df.columns:
                # Build fixed-length matrix for export (not used for training)
                pad_value = float(selected_cfg.get("pad_value", 0.0)) if isinstance(selected_cfg, dict) else 0.0
                seq_series = df[seq_col_for_log].apply(
                    lambda x: (np.asarray(x, dtype=float).reshape(-1)
                               if isinstance(x, (list, tuple, np.ndarray)) else np.array([], dtype=float))
                )
                lengths = seq_series.apply(lambda a: a.size)
                max_len_all = int(lengths.max()) if lengths.size > 0 else 0
                # Prefer model.rnn.export_len, else model.rnn.max_len, else cap global max to 1000 for file size
                export_len = None
                if isinstance(selected_cfg, dict) and selected_cfg.get("export_len") is not None:
                    export_len = int(selected_cfg.get("export_len"))
                elif isinstance(selected_cfg, dict) and selected_cfg.get("max_len") is not None:
                    export_len = int(selected_cfg.get("max_len"))
                else:
                    export_len = min(max_len_all, 1000)
                if export_len and export_len > 0:
                    mat = np.full((len(seq_series), export_len), pad_value, dtype=float)
                    for i, arr in enumerate(seq_series):
                        m = min(arr.size, export_len)
                        if m:
                            mat[i, :m] = arr[:m]
                    cols = [f"{seq_col_for_log}_{i}" for i in range(export_len)]
                    export_df = pd.DataFrame(mat, columns=cols)
                    # Prepend label and seq_len if available
                    export_df.insert(0, "seq_len", lengths.to_numpy())
                    if y_col in df.columns:
                        export_df.insert(0, y_col, df[y_col].to_numpy())
                    export_path = base_out / "processed_features_rnn.csv"
                    export_df.to_csv(export_path, index=False)
                    print(f"[mi-race][rnn] Saved processed RNN features to: {export_path} (export_len={export_len})")
        except Exception as e:
            print(f"[mi-race][rnn][WARN] Failed to export processed RNN features: {e}")
        y_test, y_pred = run_rnn(df, y, train_cfg, selected_cfg, random_state, stratify)
        y_proba = None
        # Per-split-feature loop for RNN as well
        if split_feature in df.columns:
            try:
                split_levels = sorted(pd.Series(df[split_feature]).dropna().unique().tolist())
            except Exception:
                split_levels = []
            for split_value in split_levels:
                print(f"\n=== Training for {split_feature} {split_value} (rnn) ===")
                mask = df[split_feature] == split_value
                df_sub = df.loc[mask].reset_index(drop=True)
                y_sub = y[mask.to_numpy()] if hasattr(mask, "to_numpy") else y[mask]
                if df_sub.empty or len(y_sub) == 0:
                    print(f"[mi-race][rnn] Skipping {split_feature} {split_value}: no rows after filtering")
                    continue
                sub_stratify = y_sub if train_cfg.get("stratify", True) else None
                y_t_sub, y_p_sub = run_rnn(
                    df_sub,
                    y_sub,
                    train_cfg,
                    selected_cfg,
                    random_state,
                    sub_stratify,
                )
                acc_sub = accuracy_score(y_t_sub, y_p_sub)
                labels_sub = sorted(pd.Series(y_t_sub).dropna().unique().tolist())
                cm_sub = confusion_matrix(y_t_sub, y_p_sub, labels=labels_sub)
                info_sub = info_from_confusion_matrix(cm_sub, labels=labels_sub)
                epochs_rnn = int(selected_cfg.get("epochs", 5)) if isinstance(selected_cfg, dict) else 5
                print(
                    f"[mi-race][rnn] {split_feature}={split_value}  accuracy={acc_sub:.4f}  epochs={epochs_rnn}"
                )
                print("Confusion Matrix ({}={}):".format(split_feature, split_value))
                print(pd.DataFrame(cm_sub, index=[f"true_{l}" for l in labels_sub], columns=[f"pred_{l}" for l in labels_sub]))
                print(
                    "MI: I(true;pred)={:.4f}  NMI_sqrt={:.4f}  NMI_min={:.4f}  NMI_max={:.4f}".format(
                        info_sub.get("I", float("nan")),
                        info_sub.get("NMI_sqrt", float("nan")),
                        info_sub.get("NMI_min", float("nan")),
                        info_sub.get("NMI_max", float("nan")),
                    )
                )
                _update_split_accuracy_summary(base_out, "rnn", split_feature, split_value, float(acc_sub))
    else:
        raise SystemExit("[mi-race] Unknown model type. Supported: 'mlp', 'cnn', 'rnn'.")

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
    show_clf = cfg.get("output", {}).get("show_report", True)
    # Optional KSG MI using predicted probabilities when available
    ksg_bits = None
    try:
        if 'y_proba' in locals() and y_proba is not None:
            k = int(cfg.get("output", {}).get("ksg_k", 5))
            ksg_bits = ksg_mi_labels_probs(y_test, y_proba, k=k)
    except Exception:
        ksg_bits = None
    report_txt = build_report_text(mtype, n_classes, acc, macro_f1, cm, info, y_test, y_pred,
                                   ksg_mi_bits=ksg_bits,
                                   show_clf_report=show_clf)
    saved = save_per_model_artifacts(base_out, mtype, cm, info, report_txt)
    # Build extended, time-stamped report and append to global-only file
    end_time = datetime.now()
    detailed_txt = build_detailed_run_report(
        start_time,
        end_time,
        cfg.get("data", {}),
        selected_cfg if isinstance(selected_cfg, dict) else {},
        mtype,
        n_classes,
        acc,
        macro_f1,
        cm,
        info,
        y_test,
        y_pred,
        show_clf_report=show_clf,
    )
    # Append to a global report aggregating all runs across models
    global_report_path = base_out / "report_detailed_global.txt"
    # Separate entries with divider lines for readability
    divider = "\n" + ("-" * 74) + "\n" + ("-" * 74) + "\n\n"
    with global_report_path.open("a", encoding="utf-8") as gf:
        gf.write(detailed_txt)
        gf.write(divider)
    print(report_txt)
    print(f"Saved: {saved['cm']}")
    print(f"Saved: {saved['info']}")
    print(f"Saved: {saved['report']}")
    print(f"Appended to global report: {global_report_path}")

    # Update simple accuracy summary across runs
    summary_path = _update_accuracy_summary(base_out, mtype, float(acc))
    print(f"[mi-race] Updated accuracy summary: {summary_path}")

    # No multi-model loop here; summary is updated incrementally per run.
