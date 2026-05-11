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
from typing import Any
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

from ..analysis import info_from_confusion_matrix
from ..analysis.metrics import ksg_mi_labels_probs
from .data_prep import (
    load_df_from_cfg as dp_load_df_from_cfg,
    build_features_from_config,
    ensure_outdir as dp_ensure_outdir,
)
from datetime import datetime
from ..reporting.report import (
    build_report_text,
    build_screen_summary,
    save_per_model_artifacts,
    build_detailed_run_report,
)
from .orchestrator_utils import (
    _summarize_feature_columns,
    _update_accuracy_summary,
    _sanitize_summary_models,
    _update_split_accuracy_summary,
)
from .registry import MODEL_REGISTRY, SUPPORTED_MODELS, ModelSpec, resolve_epochs

from ..cli.ui import render_box, render_confusion_matrix
from ..data_preprocessing.filter_csv import apply_filter_expression, default_filtered_temp_path


def _normalize_labels_for_metrics(y: Any) -> np.ndarray:
    """Return 1D, discrete labels suitable for sklearn classification metrics.

    scikit-learn treats non-integer float targets as "continuous", which raises
    `ValueError: continuous is not supported` for classification metrics.
    """
    arr = np.asarray(y)
    if arr.ndim > 1:
        arr = arr.reshape(-1)

    if arr.dtype.kind == "f":
        finite = np.isfinite(arr)
        if finite.all() and np.allclose(arr, np.round(arr)):
            return np.round(arr).astype(int)
        return arr.astype(str)

    return arr


def _train_for_subset(
    spec: ModelSpec,
    df: pd.DataFrame,
    feature_df: "pd.DataFrame | None",
    y: np.ndarray,
    y_discrete: np.ndarray,
    train_cfg: dict,
    model_cfg: dict,
    standardize: bool,
    random_state: int,
    full_counts: pd.Series,
    mask: "pd.Series | None" = None,
):
    """Run one training pass (full data when ``mask`` is None, else the subset).

    Returns ``(y_test, y_pred, y_proba)`` or ``None`` if the subset is empty.
    ``y_proba`` is ``None`` for cnn/rnn. Per-split runs (``mask`` not None)
    pass ``quiet=True`` so the runner skips its per-call progress lines.
    """
    if mask is not None:
        idx = mask.to_numpy() if hasattr(mask, "to_numpy") else mask
        df_use = df.loc[mask].reset_index(drop=True)
        feature_df_use = (
            feature_df.loc[mask].reset_index(drop=True) if feature_df is not None else None
        )
        y_use = y[idx]
        y_disc_use = y_discrete[idx]
        if len(y_use) == 0 or (feature_df_use is not None and feature_df_use.empty):
            return None
        counts = pd.Series(y_use).value_counts().sort_index()
        quiet = True
    else:
        df_use, feature_df_use, y_use, y_disc_use = df, feature_df, y, y_discrete
        counts = full_counts
        quiet = False

    stratify = y_disc_use if train_cfg.get("stratify", True) else None
    return spec.adapter(
        df_use,
        feature_df_use,
        y_use,
        y_disc_use,
        train_cfg,
        model_cfg,
        standardize,
        random_state,
        stratify,
        counts,
        quiet=quiet,
    )


def _print_split_summary(
    base_out: Path,
    spec: ModelSpec,
    split_feature: str,
    split_value: Any,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    epochs: "int | None",
) -> None:
    """Print a compact per-split line followed by an indented confusion matrix."""
    y_t_m = _normalize_labels_for_metrics(y_test)
    y_p_m = _normalize_labels_for_metrics(y_pred)
    acc = accuracy_score(y_t_m, y_p_m)
    labels = sorted(pd.Series(y_t_m).dropna().unique().tolist())
    cm = confusion_matrix(y_t_m, y_p_m, labels=labels)
    info = info_from_confusion_matrix(cm, labels=labels)

    mi = info.get("I", float("nan"))
    nmi = info.get("NMI_sqrt", float("nan"))
    header = (
        f"  {split_feature}={split_value:<6}  acc={acc:.4f}  "
        f"MI={mi:.3f} bits  NMI_sqrt={nmi:.3f}"
    )
    if epochs is not None:
        header += f"  epochs={epochs}"
    print(header)
    print(render_confusion_matrix(cm, [str(l) for l in labels], indent=4))
    _update_split_accuracy_summary(base_out, spec.name, split_feature, split_value, float(acc))


def _compact_feature_summary(cols: list) -> str:
    """One-line feature summary for the Setup box.

    If every column belongs to a single ``prefix_<n>`` sequence group, show
    ``prefix_<lo>..prefix_<hi>  (<n> cols)``. Otherwise show ``<n> cols``.
    """
    import re as _re

    pat = _re.compile(r"^(?P<prefix>.+)_(?P<idx>\d+)$")
    groups: dict[str, list[int]] = {}
    for c in cols:
        m = pat.match(c)
        if m:
            groups.setdefault(m.group("prefix"), []).append(int(m.group("idx")))

    n = len(cols)
    if len(groups) == 1 and sum(len(v) for v in groups.values()) == n:
        prefix, idxs = next(iter(groups.items()))
        idxs.sort()
        return f"{prefix}_{idxs[0]}..{prefix}_{idxs[-1]}  ({n} cols)"
    return f"{n} cols"


def _export_rnn_features(
    df: pd.DataFrame,
    selected_cfg: dict,
    y_col: str,
    base_out: Path,
) -> None:
    """Export a fixed-length padded sequence matrix for inspection.

    Mirrors the previous RNN-branch preamble in run_cmd. Best-effort: errors are
    logged and swallowed.
    """
    seq_col = "time_trace"
    if isinstance(selected_cfg, dict):
        seq_col = selected_cfg.get("sequence_col", "time_trace") or "time_trace"

    print(
        f"[mi-race][rnn] Using raw sequence column '{seq_col}' from the original dataframe. "
        "Split feature columns (data.sequence_mode='split') are ignored for RNN."
    )

    if seq_col not in df.columns:
        return

    try:
        pad_value = float(selected_cfg.get("pad_value", 0.0)) if isinstance(selected_cfg, dict) else 0.0
        seq_series = df[seq_col].apply(
            lambda x: (
                np.asarray(x, dtype=float).reshape(-1)
                if isinstance(x, (list, tuple, np.ndarray))
                else np.array([], dtype=float)
            )
        )
        lengths = seq_series.apply(lambda a: a.size)
        max_len_all = int(lengths.max()) if lengths.size > 0 else 0

        export_len = None
        if isinstance(selected_cfg, dict):
            if selected_cfg.get("export_len") is not None:
                export_len = int(selected_cfg.get("export_len"))
            elif selected_cfg.get("max_len") is not None:
                export_len = int(selected_cfg.get("max_len"))
        if export_len is None:
            export_len = min(max_len_all, 1000)

        if export_len > 0:
            mat = np.full((len(seq_series), export_len), pad_value, dtype=float)
            for i, arr in enumerate(seq_series):
                m = min(arr.size, export_len)
                if m:
                    mat[i, :m] = arr[:m]
            cols = [f"{seq_col}_{i}" for i in range(export_len)]
            export_df = pd.DataFrame(mat, columns=cols)
            export_df.insert(0, "seq_len", lengths.to_numpy())
            if y_col in df.columns:
                export_df.insert(0, y_col, df[y_col].to_numpy())
            export_path = base_out / "processed_features_rnn.csv"
            export_df.to_csv(export_path, index=False)
            print(
                f"[mi-race][rnn] Saved processed RNN features to: {export_path} "
                f"(export_len={export_len})"
            )
    except Exception as e:
        print(f"[mi-race][rnn][WARN] Failed to export processed RNN features: {e}")



def run_cmd(args):
    start_time = datetime.now()
    # Load config
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise SystemExit(f"[mi-race] Config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Output directory (used for pre-run summary and saving artifacts)
    out_cfg = cfg.get("output", {})
    base_out = Path(out_cfg.get("dir", "outputs"))
    dp_ensure_outdir(base_out)
    _sanitize_summary_models(base_out)

    # Data section
    if "data" not in cfg:
        raise SystemExit("[mi-race] Missing 'data' section in config.")
    data_cfg = cfg["data"]

    # Load dataset
    df = dp_load_df_from_cfg(data_cfg)

    # Optional: one-shot config filter (applied before any balancing / feature building)
    filter_expr = None
    if isinstance(cfg, dict):
        filter_expr = cfg.get("filter")
    if not filter_expr and isinstance(data_cfg, dict):
        filter_expr = data_cfg.get("filter")

    filter_expr = str(filter_expr).strip() if filter_expr is not None else ""
    if filter_expr:
        before_n = len(df)
        df = apply_filter_expression(df, filter_expr)
        after_n = len(df)

        # Save filtered artifact into temp/
        source_stem = "data"
        raw_path = data_cfg.get("path") if isinstance(data_cfg, dict) else None
        if raw_path:
            p = Path(str(raw_path))
            if p.exists() and p.is_dir():
                # Our concat default is temp/<folder>_concat.csv
                source_stem = f"{p.name}_concat"
            elif p.exists() and p.is_file() and p.suffix.lower() == ".csv":
                source_stem = p.stem

        filtered_path = default_filtered_temp_path(source_stem, filter_expr)
        df.to_csv(filtered_path, index=False)
        print(f"[mi-race] Applied config filter: {filter_expr}")
        print(f"[mi-race] Filtered rows: {after_n:,} (from {before_n:,})")
        print(f"[mi-race] Wrote: {filtered_path}")

    # Target validation
    y_col = data_cfg.get("y_col")
    if not y_col:
        raise SystemExit("[mi-race] data.y_col must be specified.")
    if y_col not in df.columns:
        raise SystemExit(f"[mi-race] y_col '{y_col}' not found. Available: {df.columns.tolist()}")

    # Train settings (read once; used for balancing and training)
    train_cfg = cfg.get("train", {})
    random_state = int(train_cfg.get("random_state", 42))
    standardize = bool(train_cfg.get("standardize", True))

    # Optional: balance by min count of label if enabled in config.
    # Silent here — the result is summarised in the Setup box below.
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

    # Build arrays for general stats and counts (from the data actually used for training)
    y = df[y_col].to_numpy()
    full_counts = pd.Series(y).value_counts().sort_index()
    class_list = sorted(pd.Series(y).dropna().unique().tolist())

    # Use discrete labels for any sklearn-based classifiers (e.g., RF) and for stratification.
    y_for_training = _normalize_labels_for_metrics(y)

    # Which column to split per-feature (e.g., 'sigma' or 'noise') for per-slice runs
    split_feature = None
    if isinstance(data_cfg, dict):
        split_feature = data_cfg.get("split_by")
    if not split_feature:
        split_feature = train_cfg.get("split_by", "noise")

    # Model selection
    model_section = cfg.get("model", {})
    if not isinstance(model_section, dict):
        model_section = {}
    selection = getattr(args, "model", None)

    if selection is not None and selection not in SUPPORTED_MODELS:
        raise SystemExit(
            f"[mi-race] Unsupported --model value. Use one of: {', '.join(SUPPORTED_MODELS)}."
        )

    if selection in SUPPORTED_MODELS:
        mtype = selection
        selected_cfg = model_section.get(mtype, {})
    elif any(k in model_section for k in SUPPORTED_MODELS):
        mtype = next(k for k in SUPPORTED_MODELS if k in model_section)
        selected_cfg = model_section[mtype]
    else:
        # Legacy flat form: {"type": "mlp", "epochs": 10}
        mtype = str(model_section.get("type", "mlp")).lower()
        selected_cfg = model_section

    if mtype not in MODEL_REGISTRY:
        raise SystemExit(
            f"[mi-race] Unknown model type. Supported: {', '.join(repr(m) for m in SUPPORTED_MODELS)}."
        )
    spec = MODEL_REGISTRY[mtype]

    # Build features (mlp/cnn/rf); RNN reads raw sequences and skips this.
    feature_df = None
    feature_summary = "—"
    if spec.needs_features:
        feature_df, resolved_feature_cols = build_features_from_config(df, cfg)
        feature_summary = _compact_feature_summary(resolved_feature_cols)
        feature_df.to_csv(base_out / "processed_features.csv", index=False)

    # ---------------- Setup box (single consolidated panel) ----------------
    data_path_disp = data_cfg.get("path", "—")
    rows_disp = f"{len(df):,} rows × {len(df.columns)} cols"
    classes_disp = [str(c) for c in class_list]
    if enabled_balance and full_counts.nunique() == 1 and not full_counts.empty:
        label_line = (
            f"{y_col} — classes {classes_disp}  "
            f"(balanced, {int(full_counts.iloc[0]):,} each)"
        )
    elif enabled_balance:
        counts_disp = {str(k): int(v) for k, v in full_counts.items()}
        label_line = f"{y_col} — classes {classes_disp}  ({counts_disp})"
    else:
        label_line = f"{y_col} — classes {classes_disp}"

    setup_lines = [
        f"Model     {mtype}",
        f"Config    {args.config}",
        f"Dataset   {data_path_disp} — {rows_disp}",
        f"Features  {feature_summary}",
        f"Label     {label_line}",
        f"Output    {base_out}/",
    ]
    print(render_box(setup_lines, title="mi-race", min_width=76))

    # ---------------- Training ----------------
    print(f"\nTraining {mtype}")

    if spec.name == "rnn":
        _export_rnn_features(df, selected_cfg, y_col, base_out)

    feature_df_for_run = feature_df if spec.needs_features else None

    # Overall training run (chatty prints from the runner are allowed here).
    result = _train_for_subset(
        spec,
        df,
        feature_df_for_run,
        y,
        y_for_training,
        train_cfg,
        selected_cfg,
        standardize,
        random_state,
        full_counts,
        mask=None,
    )
    y_test, y_pred, y_proba = result

    # Per-split-feature runs (compact, no per-call chatter).
    if split_feature in df.columns:
        try:
            split_levels = sorted(pd.Series(df[split_feature]).dropna().unique().tolist())
        except Exception:
            split_levels = []
        epochs = resolve_epochs(spec, selected_cfg, train_cfg)
        if split_levels:
            print(f"\nPer-{split_feature} results:")
        for split_value in split_levels:
            mask = df[split_feature] == split_value
            sub = _train_for_subset(
                spec,
                df,
                feature_df_for_run,
                y,
                y_for_training,
                train_cfg,
                selected_cfg,
                standardize,
                random_state,
                full_counts,
                mask=mask,
            )
            if sub is None:
                print(f"  {split_feature}={split_value}  (skipped: no rows)")
                continue
            y_t_sub, y_p_sub, _ = sub
            _print_split_summary(
                base_out, spec, split_feature, split_value, y_t_sub, y_p_sub, epochs
            )

    # ---------------- Result ----------------
    y_test_m = _normalize_labels_for_metrics(y_test)
    y_pred_m = _normalize_labels_for_metrics(y_pred)

    acc = accuracy_score(y_test_m, y_pred_m)
    n_classes = sorted(pd.Series(_normalize_labels_for_metrics(y)).dropna().unique().tolist())
    cm = confusion_matrix(y_test_m, y_pred_m, labels=n_classes)
    info = info_from_confusion_matrix(cm, labels=n_classes)

    preds_unique = sorted(pd.Series(y_pred_m).dropna().unique().tolist())
    missing_predicted = [c for c in n_classes if c not in preds_unique]
    if missing_predicted:
        print(
            f"\n[WARN][{mtype}] No predicted samples for classes: "
            f"{[str(c) for c in missing_predicted]}  (metrics use zero_division=0)"
        )
    macro_f1 = f1_score(y_test_m, y_pred_m, average="macro", zero_division=0)

    # Optional KSG MI using predicted probabilities when available
    ksg_bits = None
    try:
        if y_proba is not None:
            k = int(cfg.get("output", {}).get("ksg_k", 5))
            ksg_bits = ksg_mi_labels_probs(y_test_m, y_proba, k=k)
    except Exception:
        ksg_bits = None

    print()
    print(build_screen_summary(mtype, n_classes, acc, macro_f1, cm, info))

    # Save verbose report + global report (off-screen).
    show_clf = cfg.get("output", {}).get("show_report", True)
    report_txt = build_report_text(
        mtype, n_classes, acc, macro_f1, cm, info,
        y_test_m, y_pred_m, ksg_mi_bits=ksg_bits, show_clf_report=show_clf,
    )
    saved = save_per_model_artifacts(base_out, mtype, cm, info, report_txt)

    end_time = datetime.now()
    detailed_txt = build_detailed_run_report(
        start_time, end_time, cfg.get("data", {}),
        selected_cfg if isinstance(selected_cfg, dict) else {},
        mtype, n_classes, acc, macro_f1, cm, info,
        y_test_m, y_pred_m, show_clf_report=show_clf,
    )
    global_report_path = base_out / "report_detailed_global.txt"
    divider = "\n" + ("-" * 74) + "\n" + ("-" * 74) + "\n\n"
    with global_report_path.open("a", encoding="utf-8") as gf:
        gf.write(detailed_txt)
        gf.write(divider)

    summary_path = _update_accuracy_summary(base_out, mtype, float(acc))

    # ---------------- Saved footer ----------------
    model_artifacts = ", ".join(p.name for p in (saved["cm"], saved["info"], saved["report"]))
    print()
    print(f"Saved  {saved['dir']}/{{{model_artifacts}}}")
    print(f"       {summary_path}  (updated)")
    print(f"       {global_report_path}  (appended)")
