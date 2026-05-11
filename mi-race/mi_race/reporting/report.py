from __future__ import annotations

from pathlib import Path
from typing import Dict, List
from datetime import datetime
import json
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from mi_race.cli.ui import render_box, render_confusion_matrix


def ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _format_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _stringify_labels(labels: List) -> List[str]:
    """Coerce labels to plain str so numpy types don't leak into output."""
    return [str(l) for l in labels]


def build_screen_summary(
    model_name: str,
    n_classes: List,
    acc: float,
    macro_f1: float,
    cm: np.ndarray,
    info: dict,
) -> str:
    """A concise final-result box: headline metrics + confusion matrix.

    Verbose details (H values, KSG MI, classification report) are written to
    ``report.txt`` via :func:`build_report_text` but kept off-screen.
    """
    mi_bits = float(info.get("I", 0.0))
    nmi_sqrt = float(info.get("NMI_sqrt", 0.0))
    labels = _stringify_labels(n_classes)

    summary_lines = [
        f"Accuracy   {_format_pct(acc):<10}  Macro F1   {_format_pct(macro_f1)}",
        f"MI (cm)    {mi_bits:.4f} bits  NMI_sqrt   {nmi_sqrt:.4f}",
        "",
        "Confusion Matrix",
        *render_confusion_matrix(cm, labels, indent=0).splitlines(),
    ]
    return render_box(summary_lines, title=f"Result: {model_name} (test)", min_width=72)


def build_report_text(
    model_name: str,
    n_classes: list,
    acc: float,
    macro_f1: float,
    cm: np.ndarray,
    info: dict,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    ksg_mi_bits: float | None = None,
    show_clf_report: bool = True,
) -> str:
    """Verbose text used for the saved ``report.txt`` and global report."""
    labels = _stringify_labels(n_classes)
    parts = [
        f"\n=== Model: {model_name} ===\n",
        f"Classes: {labels}\n",
        f"Accuracy: {_format_pct(acc)}\n",
        f"Macro F1: {_format_pct(macro_f1)}\n\n",
        "Confusion Matrix (test)\n",
        render_confusion_matrix(cm, labels),
        "\n\n=== Mutual Information (from confusion matrix) ===\n",
        (
            f"H(true): {info['H_true']:.4f} bits  |  H(pred): {info['H_pred']:.4f} bits  |  Hjoint: {info['H_joint']:.4f} bits\n"
            f"I(true;pred): {info['I']:.4f} bits  |  NMI_sqrt: {info['NMI_sqrt']:.4f}  |  NMI_min: {info['NMI_min']:.4f}  |  NMI_max: {info['NMI_max']:.4f}\n"
            f"H(true|pred): {info['H_true_given_pred']:.4f} bits  |  H(pred|true): {info['H_pred_given_true']:.4f} bits\n"
        ),
    ]
    if ksg_mi_bits is not None:
        parts.append("\n=== KSG MI (labels;probs) ===\n")
        parts.append(f"KSG I(true;probs): {ksg_mi_bits:.4f} bits\n")
    if show_clf_report:
        parts.append("\n=== Classification Report (test) ===\n")
        parts.append(classification_report(y_test, y_pred, labels=n_classes, digits=4, zero_division=0))
        parts.append("\n")
    return "".join(parts)


def _format_start_time(ts: datetime) -> str:
    # Month abbreviation tweak: use 'Sept' instead of 'Sep'
    month = ts.strftime("%b")
    if month == "Sep":
        month = "Sept"
    day = ts.day
    year = ts.year
    time_part = ts.strftime("%I:%M%p").lstrip("0")
    time_part = time_part.replace("AM", "am").replace("PM", "pm")
    return f"{day} {month} {year}, {time_part}"


def build_detailed_run_report(start_time: datetime,
                              end_time: datetime,
                              data_cfg: dict,
                              model_cfg: dict,
                              model_name: str,
                              n_classes: list,
                              acc: float,
                              macro_f1: float,
                              cm: np.ndarray,
                              info: dict,
                              y_test: np.ndarray,
                              y_pred: np.ndarray,
                              show_clf_report: bool = True) -> str:
    """Build a time-stamped, config-inclusive report similar to the user's example.

    Includes:
      - Starting time (human readable)
      - Elapsed time in seconds
      - JSON-like snippets of data + model config (only relevant keys)
      - Standard metrics section (delegates to build_report_text for model metrics)
    """
    elapsed = int((end_time - start_time).total_seconds())
    # Select a subset of config keys to display to keep report concise
    data_subset_keys = ["path", "x_cols", "y_col", "sequence_mode", "balance"]
    model_subset_keys = [
        "sequence_col", "max_len", "pad_value", "standardize", "hidden_size",
        "num_layers", "bidirectional", "batch_size", "epochs", "lr"
    ]
    data_display = {k: data_cfg.get(k) for k in data_subset_keys if k in data_cfg}
    model_display = {k: model_cfg.get(k) for k in model_subset_keys if k in model_cfg}
    # JSON formatting similar to example (no trailing commas)
    data_json = json.dumps(data_display, indent=4)
    model_json = json.dumps(model_display, indent=4)

    header_lines = [
        f"Starting time: {_format_start_time(start_time)}",
        f"Elapsed time: {elapsed} second",
        "",
        f"\"data\": {data_json}",
        "",
        f"\"{model_name}\": {model_json}",
        "",
    ]
    metrics_block = build_report_text(
        model_name,
        n_classes,
        acc,
        macro_f1,
        cm,
        info,
        y_test,
        y_pred,
        show_clf_report=show_clf_report,
    )
    return "\n".join(header_lines) + metrics_block


def save_per_model_artifacts(base_out_dir: Path,
                             model_name: str,
                             cm: np.ndarray,
                             info: dict,
                             report: str) -> Dict[str, Path]:
    out_dir = base_out_dir / model_name
    ensure_outdir(out_dir)
    cm_path = out_dir / "confusion_matrix.csv"
    info_path = out_dir / "confusion_matrix_info.json"
    report_path = out_dir / "report.txt"
    pd.DataFrame(cm).to_csv(cm_path, header=False, index=False)
    with info_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    with report_path.open("w", encoding="utf-8") as f:
        f.write(report)
    return {"dir": out_dir, "cm": cm_path, "info": info_path, "report": report_path}
