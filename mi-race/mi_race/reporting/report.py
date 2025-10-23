from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import json
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report


def ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def cm_table(cm: np.ndarray, labels: List) -> str:
    """Create a clean confusion matrix table without internal vertical bars."""
    header_labels = [str(label) for label in labels]
    header = "     " + "".join(f"{label:>6}" for label in header_labels)
    separator = "    " + "-" * (6 * len(labels))
    rows = []
    for i, label in enumerate(labels):
        row_values = "".join(f"{int(cm[i][j]):>6}" for j in range(len(labels)))
        rows.append(f"{str(label):>3} |{row_values}")
    return "\n".join([header, separator] + rows)


def build_report_text(model_name: str,
                      n_classes: list,
                      acc: float,
                      macro_f1: float,
                      cm: np.ndarray,
                      info: dict,
                      y_test: np.ndarray,
                      y_pred: np.ndarray,
                      show_clf_report: bool = True) -> str:
    parts = []
    parts.append(f"\n=== Model: {model_name} ===\n")
    parts.append(f"Classes: {n_classes}\n")
    parts.append("=== Metrics (test) ===\n")
    parts.append(f"Accuracy: {acc*100:.2f}%\n")
    parts.append(f"Macro F1: {macro_f1*100:.2f}%\n")
    parts.append("\n=== Confusion Matrix (test) ===\n")
    parts.append(cm_table(cm, n_classes) + "\n")
    parts.append("\n=== Information (from confusion matrix) ===\n")
    parts.append(
        (
            f"H(true): {info['H_true']:.4f} bits  |  H(pred): {info['H_pred']:.4f} bits  |  Hjoint: {info['H_joint']:.4f} bits\n"
            f"I(true;pred): {info['I']:.4f} bits  |  NMI_sqrt: {info['NMI_sqrt']:.4f}  |  NMI_min: {info['NMI_min']:.4f}  |  NMI_max: {info['NMI_max']:.4f}\n"
            f"H(true|pred): {info['H_true_given_pred']:.4f} bits  |  H(pred|true): {info['H_pred_given_true']:.4f} bits\n"
        )
    )
    if show_clf_report:
        parts.append("\n=== Classification Report (test) ===\n")
        parts.append(classification_report(y_test, y_pred, labels=n_classes, digits=4, zero_division=0))
        parts.append("\n")
    return "".join(parts)


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
