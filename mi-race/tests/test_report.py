"""Tests for the experiment report generator (`mi-race report`)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mi_race.reporting.experiment_report import (
    _detect_groups,
    run_report,
)


def test_detect_groups_orders_by_index():
    cols = ["comp0_0", "comp0_2", "comp0_1", "comp1_0", "comp1_1", "symbol"]
    groups = _detect_groups([c for c in cols if c != "symbol"])
    assert set(groups) == {"comp0", "comp1"}
    assert groups["comp0"] == ["comp0_0", "comp0_1", "comp0_2"]  # sorted by idx


def _tiny_dataset(path: Path, n_steps: int = 12, per_class: int = 30, seed: int = 0):
    """Two well-separated symbols: symbol 0 rises early, symbol 1 rises late."""
    rng = np.random.default_rng(seed)
    rows = []
    for sid in (0, 1):
        onset = 1 if sid == 0 else n_steps // 2
        for _ in range(per_class):
            trace = np.zeros(n_steps)
            trace[onset:] = np.arange(n_steps - onset) + rng.normal(0, 0.3, n_steps - onset)
            row = {"symbol": sid}
            row.update({f"comp0_{t}": float(trace[t]) for t in range(n_steps)})
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


@pytest.fixture
def report_config(tmp_path: Path) -> Path:
    csv = tmp_path / "tiny.csv"
    _tiny_dataset(csv)
    cfg = {
        "channel": {
            "L": 4, "S": 1.0, "D": 2.0, "dt": 0.1, "T": 1.2,
            "observed_compartment": 0,
            "symbols": {"0": [[0.0, 50]], "1": [[0.6, 50]]},
        },
        "data": {
            "path": str(csv),
            "x_cols": "comp0_0:comp0_11",
            "y_col": "symbol",
            "sequence_mode": "split",
            "balance": False,
        },
        "model": {"rf": {"n_estimators": 10, "random_state": 0}},
        "train": {"test_size": 0.3, "random_state": 0, "standardize": False},
        "output": {"dir": str(tmp_path / "outputs")},
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))
    return cfg_path


def test_report_writes_html_and_bundle(report_config: Path, tmp_path: Path):
    out_dir = tmp_path / "exp"
    args = argparse.Namespace(
        config=str(report_config), model="rf", name=None, out=str(out_dir)
    )
    run_report(args)

    html = out_dir / "report.html"
    bundle = out_dir / "result.json"
    assert html.exists() and bundle.exists()

    text = html.read_text()
    assert "Confusion matrix" in text
    assert "Clean messages" in text          # codebook present → schedule figure
    assert "Averaged received" in text
    assert "data:image/png;base64," in text  # figures embedded

    payload = json.loads(bundle.read_text())
    res = payload["results"][0]
    assert res["label"] == "Baseline (hand-picked)"
    # Two well-separated symbols → RF should classify near-perfectly.
    assert res["accuracy"] > 0.8
    assert np.array(res["confusion"]).shape == (2, 2)


def test_report_unknown_model_errors(report_config: Path, tmp_path: Path):
    args = argparse.Namespace(
        config=str(report_config), model="not_a_model", name=None, out=str(tmp_path / "x")
    )
    with pytest.raises(SystemExit):
        run_report(args)
