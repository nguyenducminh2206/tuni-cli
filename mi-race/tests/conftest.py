"""Shared pytest fixtures for the mi-race test suite.

The synthetic dataset mirrors the shape of the real `OU` data:
- target column `mu` (classification)
- split column `sigma`
- feature columns `time_point_1`..`time_point_11`
- enough rows per (mu, sigma) cell to support stratified train/test splits
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


N_TIMEPOINTS = 11
MU_VALUES = (1.0, 2.0)
SIGMA_VALUES = (0.5, 1.0)
ROWS_PER_CELL = 60


def _build_synthetic_df(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for mu in MU_VALUES:
        for sigma in SIGMA_VALUES:
            for _ in range(ROWS_PER_CELL):
                trace = mu + sigma * rng.standard_normal(N_TIMEPOINTS)
                row: dict = {"mu": mu, "sigma": sigma}
                for i, v in enumerate(trace, start=1):
                    row[f"time_point_{i}"] = float(v)
                rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    return _build_synthetic_df()


@pytest.fixture(scope="session")
def synthetic_csv(tmp_path_factory: pytest.TempPathFactory, synthetic_df: pd.DataFrame) -> Path:
    path = tmp_path_factory.mktemp("data") / "synthetic.csv"
    synthetic_df.to_csv(path, index=False)
    return path


@pytest.fixture
def synthetic_config(tmp_path: Path, synthetic_csv: Path) -> Path:
    """Per-test config pointing at the synthetic CSV with a tiny model."""
    out_dir = tmp_path / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "data": {
            "path": str(synthetic_csv),
            "x_cols": "time_point_1:time_point_11",
            "y_col": "mu",
            "sequence_mode": "ignore",
            "balance": True,
            "split_by": "sigma",
        },
        "model": {
            "mlp": {
                "type": "mlp",
                "hidden_layers": [16],
                "activation": "relu",
                "optimizer": "adam",
                "lr": 0.01,
                "dropout": 0.0,
                "batch_size": 32,
                "epochs": 2,
            },
            "rf": {"n_estimators": 5, "random_state": 42},
        },
        "train": {"test_size": 0.2, "random_state": 42, "standardize": True},
        "output": {"dir": str(out_dir), "show_report": False},
    }
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg))
    return cfg_path
