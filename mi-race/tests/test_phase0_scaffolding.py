"""Phase 0 baseline tests: prove the harness works end-to-end."""

from __future__ import annotations

import pandas as pd


def test_pkg_version_returns_non_empty_string():
    from mi_race.cli.main import _pkg_version

    v = _pkg_version()
    assert isinstance(v, str)
    assert v.strip() != ""


def test_synthetic_csv_has_expected_columns(synthetic_csv):
    df = pd.read_csv(synthetic_csv)
    expected_features = {f"time_point_{i}" for i in range(1, 12)}
    assert {"mu", "sigma"}.issubset(df.columns)
    assert expected_features.issubset(df.columns)


def test_synthetic_df_is_balanced(synthetic_df):
    counts = synthetic_df.groupby(["mu", "sigma"]).size()
    assert counts.nunique() == 1, f"unbalanced cells: {counts.to_dict()}"


def test_synthetic_config_loads(synthetic_config):
    import json

    cfg = json.loads(synthetic_config.read_text())
    assert cfg["data"]["y_col"] == "mu"
    assert cfg["data"]["split_by"] == "sigma"
    assert cfg["model"]["mlp"]["epochs"] == 2
