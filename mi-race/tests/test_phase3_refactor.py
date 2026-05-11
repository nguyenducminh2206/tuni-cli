"""Phase 3 tests: per-split loop refactor.

The four near-identical per-model blocks in ``run_cmd`` have been collapsed
into a single helper ``_train_for_subset`` + ``_print_split_summary``. These
tests verify:

1. The new helpers exist and behave correctly in isolation.
2. End-to-end RF output is byte-identical to the pre-refactor baseline
   captured in ``tests/golden/`` (RF is deterministic with a fixed seed).
3. Each model's adapter has the uniform signature and returns the expected
   3-tuple shape.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


GOLDEN_DIR = Path(__file__).parent / "golden"


def test_train_for_subset_helper_exists():
    from mi_race.train.orchestrator import _train_for_subset, _print_split_summary

    assert callable(_train_for_subset)
    assert callable(_print_split_summary)


def test_adapters_share_uniform_signature():
    """Every registered adapter accepts the same 10 positional + quiet kwarg."""
    import inspect
    from mi_race.train.registry import MODEL_REGISTRY

    for name, spec in MODEL_REGISTRY.items():
        sig = inspect.signature(spec.adapter)
        # 10 positional args + 1 'quiet' kwarg = 11 total
        assert len(sig.parameters) == 11, (
            f"{name} adapter has {len(sig.parameters)} params, expected 11"
        )
        assert "quiet" in sig.parameters, f"{name} adapter missing 'quiet' param"


def test_train_for_subset_returns_none_on_empty_mask(synthetic_df):
    """Empty subset short-circuits with None instead of crashing the runner."""
    from mi_race.train.orchestrator import _train_for_subset
    from mi_race.train.registry import MODEL_REGISTRY

    spec = MODEL_REGISTRY["rf"]
    df = synthetic_df.copy()
    feature_cols = [c for c in df.columns if c.startswith("time_point_")]
    feature_df = df[feature_cols].reset_index(drop=True)
    y = df["mu"].to_numpy()
    y_disc = y.astype(int)

    # Mask that selects no rows.
    empty_mask = df["mu"] == -999

    result = _train_for_subset(
        spec,
        df,
        feature_df,
        y,
        y_disc,
        train_cfg={"test_size": 0.2, "random_state": 42},
        model_cfg={"n_estimators": 5, "random_state": 42},
        standardize=True,
        random_state=42,
        full_counts=pd.Series(y).value_counts().sort_index(),
        mask=empty_mask,
    )
    assert result is None


def test_resolve_epochs_returns_none_for_rf():
    from mi_race.train.registry import MODEL_REGISTRY, resolve_epochs

    rf_spec = MODEL_REGISTRY["rf"]
    assert resolve_epochs(rf_spec, {}, {}) is None


def test_resolve_epochs_uses_model_cfg_first():
    from mi_race.train.registry import MODEL_REGISTRY, resolve_epochs

    mlp_spec = MODEL_REGISTRY["mlp"]
    assert resolve_epochs(mlp_spec, {"epochs": 7}, {"epochs": 99}) == 7
    assert resolve_epochs(mlp_spec, {}, {"epochs": 99}) == 99
    assert resolve_epochs(mlp_spec, {}, {}) == mlp_spec.default_epochs


def test_rf_confusion_matrix_matches_golden(synthetic_config, monkeypatch, tmp_path):
    """Byte-identical CM after refactor (RF is deterministic with random_state=42)."""
    monkeypatch.chdir(tmp_path)
    from mi_race.train.orchestrator import run_cmd

    args = argparse.Namespace(model="rf", config=str(synthetic_config))
    run_cmd(args)

    cfg = json.loads(Path(synthetic_config).read_text())
    out_dir = Path(cfg["output"]["dir"])
    actual = (out_dir / "rf" / "confusion_matrix.csv").read_text()
    expected = (GOLDEN_DIR / "rf_confusion_matrix.csv").read_text()
    assert actual == expected, (
        f"RF confusion matrix changed after refactor.\n"
        f"actual:\n{actual}\nexpected:\n{expected}"
    )


def test_rf_summary_models_matches_golden(synthetic_config, monkeypatch, tmp_path):
    """Per-split accuracies + overall accuracy unchanged after refactor."""
    monkeypatch.chdir(tmp_path)
    from mi_race.train.orchestrator import run_cmd

    args = argparse.Namespace(model="rf", config=str(synthetic_config))
    run_cmd(args)

    cfg = json.loads(Path(synthetic_config).read_text())
    out_dir = Path(cfg["output"]["dir"])
    actual_df = pd.read_csv(out_dir / "summary_models.csv")
    expected_df = pd.read_csv(GOLDEN_DIR / "rf_summary_models.csv")

    # Column set + the rf row contents must match
    assert set(actual_df.columns) == set(expected_df.columns)
    a_rf = actual_df[actual_df["model"] == "rf"].iloc[0]
    e_rf = expected_df[expected_df["model"] == "rf"].iloc[0]
    for col in expected_df.columns:
        if col == "model":
            assert a_rf[col] == e_rf[col]
        else:
            assert float(a_rf[col]) == pytest.approx(float(e_rf[col]), abs=1e-12), (
                f"column {col!r} drifted: {a_rf[col]} vs {e_rf[col]}"
            )


def test_rf_mi_metrics_match_golden(synthetic_config, monkeypatch, tmp_path):
    """Mutual information metrics are bit-identical (RF is deterministic)."""
    monkeypatch.chdir(tmp_path)
    from mi_race.train.orchestrator import run_cmd

    args = argparse.Namespace(model="rf", config=str(synthetic_config))
    run_cmd(args)

    cfg = json.loads(Path(synthetic_config).read_text())
    out_dir = Path(cfg["output"]["dir"])
    actual = json.loads((out_dir / "rf" / "confusion_matrix_info.json").read_text())
    expected = json.loads((GOLDEN_DIR / "rf_confusion_matrix_info.json").read_text())

    # Compare every numeric field within float tolerance.
    def _walk(a, b, path=""):
        if isinstance(a, dict):
            assert set(a.keys()) == set(b.keys()), f"keys differ at {path}"
            for k in a:
                _walk(a[k], b[k], f"{path}.{k}")
        elif isinstance(a, list):
            assert len(a) == len(b), f"length differs at {path}"
            for i, (ai, bi) in enumerate(zip(a, b)):
                _walk(ai, bi, f"{path}[{i}]")
        elif isinstance(a, float):
            assert a == pytest.approx(b, abs=1e-12), f"{path}: {a} != {b}"
        else:
            assert a == b, f"{path}: {a} != {b}"

    _walk(actual, expected)


def test_rf_run_is_deterministic(synthetic_config, monkeypatch, tmp_path):
    """Two independent rf runs with the same seed must produce identical CMs."""
    from mi_race.train.orchestrator import run_cmd

    def _run(workdir: Path) -> str:
        workdir.mkdir(parents=True, exist_ok=True)
        monkeypatch.chdir(workdir)
        cfg = json.loads(Path(synthetic_config).read_text())
        # Redirect output to this workdir so successive runs don't collide.
        cfg["output"]["dir"] = str(workdir / "outputs")
        cfg_path = workdir / "config.json"
        cfg_path.write_text(json.dumps(cfg))

        args = argparse.Namespace(model="rf", config=str(cfg_path))
        run_cmd(args)
        return (workdir / "outputs" / "rf" / "confusion_matrix.csv").read_text()

    a = _run(tmp_path / "run_a")
    b = _run(tmp_path / "run_b")
    assert a == b


def test_mlp_smoke_run_completes(synthetic_config, monkeypatch, tmp_path):
    """MLP isn't deterministic across runs (no torch seed) — just smoke it."""
    monkeypatch.chdir(tmp_path)
    from mi_race.train.orchestrator import run_cmd

    args = argparse.Namespace(model="mlp", config=str(synthetic_config))
    run_cmd(args)

    cfg = json.loads(Path(synthetic_config).read_text())
    out_dir = Path(cfg["output"]["dir"])
    cm = pd.read_csv(out_dir / "mlp" / "confusion_matrix.csv", header=None)
    # 2 classes in fixture; CM must be 2x2 with non-negative integer counts.
    assert cm.shape == (2, 2)
    assert (cm.to_numpy() >= 0).all()
