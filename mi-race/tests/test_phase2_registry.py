"""Phase 2 tests: model registry as single source of truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest


def test_registry_exposes_all_four_models():
    from mi_race.train.registry import MODEL_REGISTRY, SUPPORTED_MODELS

    assert set(MODEL_REGISTRY.keys()) == {"mlp", "cnn", "rnn", "rf"}
    assert set(SUPPORTED_MODELS) == {"mlp", "cnn", "rnn", "rf"}


def test_supported_models_order_is_stable():
    """Iteration order matters: default-pick uses next(k for k in SUPPORTED_MODELS ...)."""
    from mi_race.train.registry import SUPPORTED_MODELS

    assert SUPPORTED_MODELS == ("mlp", "cnn", "rnn", "rf")


def test_model_specs_have_required_fields():
    from mi_race.train.registry import MODEL_REGISTRY

    for name, spec in MODEL_REGISTRY.items():
        assert spec.name == name
        assert isinstance(spec.runner_path, str) and ":" in spec.runner_path
        assert isinstance(spec.needs_features, bool)


def test_needs_features_matches_expected():
    """mlp/cnn/rf use feature_df; rnn reads raw sequences."""
    from mi_race.train.registry import MODEL_REGISTRY

    assert MODEL_REGISTRY["mlp"].needs_features is True
    assert MODEL_REGISTRY["cnn"].needs_features is True
    assert MODEL_REGISTRY["rf"].needs_features is True
    assert MODEL_REGISTRY["rnn"].needs_features is False


def test_get_runner_returns_actual_callable():
    from mi_race.train.registry import get_runner
    from mi_race.train.models.random_forest import run_random_forest

    assert get_runner("rf") is run_random_forest


def test_get_runner_raises_on_unknown_model():
    from mi_race.train.registry import get_runner

    with pytest.raises(KeyError):
        get_runner("xgboost")


def test_argparse_choices_match_registry():
    from mi_race.cli.main import _build_parser
    from mi_race.train.registry import SUPPORTED_MODELS

    parser = _build_parser()
    subparsers_action = next(
        a for a in parser._actions if isinstance(a, argparse._SubParsersAction)
    )
    run = subparsers_action.choices["run"]
    model_action = next(a for a in run._actions if "--model" in a.option_strings)
    assert tuple(model_action.choices) == SUPPORTED_MODELS


def test_cli_and_orchestrator_share_one_supported_models():
    """Both modules import SUPPORTED_MODELS from the registry — same object."""
    from mi_race.cli import main as main_mod
    from mi_race.train import orchestrator as orch_mod
    from mi_race.train.registry import SUPPORTED_MODELS

    assert main_mod.SUPPORTED_MODELS is SUPPORTED_MODELS
    assert orch_mod.SUPPORTED_MODELS is SUPPORTED_MODELS


def test_rf_run_still_works_after_registry_wire(synthetic_config, monkeypatch, tmp_path):
    """Regression check: registry-based needs_features still routes RF correctly."""
    monkeypatch.chdir(tmp_path)
    from mi_race.train.orchestrator import run_cmd

    args = argparse.Namespace(model="rf", config=str(synthetic_config))
    run_cmd(args)

    cfg = json.loads(Path(synthetic_config).read_text())
    out_dir = Path(cfg["output"]["dir"])
    assert (out_dir / "rf" / "confusion_matrix.csv").exists()
    assert (out_dir / "summary_models.csv").exists()
