"""Phase 5 tests: --dry-run flag on `run` and `run-all`."""

from __future__ import annotations

import argparse
from pathlib import Path

from mi_race.cli.main import _build_parser
from mi_race.train.orchestrator import run_cmd


def test_dry_run_flag_appears_on_run():
    parser = _build_parser()
    args = parser.parse_args(["run", "--model", "rf", "--dry-run", "-c", "x.json"])
    assert args.dry_run is True


def test_dry_run_flag_appears_on_run_all():
    parser = _build_parser()
    args = parser.parse_args(["run-all", "--dry-run", "-c", "x.json"])
    assert args.dry_run is True


def test_dry_run_default_is_false():
    parser = _build_parser()
    args = parser.parse_args(["run", "--model", "rf", "-c", "x.json"])
    assert args.dry_run is False


def test_dry_run_exits_cleanly(synthetic_config, capsys):
    """run_cmd returns normally under --dry-run (no SystemExit, no exception)."""
    args = argparse.Namespace(
        config=str(synthetic_config), model="rf", dry_run=True
    )
    run_cmd(args)
    out = capsys.readouterr().out
    assert "--dry-run" in out
    assert "No training run" in out


def test_dry_run_writes_no_artifacts(synthetic_config, tmp_path):
    """No files appear under the configured output directory after a dry run."""
    import json as _json

    with open(synthetic_config) as f:
        cfg = _json.load(f)
    out_dir = Path(cfg["output"]["dir"])
    assert not out_dir.exists() or not any(out_dir.iterdir()), (
        "test setup: output dir must be empty before dry-run"
    )

    args = argparse.Namespace(
        config=str(synthetic_config), model="rf", dry_run=True
    )
    run_cmd(args)

    if out_dir.exists():
        # The dry-run path must not create processed_features.csv,
        # summary_models.csv, or any per-model subdir.
        leftovers = sorted(p.name for p in out_dir.iterdir())
        assert leftovers == [], f"--dry-run leaked artifacts: {leftovers}"


def test_dry_run_does_not_train(synthetic_config, monkeypatch):
    """The training dispatcher must not be called under --dry-run."""
    from mi_race.train import orchestrator as orch_mod

    def tripwire(*a, **kw):
        raise AssertionError("_train_for_subset should not be called under --dry-run")

    monkeypatch.setattr(orch_mod, "_train_for_subset", tripwire)

    args = argparse.Namespace(
        config=str(synthetic_config), model="rf", dry_run=True
    )
    run_cmd(args)
