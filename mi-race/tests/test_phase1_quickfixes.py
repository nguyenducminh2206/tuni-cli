"""Phase 1 tests: quick correctness fixes.

Asserts that:
- The duplicate box renderer (_build_box) is gone.
- The argparse parser is now extractable and lists all four models.
- A run with --model rf still produces the expected artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def test_build_box_removed_from_main():
    """Phase 1: _build_box was deleted; only render_box remains."""
    from mi_race.cli import main as main_mod

    assert not hasattr(main_mod, "_build_box"), (
        "Expected _build_box to be removed from cli.main "
        "(use render_box from cli.ui)"
    )


def test_render_box_is_the_only_renderer():
    """render_box lives in cli.ui and is imported by cli.main."""
    from mi_race.cli import main as main_mod
    from mi_race.cli import ui as ui_mod

    assert hasattr(ui_mod, "render_box")
    assert getattr(main_mod, "render_box", None) is ui_mod.render_box


def test_build_parser_is_extracted():
    """The parser construction is now a standalone function for testability."""
    from mi_race.cli.main import _build_parser

    parser = _build_parser()
    assert isinstance(parser, argparse.ArgumentParser)


def test_help_lists_all_four_models():
    """The 'run' subcommand's --model choices include rf."""
    from mi_race.cli.main import _build_parser

    parser = _build_parser()
    subparsers_action = next(
        a for a in parser._actions if isinstance(a, argparse._SubParsersAction)
    )
    run = subparsers_action.choices["run"]
    model_action = next(a for a in run._actions if "--model" in a.option_strings)
    assert set(model_action.choices) == {"mlp", "cnn", "rnn", "rf"}


def test_unsupported_model_error_mentions_rf(synthetic_config):
    """Error message for an unsupported model lists all four supported names."""
    from mi_race.train.orchestrator import run_cmd

    args = argparse.Namespace(model="xgboost", config=str(synthetic_config))
    try:
        run_cmd(args)
    except SystemExit as e:
        msg = str(e)
        for name in ("mlp", "cnn", "rnn", "rf"):
            assert name in msg, f"missing '{name}' in error: {msg!r}"
    else:
        raise AssertionError("expected SystemExit for unsupported model")


def test_rf_run_produces_artifacts(synthetic_config, monkeypatch, tmp_path):
    """End-to-end RF run on synthetic data writes confusion matrix + summary."""
    monkeypatch.chdir(tmp_path)
    from mi_race.train.orchestrator import run_cmd

    args = argparse.Namespace(model="rf", config=str(synthetic_config))
    run_cmd(args)

    cfg = json.loads(Path(synthetic_config).read_text())
    out_dir = Path(cfg["output"]["dir"])
    assert (out_dir / "rf" / "confusion_matrix.csv").exists()
    assert (out_dir / "rf" / "confusion_matrix_info.json").exists()
    assert (out_dir / "rf" / "report.txt").exists()
    assert (out_dir / "summary_models.csv").exists()
