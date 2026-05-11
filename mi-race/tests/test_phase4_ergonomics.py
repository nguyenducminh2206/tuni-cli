"""Phase 4 tests: CLI ergonomics."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# --no-banner flag
# ---------------------------------------------------------------------------


def test_banner_prints_by_default(monkeypatch, capsys):
    monkeypatch.delenv("MI_RACE_NO_LOGO", raising=False)
    from mi_race.cli.main import _print_banner

    args = argparse.Namespace(no_banner=False, config="config.json")
    _print_banner(args)
    out = capsys.readouterr().out
    assert "███" in out, "expected ASCII banner in output"


def test_no_banner_flag_suppresses_banner(monkeypatch, capsys):
    monkeypatch.delenv("MI_RACE_NO_LOGO", raising=False)
    from mi_race.cli.main import _print_banner

    args = argparse.Namespace(no_banner=True, config="config.json")
    _print_banner(args)
    assert capsys.readouterr().out == ""


def test_env_var_still_suppresses_banner(monkeypatch, capsys):
    monkeypatch.setenv("MI_RACE_NO_LOGO", "1")
    from mi_race.cli.main import _print_banner

    args = argparse.Namespace(no_banner=False, config="config.json")
    _print_banner(args)
    assert capsys.readouterr().out == ""


def test_no_banner_flag_appears_in_parser():
    from mi_race.cli.main import _build_parser

    parser = _build_parser()
    actions = {tuple(a.option_strings) for a in parser._actions}
    assert any("--no-banner" in opts for opts in actions)


# ---------------------------------------------------------------------------
# $VISUAL / $EDITOR fallback
# ---------------------------------------------------------------------------


def test_open_editor_prefers_visual(monkeypatch):
    monkeypatch.setenv("VISUAL", "vim")
    monkeypatch.setenv("EDITOR", "emacs")
    from mi_race.cli.main import _open_editor

    with patch("mi_race.cli.main.subprocess.run") as mock_run:
        _open_editor("config.json")

    args, _ = mock_run.call_args
    cmd = args[0]
    assert cmd[0] == "vim"
    assert cmd[-1].endswith("config.json")


def test_open_editor_falls_back_to_editor(monkeypatch):
    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.setenv("EDITOR", "emacs")
    from mi_race.cli.main import _open_editor

    with patch("mi_race.cli.main.subprocess.run") as mock_run:
        _open_editor("config.json")

    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "emacs"


def test_open_editor_splits_editor_with_args(monkeypatch):
    """``EDITOR='code -w'`` should split into ['code', '-w', path]."""
    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.setenv("EDITOR", "code -w")
    from mi_race.cli.main import _open_editor

    with patch("mi_race.cli.main.subprocess.run") as mock_run:
        _open_editor("config.json")

    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "code"
    assert cmd[1] == "-w"


def test_open_editor_falls_back_to_nano_on_posix(monkeypatch):
    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.delenv("EDITOR", raising=False)
    monkeypatch.setattr(os, "name", "posix")
    from mi_race.cli.main import _open_editor

    with patch("mi_race.cli.main.subprocess.run") as mock_run:
        _open_editor("config.json")

    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "nano"


# ---------------------------------------------------------------------------
# run-all fail-soft + summary
# ---------------------------------------------------------------------------


def test_run_all_continues_after_failure(monkeypatch, capsys):
    """If cnn raises, mlp/rnn/rf still run and a summary is printed."""
    from mi_race.cli import main as main_mod

    calls: list[str] = []

    def fake_run_cmd(args):
        calls.append(args.model)
        if args.model == "cnn":
            raise RuntimeError("simulated cnn failure")

    monkeypatch.setattr(main_mod, "run_cmd", fake_run_cmd)

    ns = argparse.Namespace(config="dummy.json", cmd="run-all")
    main_mod._run_all(ns)

    assert calls == list(main_mod.SUPPORTED_MODELS)
    out = capsys.readouterr().out
    assert "run-all summary" in out
    assert "PASS" in out
    assert "FAIL" in out
    assert "simulated cnn failure" in out


def test_run_all_handles_systemexit(monkeypatch, capsys):
    """Config errors that raise SystemExit don't abort run-all."""
    from mi_race.cli import main as main_mod

    def fake_run_cmd(args):
        if args.model == "rf":
            raise SystemExit("[mi-race] bad config")

    monkeypatch.setattr(main_mod, "run_cmd", fake_run_cmd)

    main_mod._run_all(argparse.Namespace(config="dummy.json"))
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert "bad config" in out


def test_run_all_summary_lists_all_models(monkeypatch, capsys):
    from mi_race.cli import main as main_mod

    monkeypatch.setattr(main_mod, "run_cmd", lambda args: None)
    main_mod._run_all(argparse.Namespace(config="dummy.json"))

    out = capsys.readouterr().out
    for m in main_mod.SUPPORTED_MODELS:
        assert f"  {m}" in out, f"missing {m!r} in summary: {out}"


# ---------------------------------------------------------------------------
# Flatten concat csv-files -> concat
# ---------------------------------------------------------------------------


def test_concat_flat_form_parses():
    from mi_race.cli.main import _build_parser

    parser = _build_parser()
    args = parser.parse_args(["concat", "-c", "config.json"])
    assert args.cmd == "concat"
    assert args.config == "config.json"


def test_concat_subparser_has_no_subcommand_action():
    """`concat csv-files` is no longer a recognised subcommand of `concat`."""
    from mi_race.cli.main import _build_parser

    parser = _build_parser()
    subparsers = next(
        a for a in parser._actions if isinstance(a, argparse._SubParsersAction)
    )
    p_concat = subparsers.choices["concat"]
    # No nested subparsers anymore
    nested = [a for a in p_concat._actions if isinstance(a, argparse._SubParsersAction)]
    assert nested == []


def test_legacy_concat_csv_files_is_rewritten():
    from mi_race.cli.main import _rewrite_legacy_argv

    out = _rewrite_legacy_argv(["concat", "csv-files", "-c", "x.json"])
    assert out == ["concat", "-c", "x.json"]


def test_legacy_rewrite_is_noop_for_modern_form():
    from mi_race.cli.main import _rewrite_legacy_argv

    out = _rewrite_legacy_argv(["concat", "-c", "x.json"])
    assert out == ["concat", "-c", "x.json"]


def test_legacy_rewrite_prints_warning(capsys):
    from mi_race.cli.main import _rewrite_legacy_argv

    _rewrite_legacy_argv(["concat", "csv-files"])
    err = capsys.readouterr().err
    assert "deprecated" in err.lower()


def test_legacy_form_round_trips_through_main(monkeypatch, synthetic_config):
    """Calling ``main(['concat', 'csv-files', '-c', cfg])`` invokes the concat handler."""
    from mi_race.cli import main as main_mod

    called: dict = {}

    def fake_concat(args):
        called["config"] = args.config

    monkeypatch.setattr(main_mod, "run_concat_csv_files", fake_concat)

    # Rebuild parser so it picks up the monkeypatched handler.
    monkeypatch.delenv("MI_RACE_NO_LOGO", raising=False)
    main_mod.main(["--no-banner", "concat", "csv-files", "-c", str(synthetic_config)])
    assert called.get("config") == str(synthetic_config)


def test_modern_concat_invocation_calls_handler(monkeypatch, synthetic_config):
    from mi_race.cli import main as main_mod

    called: dict = {}
    monkeypatch.setattr(main_mod, "run_concat_csv_files", lambda args: called.setdefault("ok", True))

    main_mod.main(["--no-banner", "concat", "-c", str(synthetic_config)])
    assert called.get("ok") is True
