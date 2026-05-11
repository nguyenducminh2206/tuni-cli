"""Phase 4.5 tests: output redesign.

Asserts the new compact layout:
- One consolidated Setup box (no separate Dataset/Labels/Features boxes)
- No ``=====`` divider lines
- No numpy type leaks (``np.str_(...)``)
- Per-split confusion matrices still printed
- Per-split runner calls are quiet (no per-call training chatter)
- Single Saved footer at the end
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _run_rf(synthetic_config, capsys, monkeypatch, tmp_path) -> str:
    monkeypatch.chdir(tmp_path)
    from mi_race.train.orchestrator import run_cmd

    args = argparse.Namespace(model="rf", config=str(synthetic_config))
    run_cmd(args)
    return capsys.readouterr().out


# ---------------------------------------------------------------------------
# Layout — no leftover dividers / numpy leaks
# ---------------------------------------------------------------------------


def test_no_equals_divider_lines(synthetic_config, capsys, monkeypatch, tmp_path):
    out = _run_rf(synthetic_config, capsys, monkeypatch, tmp_path)
    # The legacy ``===== Training =====`` / ``===== Result =====`` / per-split
    # ``=============== Training for X ===============`` are all gone.
    assert "=====" not in out
    assert "===============" not in out


def test_no_numpy_type_repr_in_output(synthetic_config, capsys, monkeypatch, tmp_path):
    out = _run_rf(synthetic_config, capsys, monkeypatch, tmp_path)
    assert "np.str_" not in out
    assert "np.float" not in out


def test_setup_panel_replaces_separate_boxes(synthetic_config, capsys, monkeypatch, tmp_path):
    out = _run_rf(synthetic_config, capsys, monkeypatch, tmp_path)
    # The previous design printed four titled boxes (mi-race, Dataset, Labels,
    # Features). We now show one ``mi-race`` Setup box plus the final Result box.
    assert out.count("─ mi-race ─") == 1 or out.count("── mi-race ──") == 1
    assert "─ Dataset ─" not in out
    assert "─ Labels ─" not in out
    assert "─ Features ─" not in out


def test_setup_box_lists_consolidated_fields(synthetic_config, capsys, monkeypatch, tmp_path):
    out = _run_rf(synthetic_config, capsys, monkeypatch, tmp_path)
    for needle in ("Model", "Config", "Dataset", "Features", "Label", "Output"):
        assert needle in out, f"setup box missing '{needle}'"


# ---------------------------------------------------------------------------
# Per-split blocks remain visible (user requirement)
# ---------------------------------------------------------------------------


def test_per_split_cms_still_printed(synthetic_config, capsys, monkeypatch, tmp_path):
    out = _run_rf(synthetic_config, capsys, monkeypatch, tmp_path)
    # Synthetic fixture has split_by=sigma with two values (0.5, 1.0).
    assert "sigma=0.5" in out
    assert "sigma=1.0" in out
    # Each split prints "acc=" and a CM. CM rows start with a label like "  1  " or "  2  ".
    assert out.count("acc=") >= 2


def test_per_split_runner_is_quiet(synthetic_config, capsys, monkeypatch, tmp_path):
    """Per-split adapter calls must pass quiet=True."""
    seen: list[bool] = []
    from mi_race.train import registry

    real_adapter = registry._adapter_rf

    def spy(*args, **kwargs):
        seen.append(bool(kwargs.get("quiet", False)))
        return real_adapter(*args, **kwargs)

    monkeypatch.setattr(registry, "_adapter_rf", spy)
    # Also patch the spec stored in the registry so dispatch hits the spy.
    new_spec = registry.ModelSpec(
        name=registry.MODEL_REGISTRY["rf"].name,
        runner_path=registry.MODEL_REGISTRY["rf"].runner_path,
        adapter=spy,
        needs_features=registry.MODEL_REGISTRY["rf"].needs_features,
        default_epochs=registry.MODEL_REGISTRY["rf"].default_epochs,
    )
    monkeypatch.setitem(registry.MODEL_REGISTRY, "rf", new_spec)

    _run_rf(synthetic_config, capsys, monkeypatch, tmp_path)

    # First call is the overall run (quiet=False); subsequent calls are
    # per-split (quiet=True).
    assert seen[0] is False, f"first call should be loud, got {seen}"
    assert all(q is True for q in seen[1:]), f"per-split calls must be quiet: {seen}"


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------


def test_saved_footer_appears_once(synthetic_config, capsys, monkeypatch, tmp_path):
    out = _run_rf(synthetic_config, capsys, monkeypatch, tmp_path)
    # Previously: "Saved: <path>" printed 3 times + "[mi-race] Updated accuracy"
    # + "Appended to global report" + duplicate "Saved processed features".
    # Now: one block starting with "Saved  " (two spaces) listing per-model
    # artifacts, then summary CSV, then global report — but only one literal
    # "Saved  " header.
    assert out.count("Saved  ") == 1, (
        f"expected single 'Saved  ' footer, got {out.count('Saved  ')}"
    )
    # And no separate "[mi-race] Updated accuracy summary:" tail line.
    assert "Updated accuracy summary" not in out


def test_no_duplicate_saved_processed_features(synthetic_config, capsys, monkeypatch, tmp_path):
    out = _run_rf(synthetic_config, capsys, monkeypatch, tmp_path)
    # The two prints "Saved processed features to: ..." and the in-box "Saved"
    # both used to appear. Now neither is on screen for the default flow.
    assert "Saved processed features" not in out


# ---------------------------------------------------------------------------
# UI helpers — unit tests
# ---------------------------------------------------------------------------


def test_render_confusion_matrix_format():
    from mi_race.cli.ui import render_confusion_matrix

    cm = np.array([[10, 2], [3, 15]])
    out = render_confusion_matrix(cm, labels=["a", "b"], indent=2)
    lines = out.splitlines()
    # Three lines: header + two data rows
    assert len(lines) == 3
    # Each data row starts with 2 spaces of indent and the row label
    assert lines[1].startswith("  a")
    assert lines[2].startswith("  b")
    # Counts present
    assert "10" in lines[1] and "2" in lines[1]


def test_render_confusion_matrix_no_indent_default():
    from mi_race.cli.ui import render_confusion_matrix

    out = render_confusion_matrix(np.array([[1]]), labels=["x"])
    assert out.splitlines()[1].startswith("x")


def test_compact_feature_summary_single_range():
    from mi_race.train.orchestrator import _compact_feature_summary

    cols = [f"time_point_{i}" for i in range(1, 12)]
    summary = _compact_feature_summary(cols)
    assert summary == "time_point_1..time_point_11  (11 cols)"


def test_compact_feature_summary_mixed_falls_back():
    from mi_race.train.orchestrator import _compact_feature_summary

    cols = ["age", "income", "time_point_1", "time_point_2"]
    summary = _compact_feature_summary(cols)
    # Mixed sequence + scalar columns → fall back to just count.
    assert summary == "4 cols"


def test_compact_feature_summary_empty():
    from mi_race.train.orchestrator import _compact_feature_summary

    assert _compact_feature_summary([]) == "0 cols"


def test_render_box_right_border_aligns_for_longest_line():
    """Regression: when a content line is longer than min_width-2, the right
    border used to shift by one (content padding not accounted for)."""
    from mi_race.cli.ui import render_box

    # Pick a long line whose len() is exactly min_width - 1.
    long_line = "x" * 75
    box = render_box(["short", long_line], title="t", min_width=76)
    widths = {len(line) for line in box.splitlines()}
    assert len(widths) == 1, (
        f"box lines have differing widths: {widths}\n{box}"
    )


def test_render_box_handles_line_exceeding_min_width():
    """Even when content exceeds min_width by a lot, all rows must align."""
    from mi_race.cli.ui import render_box

    box = render_box(["a", "b" * 100], title="t", min_width=10)
    widths = {len(line) for line in box.splitlines()}
    assert len(widths) == 1, f"box lines misaligned: {widths}"
