"""Phase 8 tests: tqdm progress bars and auto-disable."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mi_race.cli.ui import progress_disabled


# ---------------------------------------------------------------------------
# progress_disabled() helper
# ---------------------------------------------------------------------------


def test_progress_disabled_when_stdout_not_tty(monkeypatch):
    class _FakeStdout:
        def isatty(self):
            return False

    monkeypatch.setattr("sys.stdout", _FakeStdout())
    assert progress_disabled() is True


def test_progress_enabled_when_stdout_is_tty(monkeypatch):
    class _FakeStdout:
        def isatty(self):
            return True

    monkeypatch.setattr("sys.stdout", _FakeStdout())
    assert progress_disabled() is False


# ---------------------------------------------------------------------------
# concat.py wraps the read loop in tqdm
# ---------------------------------------------------------------------------


def test_concat_uses_tqdm(monkeypatch, tmp_path):
    """concat path with >5 files calls tqdm; ≤5 files doesn't."""
    from mi_race.data_preprocessing import concat as concat_mod

    calls: list[dict] = []

    class _FakeTqdm:
        def __init__(self, iterable, **kw):
            calls.append(kw)
            self._it = iterable

        def __iter__(self):
            return iter(self._it)

    fake_module = type("M", (), {"tqdm": _FakeTqdm})()
    monkeypatch.setattr("tqdm.auto.tqdm", _FakeTqdm)

    # 6 identical CSVs trip the >5 threshold.
    for i in range(6):
        pd.DataFrame({"a": [i], "b": [i * 2]}).to_csv(tmp_path / f"part_{i}.csv", index=False)

    df = concat_mod._concat_csvs_in_folder(tmp_path, pattern="*.csv", recursive=False)
    assert len(df) == 6
    assert len(calls) == 1, "tqdm should be called exactly once for the per-file loop"
    assert calls[0].get("unit") == "file"


def test_concat_skips_tqdm_for_few_files(monkeypatch, tmp_path):
    from mi_race.data_preprocessing import concat as concat_mod

    calls: list[dict] = []

    class _FakeTqdm:
        def __init__(self, iterable, **kw):
            calls.append(kw)
            self._it = iterable

        def __iter__(self):
            return iter(self._it)

    monkeypatch.setattr("tqdm.auto.tqdm", _FakeTqdm)

    for i in range(3):
        pd.DataFrame({"a": [i]}).to_csv(tmp_path / f"part_{i}.csv", index=False)

    concat_mod._concat_csvs_in_folder(tmp_path, pattern="*.csv", recursive=False)
    assert calls == [], "tqdm should be skipped when file count is small"


# ---------------------------------------------------------------------------
# generate-data wraps the SSA loop in tqdm
# ---------------------------------------------------------------------------


def test_generate_data_uses_tqdm(monkeypatch, tmp_path):
    from mi_race.encoder import dataset_gen as dg_mod

    calls: list[dict] = []

    class _FakeTqdm:
        def __init__(self, *, total=None, **kw):
            calls.append({"total": total, **kw})
            self.n = 0

        def update(self, k):
            self.n += k

        def close(self):
            pass

    monkeypatch.setattr("tqdm.auto.tqdm", _FakeTqdm)

    cfg = {
        "channel": {
            "L": 4, "S": 1.0, "D": 1.0, "dt": 0.5, "T": 1.0,
            "observed_compartment": 0, "runs_per_symbol": 2, "seed": 0,
            "make_plots": False,
            "symbols": {"0": [[0.0, 10]], "1": [[0.5, 10]]},
        }
    }
    out_csv = tmp_path / "out.csv"
    dg_mod.generate_dataset(cfg, out_csv)

    assert out_csv.exists()
    assert len(calls) == 1
    assert calls[0]["total"] == 2 * 2, "total should be |symbols| × runs_per_symbol"
    assert calls[0]["unit"] == "run"


# ---------------------------------------------------------------------------
# Model trainers auto-disable on non-TTY
# ---------------------------------------------------------------------------


def test_progress_disabled_passes_through_to_model_trainer(monkeypatch, synthetic_config):
    """When progress_disabled() returns True, the mlp tqdm bar is suppressed."""
    from mi_race.train.models import mlp as mlp_mod

    real_tqdm_calls: list[dict] = []

    class _FakeTqdm:
        def __init__(self, iterable, **kw):
            real_tqdm_calls.append(kw)
            self._it = iterable

        def __iter__(self):
            return iter(self._it)

    monkeypatch.setattr("tqdm.auto.tqdm", _FakeTqdm)
    # Pytest capture already makes stdout non-TTY → progress_disabled() returns True.

    # Run a minimal mlp through the orchestrator's path. We rely on the fact that
    # if progress_disabled is honored, _FakeTqdm should NOT be constructed.
    import argparse, json as _json
    from mi_race.train.orchestrator import run_cmd

    with open(synthetic_config) as f:
        cfg = _json.load(f)
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    run_cmd(argparse.Namespace(config=str(synthetic_config), model="mlp", dry_run=False))
    assert real_tqdm_calls == [], (
        f"mlp tqdm should be suppressed when progress_disabled() is True; got {real_tqdm_calls}"
    )
