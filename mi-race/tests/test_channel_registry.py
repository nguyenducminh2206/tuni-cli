"""Tests for the pluggable channel layer."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import argparse

from mi_race.channel.registry import (
    build_channel,
    list_channels,
    scaffold_channel,
    CHANNEL_REGISTRY,
)


CFG = {"type": "ssa", "L": 4, "S": 1.0, "D": 2.0, "dt": 0.1, "T": 1.0}


def test_ssa_is_registered():
    assert "ssa" in CHANNEL_REGISTRY
    names = [n for n, _d, _p in list_channels()]
    assert "ssa" in names


def test_build_ssa_returns_times_and_X():
    channel = build_channel(CFG)
    rng = np.random.default_rng(0)
    times, X = channel([(0.0, 50)], rng)
    n_steps = int(round(CFG["T"] / CFG["dt"])) + 1
    assert X.shape == (n_steps, CFG["L"])
    assert len(times) == n_steps
    assert X[0, 0] == 50                 # released into compartment 0 at t=0


def test_unknown_channel_type_errors():
    with pytest.raises(SystemExit):
        build_channel({"type": "does_not_exist", "L": 4})


def test_absorbing_is_registered():
    assert "ssa_absorbing" in CHANNEL_REGISTRY
    assert "ssa_absorbing" in [n for n, _d, _p in list_channels()]


def test_absorbing_drains_mass_but_reflecting_conserves_it():
    # Long enough for molecules to reach the far end and be absorbed.
    cfg = {"L": 5, "S": 1.0, "D": 3.0, "dt": 0.02, "T": 4.0}
    schedule = [(0.0, 200)]

    _t, X_ref = build_channel({**cfg, "type": "ssa"})(schedule, np.random.default_rng(0))
    _t, X_abs = build_channel({**cfg, "type": "ssa_absorbing"})(schedule, np.random.default_rng(0))

    # Reflecting conserves total mass at every timestep.
    assert X_ref.sum(axis=1).min() == 200 and X_ref.sum(axis=1).max() == 200
    # Absorbing starts at 200 but drains well below it by the end.
    assert X_abs.sum(axis=1)[0] == 200
    assert X_abs.sum(axis=1)[-1] < 200


def test_default_type_is_ssa():
    channel = build_channel({"L": 4, "S": 1.0, "D": 2.0, "dt": 0.1, "T": 1.0})
    times, X = channel([(0.0, 10)], np.random.default_rng(0))
    assert X.shape[1] == 4


# --- custom channel loading ----------------------------------------------
_CUSTOM_SRC = '''
import numpy as np
def simulate(schedule, cfg, rng):
    L = int(cfg["L"])
    n = int(round(cfg["T"] / cfg["dt"])) + 1
    X = np.zeros((n, L), dtype=int)
    for t, a in schedule:
        X[0, 0] += int(a)
    times = np.linspace(0.0, cfg["T"], n)
    return times, X
'''


def test_custom_channel_loads_from_file(tmp_path: Path):
    f = tmp_path / "my_channel.py"
    f.write_text(_CUSTOM_SRC)
    cfg = {"type": "custom", "impl": f"{f}:simulate", "L": 3, "T": 1.0, "dt": 0.1}
    channel = build_channel(cfg)
    times, X = channel([(0.0, 7)], np.random.default_rng(0))
    assert X.shape == (11, 3)
    assert X[0, 0] == 7


def test_custom_missing_impl_errors():
    with pytest.raises(SystemExit):
        build_channel({"type": "custom"})


def test_custom_bad_output_shape_is_caught(tmp_path: Path):
    f = tmp_path / "bad.py"
    f.write_text(
        "import numpy as np\n"
        "def simulate(schedule, cfg, rng):\n"
        "    return np.arange(5), np.zeros(5)\n"  # X is 1D, not (n_steps, L)
    )
    cfg = {"type": "custom", "impl": f"{f}:simulate", "L": 3, "T": 1.0, "dt": 0.1}
    channel = build_channel(cfg)
    with pytest.raises(SystemExit):
        channel([(0.0, 1)], np.random.default_rng(0))


# --- scaffold (the `mi-race channels --new` template) ---------------------
def test_scaffold_produces_a_working_channel(tmp_path: Path):
    f = tmp_path / "scaffold.py"
    scaffold_channel(str(f))
    assert f.exists() and "def simulate(" in f.read_text()
    # the scaffolded file loads and runs as a real channel
    cfg = {"type": "custom", "impl": f"{f}:simulate", "L": 4, "dt": 0.1, "T": 1.0}
    times, X = build_channel(cfg)([(0.0, 9)], np.random.default_rng(0))
    assert X.shape == (11, 4)
    assert X[0, 0] == 9


def test_scaffold_refuses_to_overwrite(tmp_path: Path):
    f = tmp_path / "exists.py"
    f.write_text("# already here")
    with pytest.raises(SystemExit):
        scaffold_channel(str(f))
