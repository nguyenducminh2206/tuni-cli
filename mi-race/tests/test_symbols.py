"""Tests for the dense symbol format, codebook parsing, and `mi-race symbols`."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from mi_race.encoder.codebook import (
    codebook_from_config,
    vector_to_schedule,
    _is_dense_vector,
)
from mi_race.encoder.symbols import (
    generate_initial_symbols,
    normalize_to_budget,
    parse_pairs,
    vec_from_pairs,
    render_symbol_bars,
    run_symbols,
)


# --- generation -----------------------------------------------------------
def test_generate_time_symbols_one_pulse_each():
    syms = generate_initial_symbols(N=4, n_slots=20, budget=100, kind="time")
    assert set(syms) == {0, 1, 2, 3}
    for vec in syms.values():
        assert len(vec) == 20
        assert sum(vec) == 100               # respects budget
        assert sum(1 for v in vec if v > 0) == 1  # single pulse
    # pulses are spread: symbol 0 at slot 0, last symbol at the final slot
    assert syms[0][0] == 100
    assert syms[3][19] == 100


def test_generate_two_pulse_respects_budget():
    syms = generate_initial_symbols(N=3, n_slots=20, budget=100, kind="two-pulse")
    for vec in syms.values():
        assert sum(vec) == 100
        assert sum(1 for v in vec if v > 0) in (1, 2)


def test_normalize_to_budget_sums_exactly():
    out = normalize_to_budget([1, 1, 1], 100)
    assert sum(out) == 100
    assert all(v >= 0 for v in out)


def test_normalize_clamps_negatives():
    out = normalize_to_budget([-5, 10, 0], 50)
    assert sum(out) == 50
    assert out[0] == 0


# --- manual entry parsing -------------------------------------------------
def test_parse_pairs_basic():
    assert parse_pairs("0:30 4:70", 20) == {0: 30, 4: 70}


def test_parse_pairs_rejects_out_of_range():
    with pytest.raises(ValueError):
        parse_pairs("25:50", 20)


def test_vec_from_pairs():
    assert vec_from_pairs({0: 30, 2: 70}, 5) == [30, 0, 70, 0, 0]


def test_render_bars_marks_empty_and_full():
    bars = render_symbol_bars([100, 0, 0], budget=100)
    assert bars[0] == "█" and bars[1] == "·"


# --- codebook parsing (both formats) --------------------------------------
def test_vector_to_schedule_drops_zeros():
    assert vector_to_schedule([50, 0, 20, 0], slot_dt=0.1) == [(0.0, 50), (0.2, 20)]


def test_is_dense_vector_detection():
    assert _is_dense_vector([50, 0, 0]) is True
    assert _is_dense_vector([[0.0, 50]]) is False
    assert _is_dense_vector([]) is True


def test_codebook_parses_dense_vectors():
    cfg = {"slot_dt": 0.1, "symbols": {"0": [50, 0, 50, 0], "1": [0, 100, 0, 0]}}
    cb = codebook_from_config(cfg)
    assert cb[0] == [(0.0, 50), (0.2, 50)]
    assert cb[1] == [(0.1, 100)]


def test_codebook_still_parses_legacy_sparse():
    cfg = {"symbols": {"0": [[0.0, 100]], "1": [[0.5, 100]]}}
    cb = codebook_from_config(cfg)
    assert cb[0] == [(0.0, 100)]
    assert cb[1] == [(0.5, 100)]


# --- the symbols CLI (non-interactive) ------------------------------------
def test_run_symbols_writes_config(tmp_path: Path):
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({"data": {"y_col": "symbol"}, "channel": {}}))
    args = argparse.Namespace(
        config=str(cfg_path), type="time", n=4, slots=20, budget=100,
        slot_dt=0.1, yes=True,
    )
    run_symbols(args)

    cfg = json.loads(cfg_path.read_text())
    ch = cfg["channel"]
    assert ch["n_slots"] == 20 and ch["budget"] == 100 and ch["slot_dt"] == 0.1
    assert set(ch["symbols"]) == {"0", "1", "2", "3"}
    assert all(sum(v) == 100 for v in ch["symbols"].values())
    # the written codebook round-trips through the parser
    cb = codebook_from_config(ch)
    assert cb[0] == [(0.0, 100)]
