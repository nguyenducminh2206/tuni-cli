"""tests for src/mi_race/dataset.py"""

import dataclasses
from pathlib import Path
import numpy as np
import pytest

from mi_race.channel import transmit
from mi_race.config import load_config
from mi_race.dataset import simulate_dataset
from mi_race.streams import random_stream

EXAMPLE = Path(__file__).parents[1] / "configs" / "example.json"


def with_runs(cfg, runs):
    """the same config with another number of runs per symbol"""
    return dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, runs_per_symbol=runs))

@pytest.fixture
def cfg():
    return with_runs(load_config(EXAMPLE), 20)      # 20 runs keep the tests fast


# 4b: simulate the whole dataset
def test_one_row_per_run_and_one_column_per_value(cfg):
    rows = simulate_dataset(cfg)
    assert rows.shape == (4 * 20, 1 + 201)          # 4 symbols x 20 runs; symbol, rx_0 ... rx_200

def test_every_symbol_gets_the_same_number_of_runs(cfg):
    rows = simulate_dataset(cfg)
    assert np.bincount(rows[:, 0]).tolist() == [20, 20, 20, 20]

def test_rx_0_is_always_0(cfg):
    assert simulate_dataset(cfg)[:, 1].sum() == 0

def test_no_run_counts_more_than_the_budget(cfg):
    totals = simulate_dataset(cfg)[:, 1:].sum(axis=1)
    assert totals.max() <= cfg.encoder.budget

def test_nothing_arrives_before_the_release(cfg):
    rows = simulate_dataset(cfg)
    symbol_3 = rows[rows[:, 0] == 3, 1:]            # the signals of symbol 3, which releases at 0.9 s
    assert symbol_3[:, :91].sum() == 0              # rx_0 ... rx_90 cover t <= 0.9 s
    assert symbol_3.sum() > 0

def test_same_config_gives_the_same_dataset(cfg):
    assert np.array_equal(simulate_dataset(cfg), simulate_dataset(cfg))

def test_another_seed_gives_another_dataset(cfg):
    other = dataclasses.replace(cfg, seed=cfg.seed + 1)
    assert not np.array_equal(simulate_dataset(cfg), simulate_dataset(other))

def test_more_runs_keep_the_first_runs(cfg):
    small = simulate_dataset(cfg)
    big = simulate_dataset(with_runs(cfg, 30))
    for u in range(4):
        first_20 = big[big[:, 0] == u][:20]
        assert np.array_equal(first_20, small[small[:, 0] == u])

def test_one_run_can_be_recreated_alone(cfg):
    rows = simulate_dataset(cfg)
    u, r = 2, 7
    rx = transmit(cfg.encoder.symbols[u], cfg, random_stream(cfg.seed, "simulation", u, r))
    assert rows[rows[:, 0] == u][r, 1:].tolist() == rx

def test_mean_absorbed_matches_the_exact_values():
    rows = simulate_dataset(load_config(EXAMPLE))   # the full 300 runs per symbol
    totals = rows[:, 1:].sum(axis=1)
    means = [totals[rows[:, 0] == u].mean() for u in range(4)]
    assert means == pytest.approx([22.40, 18.95, 15.13, 10.99], abs=1.0)