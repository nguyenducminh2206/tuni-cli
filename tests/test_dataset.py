"""tests for src/mi_race/dataset.py"""

import dataclasses
from pathlib import Path
import numpy as np
import pytest

from mi_race.channel import transmit
from mi_race.config import load_config
from mi_race.dataset import DatasetError, simulate_dataset, write_dataset, mean_signals, read_dataset
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


# 4c: write the csv
def test_csv_looks_like_this(tmp_path):
    path = tmp_path / "small.csv"
    write_dataset(path, np.array([[0, 0, 1, 2], [1, 0, 0, 3]]))
    assert path.read_text() == "symbol,rx_0,rx_1,rx_2\n0,0,1,2\n1,0,0,3\n"

def test_example_csv_has_a_header_and_one_line_per_run(cfg, tmp_path):
    path = tmp_path / "example.csv"
    write_dataset(path, simulate_dataset(cfg))
    lines = path.read_text().splitlines()
    assert lines[0].startswith("symbol,rx_0,rx_1,")
    assert lines[0].endswith(",rx_200")
    assert len(lines) == 1 + 4 * 20

def test_csv_reads_back_the_same_numbers(cfg, tmp_path):
    path = tmp_path / "example.csv"
    rows = simulate_dataset(cfg)
    write_dataset(path, rows)
    assert np.array_equal(np.loadtxt(path, delimiter=",", skiprows=1, dtype=int), rows)

def test_missing_folders_are_created(tmp_path):
    path = tmp_path / "data" / "new" / "x.csv"
    write_dataset(path, np.array([[0, 0, 1], [1, 0, 2]]))
    assert path.exists()

def test_writing_again_replaces_the_file(tmp_path):
    path = tmp_path / "x.csv"
    write_dataset(path, np.array([[0, 0, 1], [1, 0, 2]]))
    write_dataset(path, np.array([[0, 0, 5], [1, 0, 6]]))
    assert path.read_text() == "symbol,rx_0,rx_1\n0,0,5\n1,0,6\n"


# 4d: the mean signal of each symbol
def test_mean_signals_of_a_small_table():
    rows = np.array([
        [0, 0, 1, 2],
        [0, 0, 3, 4],
        [1, 0, 5, 0],
    ])
    assert mean_signals(rows).tolist() == [[0, 2, 3], [0, 5, 0]]

def test_one_mean_signal_per_symbol(cfg):
    assert mean_signals(simulate_dataset(cfg)).shape == (4, 201)        # 4 symbols, rx_0 ... rx_200

def test_mean_signal_adds_up_to_the_mean_absorbed(cfg):
    rows = simulate_dataset(cfg)
    totals = rows[:, 1:].sum(axis=1)
    expected = [totals[rows[:, 0] == u].mean() for u in range(4)]
    assert mean_signals(rows).sum(axis=1) == pytest.approx(expected)

def test_mean_signal_is_zero_until_the_release(cfg):
    means = mean_signals(simulate_dataset(cfg))
    for u in range(4):
        release = 30 * u                            # symbol u releases at 0.3 * u s = t_(30 u)
        assert means[u, :release + 1].sum() == 0    # rx_0 ... rx_release
        assert means[u].sum() > 0


# 5a: read the dataset back
def csv_file(tmp_path, text):
    """a csv file with this text"""
    path = tmp_path / "data.csv"
    path.write_text(text, encoding="utf-8")
    return path

def test_read_gives_back_what_was_written(cfg, tmp_path):
    rows = simulate_dataset(cfg)
    write_dataset(tmp_path / "data.csv", rows)
    assert np.array_equal(read_dataset(tmp_path / "data.csv"), rows)     # same shape, same numbers

def test_one_run_is_still_a_table(tmp_path):
    path = csv_file(tmp_path, "symbol,rx_0,rx_1\n0,0,3\n")
    assert read_dataset(path).tolist() == [[0, 0, 3]]

def test_missing_file(tmp_path):
    with pytest.raises(DatasetError, match="not found"):
        read_dataset(tmp_path / "nothing.csv")

@pytest.mark.parametrize("text", [
    "",                                         # empty file
    "0,0,1\n1,0,2\n",                           # no header
    "label,rx_0,rx_1\n0,0,1\n",                 # another name
    "symbol,rx_0,rx_2\n0,0,1\n",                # rx_1 missing
    "symbol, rx_0, rx_1\n0,0,1\n",              # spaces
])
def test_wrong_header(tmp_path, text):
    with pytest.raises(DatasetError, match="must start with the line"):
        read_dataset(csv_file(tmp_path, text))


def test_header_but_no_runs(tmp_path):
    with pytest.raises(DatasetError, match="no runs"):
        read_dataset(csv_file(tmp_path, "symbol,rx_0,rx_1\n"))


@pytest.mark.parametrize("lines", [
    "0,0,1.5\n",                                # not a whole number
    "0,0,x\n",                                  # not a number
    "0,0,-1\n",                                 # negative
    "0,0,1\n1,0\n",                             # a line is too short
    "0,0,1,2\n",                                # more values than the header has names
])
def test_bad_lines(tmp_path, lines):
    with pytest.raises(DatasetError, match="3 whole numbers"):
        read_dataset(csv_file(tmp_path, "symbol,rx_0,rx_1\n" + lines))