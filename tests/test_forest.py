"""tests for src/mi_race/decoders/forest.py"""

import dataclasses
from pathlib import Path
import numpy as np
import pytest

from mi_race.config import RandomForestConfig, load_config
from mi_race.dataset import simulate_dataset, split_dataset
from mi_race.decoders.forest import train_random_forest
from mi_race.streams import random_stream

EXAMPLE = Path(__file__).parents[1] / "configs" / "example.json"
SMALL_FOREST = RandomForestConfig(n_estimators=20)          # few trees keep the tests fast


@pytest.fixture(scope="module")
def data():
    """the example config with 20 runs per symbol, split: X_train, y_train, X_test, y_test"""
    cfg = load_config(EXAMPLE)
    cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, runs_per_symbol=20))
    train, test = split_dataset(simulate_dataset(cfg), cfg.data.test_fraction, random_stream(cfg.seed, "split"))
    return train[:, 1:], train[:, 0], test[:, 1:], test[:, 0]


def forest(data, seed=12345):
    """a small forest trained on the training rows"""
    X_train, y_train, _, _ = data
    return train_random_forest(X_train, y_train, SMALL_FOREST, random_stream(seed, "random_forest"))


# 5c: random forest
def test_one_probability_per_symbol(data):
    probabilities = forest(data).predict_proba(data[2])
    assert probabilities.shape == (16, 4)                   # 4 test runs of each of the 4 symbols
    assert probabilities.sum(axis=1) == pytest.approx(1)    # every row adds up to 1


def test_columns_are_the_symbols_in_order(data):
    assert forest(data).classes_.tolist() == [0, 1, 2, 3]


def test_number_of_trees_comes_from_the_settings(data):
    assert len(forest(data).estimators_) == 20


def test_same_stream_gives_the_same_forest(data):
    assert np.array_equal(forest(data).predict_proba(data[2]), forest(data).predict_proba(data[2]))


def test_another_seed_gives_another_forest(data):
    assert not np.array_equal(forest(data, seed=12345).predict_proba(data[2]),
                              forest(data, seed=54321).predict_proba(data[2]))


def test_learns_a_signal_that_is_easy_to_tell_apart():
    y = np.repeat(np.arange(4), 10)
    X = np.zeros((40, 41), dtype=int)
    X[np.arange(40), 10 * y] = 5                            # symbol u: 5 molecules in interval 10 u, nothing else
    model = train_random_forest(X, y, SMALL_FOREST, random_stream(0, "random_forest"))
    assert model.predict(X).tolist() == y.tolist()