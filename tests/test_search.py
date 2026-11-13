"""tests for src/mi_race/search.py"""

import numpy as np
import pytest

from mi_race.config import RandomForestConfig
from mi_race.decoders.forest import train_random_forest
from mi_race.metrics import log_loss
from mi_race.search import FLOOR, floor_probabilities, out_of_fold_probabilities


# 8b part 1: the probability floor
def test_floor_by_hand():
    """a sure decoder: the three zeros become the floor, then the row is rescaled to add up to 1"""
    q = floor_probabilities(np.array([[1.0, 0.0, 0.0, 0.0]]), floor=0.01)
    assert q == pytest.approx(np.array([[1, 0.01, 0.01, 0.01]]) / 1.03)


def test_rows_add_up_to_1_and_no_probability_stays_near_0():
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(4), size=50)
    p[p < 0.2] = 0                                              # many zeros, like the forest gives
    p /= p.sum(axis=1, keepdims=True)
    q = floor_probabilities(p)
    assert q.sum(axis=1) == pytest.approx(np.ones(50))
    assert q.min() >= FLOOR / (1 + 3 * FLOOR)                  # the floor after the worst-case rescaling


def test_probabilities_above_the_floor_are_left_alone():
    p = np.array([[0.7, 0.2, 0.1], [0.4, 0.3, 0.3]])
    assert floor_probabilities(p) == pytest.approx(p)


def test_the_most_likely_symbol_never_changes():
    """so accuracy, the confusion matrix and hard MI are the same with or without the floor"""
    rng = np.random.default_rng(1)
    p = rng.dirichlet(np.ones(4) * 0.3, size=200)              # spiky rows, many close to 0
    assert (floor_probabilities(p).argmax(axis=1) == p.argmax(axis=1)).all()


def test_a_sure_but_wrong_run_costs_much_less():
    """without the floor it costs about 36 nats (the machine-epsilon guard); with it, about 6"""
    sent, p = np.array([1]), np.array([[1.0, 0.0, 0.0, 0.0]])
    assert log_loss(sent, p) == pytest.approx(36.04, abs=0.01)
    assert log_loss(sent, floor_probabilities(p)) == pytest.approx(-np.log(FLOOR / (1 + 3 * FLOOR)))


def test_the_decoder_output_is_not_changed():
    p = np.array([[1.0, 0.0]])
    floor_probabilities(p)
    assert (p == np.array([[1.0, 0.0]])).all()


# 8b part 2: out-of-fold probabilities
class Memorizer:
    """a fake decoder: it knows the symbol of every run it trained on, and nothing about any other run"""

    def __init__(self, X, y):
        self.known = {tuple(x): u for x, u in zip(X, y)}
        self.n_symbols = int(y.max()) + 1

    def predict_proba(self, X):
        p = np.full((len(X), self.n_symbols), 1 / self.n_symbols)       # an unknown run: every symbol equally likely
        for i, x in enumerate(X):
            if tuple(x) in self.known:
                p[i] = np.eye(self.n_symbols)[self.known[tuple(x)]]   # a known run: sure of its symbol
        return p


def memorize(X, y, settings, rng):
    """the Memorizer's train function, with the same arguments as the real ones"""
    return Memorizer(X, y)


X_UNIQUE = np.arange(200).reshape(-1, 1)                           # 200 runs, no two alike
Y_UNIQUE = np.repeat(np.arange(4), 50)                              # 50 runs of each symbol


def test_no_run_is_scored_by_a_decoder_that_trained_on_it():
    """if any run had been in its decoder's training rows, the Memorizer would be sure about it"""
    p = out_of_fold_probabilities(X_UNIQUE, Y_UNIQUE, memorize, None, np.random.default_rng(0), np.random.default_rng(1))
    assert p.shape == (200, 4)
    assert (p == 1 / 4).all()


def test_every_fold_holds_out_the_same_number_of_runs_of_each_symbol():
    trained_on = []
    def record(X, y, settings, rng):
        trained_on.append(np.bincount(y).tolist())
        return Memorizer(X, y)
    out_of_fold_probabilities(X_UNIQUE, Y_UNIQUE, record, None, np.random.default_rng(0), np.random.default_rng(1))
    assert trained_on == [[40, 40, 40, 40]] * 5                    # 5 decoders, each trained on 4/5 of every symbol


def forest_probabilities(fold_seed, forest_seed):
    rng = np.random.default_rng(2)
    X, y = rng.poisson(2.0, size=(100, 10)), np.repeat(np.arange(4), 25)
    return out_of_fold_probabilities(X, y, train_random_forest, RandomForestConfig(n_estimators=20),
                                     np.random.default_rng(fold_seed), np.random.default_rng(forest_seed))


def test_the_same_generators_give_the_same_probabilities():
    first = forest_probabilities(0, 1)
    assert (first == forest_probabilities(0, 1)).all()
    assert first.sum(axis=1) == pytest.approx(np.ones(100))


def test_other_folds_give_other_probabilities():
    assert not (forest_probabilities(0, 1) == forest_probabilities(5, 1)).all()