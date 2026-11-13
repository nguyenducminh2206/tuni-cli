"""tests for src/mi_race/search.py"""

import numpy as np
import pytest

from mi_race.metrics import log_loss
from mi_race.search import FLOOR, floor_probabilities


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