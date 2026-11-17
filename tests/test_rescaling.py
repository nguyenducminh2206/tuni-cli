"""tests for src/mi_race/decoders/rescaling.py"""

import numpy as np
import pytest

from mi_race.decoders.rescaling import rescale

P = np.array([[0.7, 0.2, 0.1],
              [0.98, 0.01, 0.01]])


# 6f part 1: rescale
def test_rescale_by_hand():
    roots = np.sqrt([0.7, 0.2, 0.1])                                    # T = 2: the square root of each, then normalise
    assert rescale(P[:1], 2)[0] == pytest.approx(roots / roots.sum())


def test_temperature_1_changes_nothing():
    assert rescale(P, 1) == pytest.approx(P)


def test_rows_add_up_to_1_and_the_most_likely_symbol_never_changes():
    p = np.random.default_rng(0).dirichlet(np.ones(4), size=50)         # 50 runs, 4 symbols
    for temperature in [0.1, 0.5, 2, 10]:
        q = rescale(p, temperature)
        assert q.sum(axis=1) == pytest.approx(np.ones(50))
        assert (q.argmax(axis=1) == p.argmax(axis=1)).all()


def test_above_1_softens_and_below_1_sharpens():
    assert (rescale(P, 3).max(axis=1) < P.max(axis=1)).all()
    assert (rescale(P, 0.5).max(axis=1) > P.max(axis=1)).all()


def test_extreme_temperatures_give_uniform_or_sure_probabilities():
    assert rescale(P, 1e9) == pytest.approx(np.full((2, 3), 1 / 3))
    assert rescale(P, 1e-4) == pytest.approx(np.array([[1, 0, 0], [1, 0, 0]]))    # 0.7^10000 alone would round to 0: no 0/0


def test_a_probability_of_0_stays_0():
    q = rescale(np.array([[0.6, 0.4, 0.0]]), 5)
    assert q[0, 2] == 0 and q[0, :2].sum() == pytest.approx(1)


def test_same_as_dividing_a_networks_scores_by_the_temperature():
    scores = np.array([[2.0, -1.0, 0.5], [0.0, 3.0, 1.0]])
    softmax = lambda z: np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    assert rescale(softmax(scores), 2.5) == pytest.approx(softmax(scores / 2.5))


def test_the_probabilities_passed_in_are_not_changed():
    p = P.copy()
    rescale(p, 2)
    assert (p == P).all()


@pytest.mark.parametrize("temperature", [0, -1])
def test_a_temperature_of_0_or_below_is_a_bug(temperature):
    with pytest.raises(ValueError, match="above 0"):
        rescale(P, temperature)