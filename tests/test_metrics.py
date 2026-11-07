"""tests for src/mi_race/metrics.py"""

import numpy as np
import pytest
from sklearn.metrics import f1_score

from mi_race.metrics import confusion_matrix, accuracy, macro_f1


# 6a: confusion matrix
def test_counts_every_pair_of_sent_and_predicted():
    sent = np.array([0, 0, 1, 1, 1])
    predicted = np.array([0, 1, 1, 1, 0])
    assert confusion_matrix(sent, predicted, 2).tolist() == [[1, 1], [1, 2]]


def test_repeated_pairs_are_all_counted():
    """confusion[sent, predicted] += 1 would count the pair (0, 1) only once"""
    assert confusion_matrix(np.array([0, 0, 0]), np.array([1, 1, 1]), 2).tolist() == [[0, 3], [0, 0]]


def test_a_symbol_that_is_never_predicted_still_gets_a_column():
    confusion = confusion_matrix(np.array([0, 1, 2]), np.array([0, 1, 1]), 3)
    assert confusion.shape == (3, 3)
    assert confusion[:, 2].tolist() == [0, 0, 0]


def test_a_perfect_decoder_gives_a_diagonal_matrix():
    sent = np.repeat(np.arange(4), 5)
    assert (confusion_matrix(sent, sent, 4) == 5 * np.eye(4, dtype=int)).all()


def test_rows_add_up_to_the_runs_of_each_symbol():
    rng = np.random.default_rng(0)
    sent = np.repeat(np.arange(4), 60)
    predicted = rng.integers(4, size=240)
    confusion = confusion_matrix(sent, predicted, 4)
    assert confusion.sum(axis=1).tolist() == [60, 60, 60, 60]
    assert confusion.sum(axis=0).tolist() == np.bincount(predicted, minlength=4).tolist()
    assert confusion.sum() == 240


# 6b: accuracy and macro-F1
SMALL = np.array([[1, 1],                                   # sent 0: once right, once taken for 1
                  [1, 2]])                                  # sent 1: once taken for 0, twice right


def test_accuracy_is_the_diagonal_over_all_runs():
    assert accuracy(SMALL) == pytest.approx(3 / 5)


def test_accuracy_of_a_perfect_and_an_always_wrong_decoder():
    assert accuracy(np.eye(4, dtype=int)) == 1
    assert accuracy(np.array([[0, 2], [2, 0]])) == 0


def test_macro_f1_by_hand():
    """symbol 0: 2 · 1 / (2 + 2) = 1/2; symbol 1: 2 · 2 / (3 + 3) = 2/3; mean 7/12"""
    assert macro_f1(SMALL) == pytest.approx(7 / 12)


def test_macro_f1_of_a_perfect_decoder_is_1():
    assert macro_f1(5 * np.eye(4, dtype=int)) == 1


def test_a_symbol_that_is_never_predicted_pulls_macro_f1_down():
    """every run is predicted as 0: accuracy is 1/2, but symbol 1 has F1 0"""
    confusion = np.array([[2, 0], [2, 0]])
    assert accuracy(confusion) == pytest.approx(1 / 2)
    assert macro_f1(confusion) == pytest.approx((2 * 2 / (2 + 4) + 0) / 2)        # (2/3 + 0) / 2 = 1/3


def test_macro_f1_agrees_with_scikit_learn():
    rng = np.random.default_rng(0)
    for _ in range(20):
        sent = np.repeat(np.arange(4), 15)
        predicted = rng.integers(4, size=60)
        expected = f1_score(sent, predicted, labels=range(4), average="macro")
        assert macro_f1(confusion_matrix(sent, predicted, 4)) == pytest.approx(expected)


def test_both_give_plain_python_numbers():
    assert type(accuracy(SMALL)) is float
    assert type(macro_f1(SMALL)) is float