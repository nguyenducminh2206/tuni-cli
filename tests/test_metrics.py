"""tests for src/mi_race/metrics.py"""

import numpy as np

from mi_race.metrics import confusion_matrix


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