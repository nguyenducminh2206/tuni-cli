"""tests for src/mi_race/metrics.py"""

import json
import numpy as np
import pytest
from sklearn.metrics import f1_score, mutual_info_score
from sklearn.metrics import log_loss as sklearn_log_loss

from mi_race.metrics import confusion_matrix, accuracy, macro_f1, hard_decision_mi, mi_bias, log_loss, ba_bound, evaluate



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


# 6c: hard-decision MI and its bias
def test_a_perfect_decoder_gives_log2_n_bits():
    assert hard_decision_mi(15 * np.eye(4, dtype=int)) == pytest.approx(2)


def test_mi_does_not_care_what_the_decoder_calls_the_symbols():
    """always predicting the next symbol is never right, but tells you everything"""
    shifted = np.roll(15 * np.eye(4, dtype=int), 1, axis=1)
    assert accuracy(shifted) == 0
    assert hard_decision_mi(shifted) == pytest.approx(2)


def test_a_decoder_that_always_says_the_same_gives_0_bits():
    """three of the four columns are empty, so this also checks 0 · log 0 = 0"""
    confusion = np.zeros((4, 4), dtype=int)
    confusion[:, 2] = 15
    assert hard_decision_mi(confusion) == pytest.approx(0, abs=1e-12)


def test_a_binary_channel_with_one_error_in_four():
    """the textbook answer: 1 - H(1/4) bits"""
    binary_entropy = -(1 / 4 * np.log2(1 / 4) + 3 / 4 * np.log2(3 / 4))
    assert hard_decision_mi(np.array([[3, 1], [1, 3]])) == pytest.approx(1 - binary_entropy)


def test_hard_decision_mi_agrees_with_scikit_learn():
    """scikit-learn gives nats, so divide by ln 2 for bits"""
    rng = np.random.default_rng(0)
    for _ in range(20):
        sent = np.repeat(np.arange(4), 15)
        predicted = np.where(rng.random(60) < 0.5, sent, rng.integers(4, size=60))
        expected = mutual_info_score(sent, predicted) / np.log(2)
        assert hard_decision_mi(confusion_matrix(sent, predicted, 4)) == pytest.approx(expected)


def test_mi_bias_by_hand():
    assert mi_bias(2, 1) == pytest.approx(1 / (2 * np.log(2)))
    assert mi_bias(4, 240) == pytest.approx(9 / (480 * np.log(2)))
    assert mi_bias(4, 480) == pytest.approx(mi_bias(4, 240) / 2)


def test_a_random_guesser_reads_about_mi_bias_bits():
    """the guesses carry 0 bits, but on 240 test runs the plug-in estimate gives about mi_bias on average"""
    rng = np.random.default_rng(0)
    sent = np.repeat(np.arange(4), 60)
    estimates = [hard_decision_mi(confusion_matrix(sent, rng.integers(4, size=240), 4)) for _ in range(500)]
    assert np.mean(estimates) == pytest.approx(mi_bias(4, 240), rel=0.1)


def test_mi_and_bias_give_plain_python_numbers():
    assert type(hard_decision_mi(SMALL)) is float
    assert type(mi_bias(4, 240)) is float


# 6d: log-loss and the Barber–Agakov bound
def test_log_loss_by_hand():
    """only the probability of the sent symbol counts, not the largest one"""
    probabilities = np.array([[0.5, 0.5],
                              [0.9, 0.1]])
    assert log_loss(np.array([0, 1]), probabilities) == pytest.approx((np.log(2) + np.log(10)) / 2)


def test_log_loss_agrees_with_scikit_learn():
    rng = np.random.default_rng(0)
    for _ in range(20):
        sent = np.repeat(np.arange(4), 15)
        probabilities = rng.dirichlet(np.ones(4), size=60)          # random rows that add up to 1
        expected = sklearn_log_loss(sent, probabilities, labels=range(4))
        assert log_loss(sent, probabilities) == pytest.approx(expected)


def test_a_sure_but_wrong_guess_costs_a_lot_but_not_infinity():
    """the forest can give 0 to the sent symbol; np.log(0) would give -inf and a warning"""
    loss = log_loss(np.array([1]), np.array([[1.0, 0.0]]))
    assert np.isfinite(loss)
    assert loss == pytest.approx(-np.log(np.finfo(float).eps))     # about 36 nats


def test_a_sure_and_right_decoder_reaches_log2_n_bits():
    assert ba_bound(np.arange(4), np.eye(4)) == pytest.approx(2)


def test_a_decoder_that_always_says_one_in_n_gets_0_bits():
    assert ba_bound(np.arange(4), np.full((4, 4), 1 / 4)) == pytest.approx(0, abs=1e-12)


def test_an_overconfident_decoder_can_get_a_negative_bound():
    """right half the time but always 99% sure: the bound is still true, it just says nothing"""
    probabilities = np.array([[0.99, 0.01],
                              [0.99, 0.01]])
    assert ba_bound(np.array([0, 1]), probabilities) < 0


def test_log_loss_and_ba_bound_give_plain_python_numbers():
    assert type(log_loss(np.arange(4), np.eye(4))) is float
    assert type(ba_bound(np.arange(4), np.eye(4))) is float


# 6e: evaluate
SENT = np.array([0, 0, 1, 1, 2, 2])
PROBABILITIES = np.array([[0.7, 0.2, 0.1],                  # predicted 0, right
                          [0.4, 0.5, 0.1],                  # predicted 1, wrong
                          [0.1, 0.8, 0.1],                  # predicted 1, right
                          [0.6, 0.3, 0.1],                  # predicted 0, wrong
                          [0.2, 0.3, 0.5],                  # predicted 2, right
                          [0.1, 0.6, 0.3]])                 # predicted 1, wrong


def test_evaluate_gives_every_metric_in_a_fixed_order():
    """the keys and their order are what metrics.json will show"""
    assert list(evaluate(SENT, PROBABILITIES)) == [
        "n_symbols", "n_test", "accuracy", "macro_f1", "hard_decision_mi_bits", "mi_bias_bits",
        "log_loss_nats", "ba_bound_bits", "confusion_matrix",
    ]


def test_the_prediction_is_the_most_likely_symbol():
    assert evaluate(SENT, PROBABILITIES)["confusion_matrix"] == [[1, 1, 0],
                                                                  [1, 1, 0],
                                                                  [0, 1, 1]]


def test_evaluate_agrees_with_the_single_functions():
    result = evaluate(SENT, PROBABILITIES)
    confusion = np.array(result["confusion_matrix"])
    assert result["n_symbols"] == 3
    assert result["n_test"] == 6
    assert result["accuracy"] == accuracy(confusion) == pytest.approx(1 / 2)
    assert result["macro_f1"] == macro_f1(confusion)
    assert result["hard_decision_mi_bits"] == hard_decision_mi(confusion)
    assert result["mi_bias_bits"] == mi_bias(3, 6)
    assert result["log_loss_nats"] == log_loss(SENT, PROBABILITIES)
    assert result["ba_bound_bits"] == ba_bound(SENT, PROBABILITIES)


def test_a_symbol_that_is_never_predicted_keeps_its_column():
    probabilities = np.array([[0.6, 0.4, 0.0],
                              [0.3, 0.7, 0.0],
                              [0.5, 0.2, 0.3]])             # symbol 2 is never the most likely
    assert evaluate(np.array([0, 1, 2]), probabilities)["confusion_matrix"] == [[1, 0, 0],
                                                                                 [0, 1, 0],
                                                                                 [1, 0, 0]]


def test_a_sure_and_right_decoder():
    result = evaluate(np.repeat(np.arange(4), 5), np.repeat(np.eye(4), 5, axis=0))
    assert result["accuracy"] == 1
    assert result["macro_f1"] == 1
    assert result["hard_decision_mi_bits"] == pytest.approx(2)
    assert result["log_loss_nats"] == pytest.approx(0)
    assert result["ba_bound_bits"] == pytest.approx(2)


def test_the_result_can_be_saved_as_json_and_read_back_unchanged():
    """json.dumps fails on numpy whole numbers and arrays, so this checks every value is plain Python"""
    result = evaluate(SENT, PROBABILITIES)
    assert json.loads(json.dumps(result)) == result