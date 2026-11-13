"""tests for src/mi_race/search.py"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mi_race.config import ConfigError, RandomForestConfig, _parse_symbols, load_config
from mi_race.decoders.forest import train_random_forest
from mi_race.metrics import log_loss
from mi_race.search import (FLOOR, floor_probabilities, out_of_fold_probabilities, score_codebook,
                            legal_moves, propose_move)


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



# 8b part 3: score_codebook
EXAMPLE = load_config(Path(__file__).parents[1] / "configs" / "example.json")
SMALL = replace(EXAMPLE, data=replace(EXAMPLE.data, runs_per_symbol=50),                  # 200 runs and small
                decoders={"random_forest": RandomForestConfig(n_estimators=20)})        # forests: fast tests
HAND_PICKED = EXAMPLE.encoder.symbols                                   # bursts at slots 0, 3, 6, 9
CLOSE = tuple(tuple(50 if k == slot else 0 for k in range(10)) for slot in (0, 1, 2, 3))


def test_the_same_seed_gives_the_same_score():
    assert score_codebook(SMALL, HAND_PICKED, seed=7) == score_codebook(SMALL, HAND_PICKED, seed=7)


def test_another_seed_gives_another_score():
    assert score_codebook(SMALL, HAND_PICKED, seed=7) != score_codebook(SMALL, HAND_PICKED, seed=8)


def test_only_the_seed_argument_matters_not_the_seed_in_the_base_config():
    other_base = replace(SMALL, seed=999)
    assert score_codebook(other_base, HAND_PICKED, seed=7) == score_codebook(SMALL, HAND_PICKED, seed=7)


def test_lists_and_tuples_give_the_same_score():
    as_lists = [list(v) for v in HAND_PICKED]
    assert score_codebook(SMALL, as_lists, seed=7) == score_codebook(SMALL, HAND_PICKED, seed=7)


def test_every_run_is_scored():
    assert score_codebook(SMALL, HAND_PICKED, seed=7)["n_test"] == 4 * 50


def test_the_score_tells_a_good_codebook_from_a_bad_one():
    """bursts 0.3 s apart are much easier to tell apart than bursts 0.1 s apart"""
    good = score_codebook(SMALL, HAND_PICKED, seed=7)["ba_bound_bits"]
    bad = score_codebook(SMALL, CLOSE, seed=7)["ba_bound_bits"]
    assert good > bad + 0.3


@pytest.mark.parametrize("symbols, message", [
    ([[50] + [0] * 9, [49] + [0] * 9], "budget"),                     # spends 49 molecules, not 50
    ([[50] + [0] * 9, [50] + [0] * 9], "same as"),                    # two symbols alike: no decoder could tell them apart
    ([[50] + [0] * 8, [0, 50] + [0] * 7], "entries"),                 # 9 slots, not 10
])
def test_a_codebook_that_breaks_the_config_rules_is_rejected(symbols, message):
    with pytest.raises(ConfigError, match=message):
        score_codebook(SMALL, symbols, seed=7)


# 8c: moves
def test_legal_moves_by_hand():
    """moving 10 from slot 0 to slot 1 would turn symbol 0 into symbol 1, and moving symbol 1's
    last 10 back would turn it into symbol 0: only one move is left"""
    assert legal_moves([[50, 0], [40, 10]], chunk=10) == [((1, 0, 1), ((50, 0), (30, 20)))]


def test_every_pulse_can_send_a_chunk_to_any_other_slot():
    """4 symbols x 1 full slot x 9 other slots"""
    assert len(legal_moves(HAND_PICKED)) == 36


def test_every_move_keeps_the_codebook_legal():
    for move, codebook in legal_moves(HAND_PICKED) + legal_moves(CLOSE):
        assert _parse_symbols([list(v) for v in codebook], n_slots=10, budget=50)        # raises if it breaks a rule


def test_a_move_changes_exactly_two_slots_of_one_symbol():
    for (u, source, target), codebook in legal_moves(HAND_PICKED):
        diff = np.array(codebook) - np.array(HAND_PICKED)
        assert diff[u, source] == -10 and diff[u, target] == 10
        assert np.count_nonzero(diff) == 2


def test_no_legal_move_gives_none():
    """no slot holds a whole chunk"""
    assert legal_moves([[5, 5], [4, 6]], chunk=10) == []
    assert propose_move([[5, 5], [4, 6]], np.random.default_rng(0), chunk=10) is None


def test_every_legal_move_is_equally_likely():
    rng = np.random.default_rng(0)
    counts = {}
    for _ in range(3600):
        move, _ = propose_move(HAND_PICKED, rng)
        counts[move] = counts.get(move, 0) + 1
    assert len(counts) == 36                                        # every move turned up
    assert all(60 < n < 140 for n in counts.values())               # about 100 each (3600 / 36)


def test_the_same_generator_gives_the_same_move():
    assert propose_move(HAND_PICKED, np.random.default_rng(3)) == propose_move(HAND_PICKED, np.random.default_rng(3))


def test_the_codebook_passed_in_is_not_changed():
    symbols = [[50, 0], [0, 50]]
    propose_move(symbols, np.random.default_rng(0), chunk=10)
    assert symbols == [[50, 0], [0, 50]]


def test_a_chunk_below_1_is_a_bug():
    with pytest.raises(ValueError, match="at least 1"):
        legal_moves(HAND_PICKED, chunk=0)