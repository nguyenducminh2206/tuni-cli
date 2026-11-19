"""tests for src/mi_race/compare.py"""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mi_race import compare
from mi_race.compare import (NOISE_PAIRS, NOISE_RUN, compare_codebooks, evaluate_codebook, measure_noise,
                             neutral_move, paired_gain)
from mi_race.config import RandomForestConfig, load_config
from mi_race.dataset import simulate_dataset, split_dataset
from mi_race.decoders.forest import train_random_forest
from mi_race.decoders.rescaling import train_rescaled
from mi_race.metrics import evaluate
from mi_race.streams import dataset_seed, random_stream

EXAMPLE = load_config(Path(__file__).parents[1] / "configs" / "example.json")
SMALL = replace(EXAMPLE, data=replace(EXAMPLE.data, runs_per_symbol=50),            # 200 runs, small forests: fast
                decoders={"random_forest": RandomForestConfig(n_estimators=20)})
HAND_PICKED = EXAMPLE.encoder.symbols                                              # bursts at slots 0, 3, 6, 9
CLOSE = tuple(tuple(50 if k == slot else 0 for k in range(10)) for slot in (0, 1, 2, 3))


def test_evaluating_a_codebook_is_the_v1_pipeline_with_rescaling():
    """simulate, split, the forest rescaled with its own stream, evaluate on the test runs"""
    rows = simulate_dataset(replace(SMALL, seed=7))
    train, test = split_dataset(rows, SMALL.data.test_fraction, random_stream(7, "split"))
    model = train_rescaled(train_random_forest, train[:, 1:], train[:, 0], SMALL.decoders["random_forest"],
                           random_stream(7, "random_forest"))
    expected = evaluate(test[:, 0], model.predict_proba(test[:, 1:])) | {"temperature": model.temperature}
    assert evaluate_codebook(SMALL, HAND_PICKED, seed=7) == {"random_forest": expected}


def test_a_plain_evaluation_has_temperature_1():
    rows = simulate_dataset(replace(SMALL, seed=7))
    train, test = split_dataset(rows, SMALL.data.test_fraction, random_stream(7, "split"))
    forest = train_random_forest(train[:, 1:], train[:, 0], SMALL.decoders["random_forest"],
                                 random_stream(7, "random_forest"))
    expected = evaluate(test[:, 0], forest.predict_proba(test[:, 1:])) | {"temperature": 1.0}
    assert evaluate_codebook(SMALL, HAND_PICKED, seed=7, rescaled=False) == {"random_forest": expected}


def test_a_config_without_decoders_is_scored_by_all_three(monkeypatch):
    quick = lambda X, y, settings, rng: train_random_forest(X, y, RandomForestConfig(n_estimators=5), rng)
    monkeypatch.setattr(compare, "TRAINERS", {"random_forest": quick, "mlp": quick, "cnn": quick})
    assert list(evaluate_codebook(replace(SMALL, decoders={}), HAND_PICKED, seed=7)) == ["random_forest", "mlp", "cnn"]


def test_codebooks_are_compared_on_the_datasets_of_a_job():
    scores = compare_codebooks(SMALL, {"hand-picked": HAND_PICKED, "close": CLOSE}, job="check_data", n_datasets=2)
    assert list(scores) == ["hand-picked", "close"] and len(scores["close"]) == 2
    assert scores["close"][1] == evaluate_codebook(SMALL, CLOSE, dataset_seed(SMALL.seed, "check_data", 1))
    final = compare_codebooks(SMALL, {"close": CLOSE}, job="final_data", n_datasets=1)
    assert final["close"][0] == evaluate_codebook(SMALL, CLOSE, dataset_seed(SMALL.seed, "final_data", 0))


def test_a_codebook_given_twice_is_scored_once():
    calls = []
    scores = compare_codebooks(SMALL, {"a": HAND_PICKED, "b": CLOSE, "a again": [list(v) for v in HAND_PICKED]},
                               n_datasets=2, progress=lambda *step: calls.append(step))
    assert scores["a again"] is scores["a"] and [step[:2] for step in calls] == [(1, 4), (2, 4), (3, 4), (4, 4)]


def test_a_paired_gain_by_hand():
    def run(*bits):
        return [{"mlp": {"ba_bound_bits": b}} for b in bits]
    scores = {"found": run(1.2, 1.0, 1.3), "start": run(1.0, 1.1, 1.0)}
    gain = paired_gain(scores, "found", "start", "mlp")
    assert gain["mean"] == pytest.approx(0.4 / 3) and gain["higher"] == 2 and gain["datasets"] == 3
    assert gain["se"] == pytest.approx(np.std([0.2, -0.1, 0.3], ddof=1) / np.sqrt(3))


def test_the_neutral_move_of_the_example_is_the_one_of_section_7():
    moved = neutral_move(HAND_PICKED, budget=50)
    assert moved[1] == (0, 0, 0, 40, 10, 0, 0, 0, 0, 0) and moved[0] == HAND_PICKED[0] and moved[2:] == HAND_PICKED[2:]


def test_the_neutral_move_never_makes_two_symbols_equal():
    symbols = ((40, 10, 0), (50, 0, 0))                     # symbol 1 moved would equal symbol 0
    assert neutral_move(symbols, budget=50) == ((30, 20, 0), (50, 0, 0))
    assert neutral_move(((50, 0), (0, 50)), budget=50) == ((50, 0), (10, 40))       # from the last slot: one back


def test_the_noise_is_measured_on_the_datasets_of_an_unused_search_run():
    seen = []
    def score(cfg, symbols, seed):
        seen.append(seed)
        return {"ba_bound_bits": 1.0 + (0.001 * (seed % 97) if symbols != HAND_PICKED else 0.0)}
    noise = measure_noise(EXAMPLE, score)
    seeds = [dataset_seed(EXAMPLE.seed, "search_data", NOISE_RUN, i) for i in range(NOISE_PAIRS)]
    differences = [0.001 * (seed % 97) for seed in seeds]
    assert seen == [seed for seed in seeds for _ in (0, 1)]              # the moved codebook and its parent, paired
    assert noise["differences"] == pytest.approx(differences)
    assert noise["sd"] == pytest.approx(np.std(differences, ddof=1))
    assert noise["threshold"] == round(2 * noise["sd"], 2) and noise["moved"] == neutral_move(HAND_PICKED, 50)