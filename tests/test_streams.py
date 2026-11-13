"""tests for src/mi_race/streams.py"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mi_race.config import load_config
from mi_race.dataset import simulate_dataset
from mi_race.streams import JOBS, dataset_seed, random_stream

def numbers(*args):
    """the first few random numbers of a stream"""
    return list(random_stream(*args).random(5))


# 4a: random streams
def test_same_seed_job_and_indices_give_the_same_numbers():
    assert numbers(12345, "simulation", 2, 7) == numbers(12345, "simulation", 2, 7)


@pytest.mark.parametrize("other", [
    (54321, "simulation", 2, 7),                    # another seed
    (12345, "split"),                               # another job
    (12345, "simulation", 3, 7),                    # another symbol
    (12345, "simulation", 2, 8),                    # another run
    (12345, "simulation", 7, 2),                    # symbol and run swapped
])
def test_anything_else_gives_other_numbers(other):
    assert numbers(*other) != numbers(12345, "simulation", 2, 7)


def test_each_decoder_has_its_own_stream():
    forest, mlp, cnn = [numbers(12345, name) for name in ("random_forest", "mlp", "cnn")]
    assert forest != mlp and mlp != cnn and forest != cnn


def test_a_stream_does_not_depend_on_what_ran_before():
    first = numbers(12345, "split")
    random_stream(12345, "simulation", 0, 0).random(1000)      # another job uses many numbers
    assert numbers(12345, "split") == first


def test_job_numbers_never_change():
    # renumbering would silently change the results of old experiments; new jobs only go at the end
    assert JOBS == {"simulation": 0, "split": 1, "random_forest": 2, "mlp": 3, "cnn": 4,
                    "search_moves": 5, "search_data": 6, "check_data": 7, "final_data": 8}


def test_unknown_job_is_a_bug():
    with pytest.raises(KeyError):
        random_stream(12345, "rnn")


# 8a: a seed for every dataset of the encoder search
def test_dataset_seed_is_repeatable_and_a_plain_int():
    assert dataset_seed(12345, "final_data", 3) == dataset_seed(12345, "final_data", 3)
    assert type(dataset_seed(12345, "final_data", 3)) is int


def test_search_check_and_final_datasets_never_share_a_seed():
    search = {dataset_seed(12345, "search_data", run, step) for run in range(5) for step in range(200)}
    check = {dataset_seed(12345, "check_data", i) for i in range(20)}
    final = {dataset_seed(12345, "final_data", i) for i in range(20)}
    assert (len(search), len(check), len(final)) == (1000, 20, 20)        # no repeats inside a kind
    assert not search & check and not search & final and not check & final


def test_another_master_seed_gives_other_dataset_seeds():
    assert dataset_seed(1, "final_data", 0) != dataset_seed(2, "final_data", 0)


def test_a_dataset_seed_works_as_the_master_seed_of_a_dataset():
    cfg = load_config(Path(__file__).parents[1] / "configs" / "example.json")
    small = replace(cfg, data=replace(cfg.data, runs_per_symbol=5))                 # 20 runs, fast
    check = replace(small, seed=dataset_seed(cfg.seed, "check_data", 0))
    assert np.array_equal(simulate_dataset(check), simulate_dataset(check))       # the same dataset every time
    assert not np.array_equal(simulate_dataset(check), simulate_dataset(small))   # but not the example's