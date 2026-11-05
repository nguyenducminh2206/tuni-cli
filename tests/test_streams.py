"""tests for src/mi_race/streams.py"""

from mi_race.streams import JOBS, random_stream
import pytest

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
    # renumbering would silently change the results of old experiments
    assert JOBS == {"simulation": 0, "split": 1, "random_forest": 2, "mlp": 3, "cnn": 4}


def test_unknown_job_is_a_bug():
    with pytest.raises(KeyError):
        random_stream(12345, "rnn")