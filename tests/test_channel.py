"""tests for src/mi_race/channel.py"""

from mi_race.channel import jump_rates, next_jump
import pytest
import numpy as np

# 2b: jump rates
def test_compartment_0_can_only_jump_right():
    assert jump_rates([50, 0, 0], jump_rate=2.0) == [(0, 1, 100.0)]

def test_middle_compartment_jumps_both_ways():
    assert jump_rates([0, 15, 0], jump_rate=2.0) == [(1, 0, 30.0), (1, 2, 30.0)]

def test_last_compartment_can_jump_into_the_receiver():
    receiver = 3                                    # L = 4: compartments 0, 1, 2; receiver at 3
    assert (2, receiver, 14.0) in jump_rates([0, 0, 7], jump_rate=2.0)

def test_empty_tube_has_no_jumps():
    assert jump_rates([0, 0, 0], jump_rate=2.0) == []

def test_rate_grows_with_the_number_of_molecules():
    one = jump_rates([0, 1, 0], jump_rate=2.0)
    ten = jump_rates([0, 10, 0], jump_rate=2.0)
    assert [rate for _, _, rate in ten] == [10 * rate for _, _, rate in one]

def test_shortest_tube():
    # L = 2: one compartment, right next to the receiver
    assert jump_rates([5], jump_rate=2.0) == [(0, 1, 10.0)]


# 2c: 1 SSA event
class FixedRng:
    """a fake random generator: the dart always lands at the fraction we choose"""
    def __init__(self, fraction):
        self.fraction = fraction

    def exponential(self, scale):
        return scale                                # the mean wait

    def random(self):
        return self.fraction


@pytest.mark.parametrize("fraction, expected", [
    (0.0, (0, 1)),                                  # dart at 0   -> piece 0 ... 60
    (0.5, (1, 0)),                                  # dart at 70  -> piece 60 ... 90
    (0.95, (2, 3)),                                 # dart at 133 -> piece 130 ... 140
    (0.9999999999999999, (2, 3)),                   # the very end of the ruler
])
def test_dart_picks_the_piece_it_lands_on(fraction, expected):
    jumps = jump_rates([30, 15, 5], jump_rate=2.0)  # pieces 60, 30, 30, 10, 10 (total 140)
    _, source, target = next_jump(jumps, FixedRng(fraction))
    assert (source, target) == expected

def test_a_single_jump_is_always_picked():
    rng = np.random.default_rng(0)
    for _ in range(100):
        _, source, target = next_jump([(0, 1, 100.0)], rng)
        assert (source, target) == (0, 1)

def test_jumps_are_picked_in_proportion_to_their_rate():
    rng = np.random.default_rng(0)
    jumps = [(1, 0, 30.0), (1, 2, 10.0)]            # left is 3 times as likely as right
    picks = [next_jump(jumps, rng)[1:] for _ in range(10_000)]
    assert picks.count((1, 0)) / len(picks) == pytest.approx(0.75, abs=0.02)

def test_mean_wait_is_one_over_the_total_rate():
    rng = np.random.default_rng(0)
    jumps = jump_rates([30, 15, 5], jump_rate=2.0)  # total 140 per second
    waits = [next_jump(jumps, rng)[0] for _ in range(10_000)]
    assert sum(waits) / len(waits) == pytest.approx(1 / 140, rel=0.05)

def test_same_seed_gives_the_same_jumps():
    jumps = jump_rates([30, 15, 5], jump_rate=2.0)
    a = np.random.default_rng(7)
    b = np.random.default_rng(7)
    assert [next_jump(jumps, a) for _ in range(20)] == [next_jump(jumps, b) for _ in range(20)]