"""tests for src/mi_race/channel.py"""

from mi_race.channel import jump_rates

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