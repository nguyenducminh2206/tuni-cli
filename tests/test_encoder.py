"""test for src/mi_race/encoder.py"""

from pathlib import Path

from mi_race.config import load_config
from mi_race.encoder import release_schedule

EXAMPLE = Path(__file__).parents[1] / "configs" / "example.json"

def test_single_pulse():
    assert release_schedule((0, 0, 0, 50), slot_dt=0.1) == [(0.3, 50)]

def test_split_release_keeps_time_order():
    assert release_schedule((10, 0, 20, 0, 20), slot_dt=0.1) == [(0.0, 10), (0.2, 20), (0.4, 20)]

def test_times_have_no_float_noise():
    schedule = release_schedule((0, 0, 0, 50), slot_dt=0.1)
    assert schedule[0][0] == 0.3            # not 0.30000000000000004

def test_every_molecule_is_released():
    vector = (5, 0, 15, 30)
    schedule = release_schedule(vector, slot_dt=0.1)
    assert sum(amount for time, amount in schedule) == sum(vector)

def test_schedules_of_the_example():
    cfg = load_config(EXAMPLE)
    schedules = [release_schedule(vector, cfg.encoder.slot_dt) for vector in cfg.encoder.symbols]
    assert schedules == [[(0.0, 50)], [(0.3, 50)], [(0.6, 50)], [(0.9, 50)]]