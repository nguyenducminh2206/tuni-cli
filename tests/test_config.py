"""Tests for config.py"""

import dataclasses
import pytest
from pathlib import Path
from mi_race.config import (ChannelConfig, EncoderConfig, DataConfig, ConfigError,
                            OutputConfig, Config, read_config_file, _check_keys, _whole, _positive,
                            _parse_channel, _parse_symbols)


# 3a: ChannelConfig
def test_jump_rate_is_D_over_h_squared():
    channel = ChannelConfig(L=4, h=0.5,  D=2.0, dt=0.01, T=2.0)
    assert channel.jump_rate == 8.0     # 2.0 / 0.5² = 8

def test_n_intervals_is_T_over_dt():
    channel = ChannelConfig(L=4, h=1.0, D=2.0, dt=0.01, T=2.0)
    assert channel.n_intervals == 200

def test_channel_cannot_be_changed():
    channel = ChannelConfig(L=4, h=1.0, D=2.0, dt=0.01, T=2.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        channel.L = 5


# 3b: EncoderConfig
def test_n_symbols_counts_the_vectors():
    encoder = EncoderConfig(n_slots=3, slot_dt=0.1, budget=50, symbols=((50, 0, 0), (0, 0, 50)))
    assert encoder.n_symbols == 2

def test_symbols_cannot_be_changed():
    encoder = EncoderConfig(n_slots=3, slot_dt=0.1, budget=50, symbols=((50, 0, 0), (0, 0, 50)))
    with pytest.raises(TypeError):
        encoder.symbols[0][1] = 7


# 3c: DataConfig and OutputConfig
def test_data_defaults_match_the_spec():
    data = DataConfig(path="data/example.csv")
    assert data.runs_per_symbol == 300
    assert data.test_fraction == 0.2

def test_data_path_is_required():
    with pytest.raises(TypeError):
        DataConfig()

def test_output_dir_defaults_to_outputs():
    assert OutputConfig().dir == "outputs"


# 3d: Config
def make_config():
    """A small, complete Config built by hand (only for testing)"""
    return Config(
        schema_version=1, 
        channel=ChannelConfig(L=4, h=1.0, D=2.0, dt=0.01, T=2.0),
        encoder=EncoderConfig(n_slots=3, slot_dt=0.1, budget=50, symbols=((50, 0, 0), (0, 0, 50))),
        data=DataConfig(path="data/example.csv")
    )

def test_config_defaults_match_the_spec():
    cfg = make_config()
    assert cfg.seed == 0
    assert cfg.decoders == {}
    assert cfg.output == OutputConfig(dir="outputs")

def test_each_config_gets_its_own_decoders_dict():
    assert make_config().decoders is not make_config().decoders

def test_config_cannot_be_changed():
    cfg = make_config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.seed = 1


# 4: reading the file
EXAMPLE = Path(__file__).parents[1] / "configs" / "example.json"

def test_read_the_example_file():
    raw = read_config_file(EXAMPLE)
    assert raw["schema_version"] == 1
    assert raw["encoder"]["symbols"][1][3] == 50

def test_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        read_config_file(tmp_path / "nope.json")

def test_broken_json(tmp_path):
    broken = tmp_path / "broken.json"
    broken.write_text('{"budget": 50, }')
    with pytest.raises(ConfigError, match="not valid json"):
        read_config_file(broken)

def test_duplicate_key(tmp_path):
    dup = tmp_path / "dup.json"
    dup.write_text('{"budget": 50, "budget": 40 }')
    with pytest.raises(ConfigError, match="'budget' appears twice"):
        read_config_file(dup)

def test_duplicate_key_inside_a_section(tmp_path):
    dup = tmp_path / "dup.json"
    dup.write_text('{"channel": {"L": 4, "L": 5} }')
    with pytest.raises(ConfigError, match="'L' appears twice"):
        read_config_file(dup)


# 5: missing and unknown keys
DATA_REQUIRED = ("path",)
DATA_OPTIONAL = ("runs_per_symbol", "test_fraction")
CHANNEL_KEYS = ("L", "h", "D", "dt", "T")

def test_keys_ok():
    _check_keys({"path": "x.csv", "runs_per_symbol": 300}, "data", DATA_REQUIRED, DATA_OPTIONAL)

def test_unknown_key_suggests_the_right_one():
    with pytest.raises(ConfigError, match="unknown key 'runs_per_symbo' in 'data'. Did you mean 'runs_per_symbol'"):
        _check_keys({"path": "x.csv", "runs_per_symbo": 300}, "data", DATA_REQUIRED, DATA_OPTIONAL)

def test_capitals_typo_suggests_the_right_key():
    channel = {"L": 4, "h": 1.0, "D": 2.0, "dT": 0.01, "T": 2.0}
    with pytest.raises(ConfigError, match="Did you mean 'dt'"):
        _check_keys(channel, "channel", CHANNEL_KEYS)

def test_unknown_key_without_a_close_match_lists_the_allowed_keys():
    with pytest.raises(ConfigError, match="Allowed: path, runs_per_symbol, test_fraction"):
        _check_keys({"path": "x.csv", "colour": "red"}, "data", DATA_REQUIRED, DATA_OPTIONAL)

def test_missing_keys_are_all_listed():
    with pytest.raises(ConfigError, match="'channel' is missing: D, T"):
        _check_keys({"L": 4, "h": 1.0, "dt": 0.01}, "channel", CHANNEL_KEYS)

def test_section_must_be_an_object():
    with pytest.raises(ConfigError, match="'data' must be a json object, not list"):
        _check_keys(["x.csv"], "data", DATA_REQUIRED, DATA_OPTIONAL)


# 6: number checks
def test_whole_accepts_whole_numbers():
    assert _whole(4, "channel.L", minimum=2) == 4

@pytest.mark.parametrize("value", [True, False, 4.5, 4.0, "4", None])
def test_whole_rejects_anything_else(value):
    with pytest.raises(ConfigError, match="must be a whole number"):
        _whole(value, "channel.L", minimum=2)

def test_whole_checks_the_minimum():
    with pytest.raises(ConfigError, match="'channel.L' must be at least 2, got 1"):
        _whole(1, "channel.L", minimum=2)

def test_positive_accepts_numbers_and_returns_a_float():
    assert _positive(2, "channel.D") == 2.0
    assert isinstance(_positive(2, "channel.D"), float)

@pytest.mark.parametrize("value", [0, -1.5, float("nan"), float("inf")])
def test_positive_rejects_zero_negative_nan_and_infinity(value):
    with pytest.raises(ConfigError, match="finite number greater than 0"):
        _positive(value, "channel.D")

@pytest.mark.parametrize("value", [True, "2.0", None])
def test_positive_rejects_non_numbers(value):
    with pytest.raises(ConfigError, match="must be a number"):
        _positive(value, "channel.D")


# 7: channel section
@pytest.fixture
def channel_raw():
    """the channel section of the example config, as a fresh dict for each test."""
    return read_config_file(EXAMPLE)["channel"]

def test_channel_from_the_example(channel_raw):
    channel = _parse_channel(channel_raw)
    assert channel == ChannelConfig(L=4, h=1.0, D=2.0, dt=0.01, T=2.0)
    assert channel.n_intervals == 200

def test_channel_L_at_least_2(channel_raw):
    channel_raw["L"] = 1
    with pytest.raises(ConfigError, match="'channel.L' must be at least 2"):
        _parse_channel(channel_raw)

@pytest.mark.parametrize("key", ["h", "D", "dt", "T"])
def test_channel_physics_must_be_positive(channel_raw, key):
    channel_raw[key] = 0
    with pytest.raises(ConfigError, match=f"'channel.{key}' must be a finite number greater than 0"):
        _parse_channel(channel_raw)

def test_channel_T_over_dt_must_be_whole(channel_raw):
    channel_raw["dt"] = 0.03                        # 2.0 / 0.03 = 66.67 steps
    with pytest.raises(ConfigError, match="whole number of steps"):
        _parse_channel(channel_raw)

def test_channel_T_over_dt_tolerates_float_rounding(channel_raw):
    channel_raw.update(dt=0.1, T=0.3)               # 0.3 / 0.1 = 2.9999999999999996 in Python
    assert _parse_channel(channel_raw).n_intervals == 3

def test_channel_dt_longer_than_T(channel_raw):
    channel_raw["dt"] = 3.0                         # 2.0 / 3.0 = 0.67 steps
    with pytest.raises(ConfigError, match="whole number of steps"):
        _parse_channel(channel_raw)

def test_channel_typo_in_a_key(channel_raw):
    channel_raw["Dt"] = channel_raw.pop("dt")
    with pytest.raises(ConfigError, match="Did you mean 'dt'"):
        _parse_channel(channel_raw)


# 8: the symbols
def test_symbols_become_tuples():
    symbols = _parse_symbols([[50, 0, 0], [0, 20, 30]], n_slots=3, budget=50)
    assert symbols == ((50, 0, 0), (0, 20, 30))
    assert isinstance(symbols[0], tuple)

@pytest.mark.parametrize("symbols", [[[50, 0, 0]], "abc", []])
def test_symbols_needs_a_list_of_at_least_two(symbols):
    with pytest.raises(ConfigError, match="at least 2 release vectors"):
        _parse_symbols(symbols, n_slots=3, budget=50)

def test_each_symbol_must_be_a_list():
    with pytest.raises(ConfigError, match=r"'encoder.symbols\[1\]' must be a list"):
        _parse_symbols([[50, 0, 0], "abc"], n_slots=3, budget=50)

def test_each_symbol_has_n_slots_entries():
    with pytest.raises(ConfigError, match=r"'encoder.symbols\[1\]' has 2 entries, but 'n_slots' = 3"):
        _parse_symbols([[50, 0, 0], [0, 50]], n_slots=3, budget=50)

def test_amounts_cannot_be_negative():
    with pytest.raises(ConfigError, match=r"'encoder.symbols\[1\]\[1\]' must be at least 0"):
        _parse_symbols([[50, 0, 0], [60, -10, 0]], n_slots=3, budget=50)   # still sums to 50

def test_amounts_must_be_whole():
    with pytest.raises(ConfigError, match=r"'encoder.symbols\[1\]\[0\]' must be a whole number"):
        _parse_symbols([[50, 0, 0], [49.5, 0.5, 0]], n_slots=3, budget=50)

def test_every_symbol_uses_the_whole_budget():
    with pytest.raises(ConfigError, match=r"'encoder.symbols\[1\]' releases 40 molecules, but the 'budget' is 50"):
        _parse_symbols([[50, 0, 0], [0, 40, 0]], n_slots=3, budget=50)