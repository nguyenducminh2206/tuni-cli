"""Tests for config.py"""

import dataclasses
import pytest
from pathlib import Path
from mi_race.config import (ChannelConfig, EncoderConfig, DataConfig, ConfigError,
                            OutputConfig, Config, read_config_file)

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
