"""Tests for config.py"""

import dataclasses
import pytest
from mi_race.config import ChannelConfig, EncoderConfig, DataConfig, OutputConfig

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
        encoder.symbols[0][1] == 7


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