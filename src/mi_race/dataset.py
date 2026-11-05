"""dataset: simulate every symbol many times and save the receiver signal"""

import numpy as np
from mi_race.channel import transmit
from mi_race.config import Config
from mi_race.streams import random_stream


def simulate_dataset(cfg: Config) -> np.ndarray:
    """every run of every symbol, one row per run: symbol, rx_0, ... , rx_m"""
    rows = []
    for u, vector in enumerate(cfg.encoder.symbols):
        for r in range(cfg.data.runs_per_symbol):
            rx = transmit(vector, cfg, random_stream(cfg.seed, "simulation", u, r))
            rows.append([u, *rx])
    return np.array(rows)