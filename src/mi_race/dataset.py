"""dataset: simulate every symbol many times and save the receiver signal"""

import numpy as np
from pathlib import Path

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

def write_dataset(path: str | Path, rows: np.ndarray) -> None:
    """save the dataset as csv: a header line 'symbol, rx_0, ..., rx_M', then one line per run"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)      # e.g. create data/ the first time
    header = ",".join(["symbol"] + [f"rx_{m}" for m in range(rows.shape[1] - 1)])
    np.savetxt(path, rows, fmt="%d", delimiter=",", header=header, comments="")

def mean_signals(rows: np.ndarray) -> np.ndarray:
    """the mean receiver signal of each symbol: row u is rx_0 ... rx_M averaged over the runs of symbol u"""
    symbols = rows[:, 0]
    return np.array([rows[symbols == u, 1:].mean(axis=0) for u in range(symbols.max() + 1)])