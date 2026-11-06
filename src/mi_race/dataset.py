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


class DatasetError(ValueError):
    """the dataset file is missing, broken, or does not match the config"""


def read_dataset(path:str | Path) -> np.ndarray:
    """read a csv saved by write_dataset back into a table of whole numbers"""
    path = Path(path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        raise DatasetError(f"dataset file not found: {path}") from None
    header = lines[0] if lines else ""      # an empty file has no lines at all
    names = header.split(",")
    if names != ["symbol"] + [f"rx_{m}" for m in range(len(names) - 1)]:
        raise DatasetError(f"{path} must start with the line 'symbol,rx_0,...,rx_M'")
    if len(lines) == 1:
        raise DatasetError(f"{path} has no runs, only the header line")
    rule = f"every line after the first must be {len(names)} whole numbers >= 0, separated by commas"
    try:
        rows = np.loadtxt(lines[1:], delimiter=",", dtype=int, ndmin=2)
    except ValueError:
        raise DatasetError(f"{path}: {rule}") from None
    if rows.shape[1] != len(names) or (rows < 0).any():
        raise DatasetError(f"{path}: {rule}")
    return rows


def check_dataset(rows: np.ndarray, cfg: Config) -> None:
    """make sure the rows read from config.data.path fit this config: rx cols, symbol numbers, runs per symbol"""
    path, again = cfg.data.path, "simulate it again with this config"
    n_values = cfg.channel.n_intervals + 1
    if rows.shape[1] - 1 != n_values:
        raise DatasetError(
            f"{path} has {rows.shape[1] - 1} rx values per run, but the config needs {n_values} (T/dt + 1): {again}"
        )
    n_symbols, symbols = cfg.encoder.n_symbols, rows[:, 0]
    if symbols.max() >= n_symbols:
        raise DatasetError(
            f"{path} has runs of symbol {symbols.max()}, but the config has only symbols 0 ... {n_symbols - 1}: {again}"
        )
    for u, runs in enumerate(np.bincount(symbols, minlength=n_symbols)):
        if runs != cfg.data.runs_per_symbol:
            raise DatasetError(
                f"{path} has {runs} runs of symbol {u}, but 'data.runs_per_symbol' is {cfg.data.runs_per_symbol}: {again}"
            )
        