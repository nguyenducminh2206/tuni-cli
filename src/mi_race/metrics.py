"""metrics: how well a decoder tells the symbols apart, measured on the test rows"""

import numpy as np


def confusion_matrix(sent: np.ndarray, predicted: np.ndarray, n_symbols: int) -> np.ndarray:
    """confusion[u, û] = the number of runs where symbol u was sent and û was predicted;
    every symbol 0 ... N-1 gets a row and a column, also one that is never predicted"""
    confusion = np.zeros((n_symbols, n_symbols), dtype=int)
    np.add.at(confusion, (sent, predicted), 1)              # adds 1 for every run, also for repeated pairs
    return confusion