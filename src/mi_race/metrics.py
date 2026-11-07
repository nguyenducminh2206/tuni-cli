"""metrics: how well a decoder tells the symbols apart, measured on the test rows"""

import numpy as np


def confusion_matrix(sent: np.ndarray, predicted: np.ndarray, n_symbols: int) -> np.ndarray:
    """confusion[u, û] = the number of runs where symbol u was sent and û was predicted;
    every symbol 0 ... N-1 gets a row and a column, also one that is never predicted"""
    confusion = np.zeros((n_symbols, n_symbols), dtype=int)
    np.add.at(confusion, (sent, predicted), 1)              # adds 1 for every run, also for repeated pairs
    return confusion

def accuracy(confusion: np.ndarray) -> float:
    """the share of runs where the predicted symbol is the one that was sent"""
    return float(np.trace(confusion) / confusion.sum())

def macro_f1(confusion: np.ndarray) -> float:
    """the mean of the symbols' F1 scores, so every symbol counts the same;
    F1 of symbol u = 2 · (runs right) / (runs sent as u + runs predicted as u)"""
    right = np.diag(confusion)
    sent = confusion.sum(axis=1)        # row sums: runs where u was sent
    predicted = confusion.sum(axis=0)   # column sums: runs where u was predicted
    return float(np.mean(2 * right / (sent + predicted)))