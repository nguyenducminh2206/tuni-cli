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

def hard_decision_mi(confusion: np.ndarray) -> float:
    """the plug-in estimate of I(U;Û) in bits, from the test confusion matrix;
    it reads too high on small test sets, by about mi_bias(N, n_test)"""
    joint = confusion / confusion.sum()                     # p(u, û)
    sent = joint.sum(axis=1, keepdims=True)                 # p(u), as a column
    predicted = joint.sum(axis=0, keepdims=True)            # p(û), as a row
    independent = sent * predicted                          # p(u) · p(û) for every cell
    seen = joint > 0                                        # 0 · log 0 = 0: the empty cells add nothing
    return float(np.sum(joint[seen] * np.log2(joint[seen] / independent[seen])))

def mi_bias(n_symbols: int, n_test: int) -> float:
    """roughly how many bits hard_decision_mi reads too high on n_test runs: (N-1)² / (2 · n_test · ln 2)"""
    return float((n_symbols - 1) ** 2 / (2 * n_test * np.log(2)))