"""encoder search: improve the codebook by moving molecules between slots, scored with the decoders of v1"""

import numpy as np
from sklearn.model_selection import StratifiedKFold

from collections.abc import Callable


FLOOR = 0.0025      # smallest prob a decoder may give in the search
FOLDS = 5           # every run is scored by a decoder trained on the other 4 folds


def floor_probabilities(probabilities: np.ndarray, floor: float = FLOOR) -> np.ndarray:
    """raise every prob below 'floor' up to 'floor', then rescal each row so it adds up to 1 again
    q_k = max(p_k, floor) / sum_j max(p_j, floor). No symbol is ever called impossible, and q is still
    a probability for every symbol, so the BA bound computed from q stays a true lower bound"""
    raised = np.maximum(probabilities, floor)       # new array
    return raised / raised.sum(axis=1, keepdims=True)


def out_of_fold_probabilities(X: np.ndarray, y: np.ndarray, train_decoder: Callable, settings,
                              fold_rng: np.random.Generator, decoder_rng: np.random.Generator,
                              n_folds: int = FOLDS) -> np.ndarray:
    """a probability for every symbol for every run, each from a decoder that did not train on that run.
    The runs are cut into stratified folds (the same number of runs of each symbol in every fold);
    the runs of one fold are scored by a decoder trained on all the other folds"""
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=int(fold_rng.integers(2**32)))
    probabilities = np.zeros((len(y), int(y.max()) + 1))
    for train_rows, held_out_rows in folds.split(X, y):
        model = train_decoder(X[train_rows], y[train_rows], settings, decoder_rng)
        probabilities[held_out_rows] = model.predict_proba(X[held_out_rows])
    return probabilities