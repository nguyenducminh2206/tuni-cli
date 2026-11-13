"""encoder search: improve the codebook by moving molecules between slots, scored with the decoders of v1"""

from collections.abc import Callable
from dataclasses import replace

import numpy as np
from sklearn.model_selection import StratifiedKFold

from mi_race.config import Config, RandomForestConfig, _parse_symbols
from mi_race.dataset import simulate_dataset
from mi_race.decoders.forest import train_random_forest
from mi_race.metrics import evaluate
from mi_race.streams import random_stream


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


def score_codebook(base_config: Config, symbols, seed: int) -> dict:
    """every metrics of 1 codebook: simulate, score every run
    out-of-fold witht the random forest, floor the probabilities, evaluate
    seed is the dataset seed (dataset_seed): 2 codebooks scored with the same seed share their random numbers"""
    encoder = base_config.encoder
    checked = _parse_symbols([list(v) for v in symbols], encoder.n_slots, encoder.budget)   # config's own rules
    cfg = replace(base_config, seed=seed, encoder=replace(encoder, symbols=checked))
    rows = simulate_dataset(cfg)        # in memory: no CSV for a codebook that may be thrown away
    X, y = rows[:, 1:], rows[:, 0]
    settings = cfg.decoders.get("random_forest", RandomForestConfig())                      # the default if the config has no forest
    probabilities = out_of_fold_probabilities(X, y, train_random_forest, settings, random_stream(seed, "split"),
                                              random_stream(seed, "random_forest"))
    return evaluate(y, floor_probabilities(probabilities))