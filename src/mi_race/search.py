"""encoder search: improve the codebook by moving molecules between slots, scored with the decoders of v1"""

from collections.abc import Callable
from dataclasses import replace

import numpy as np
from sklearn.model_selection import StratifiedKFold

from mi_race.config import Config, RandomForestConfig, _parse_symbols
from mi_race.dataset import simulate_dataset
from mi_race.decoders.forest import train_random_forest
from mi_race.metrics import evaluate
from mi_race.streams import random_stream, dataset_seed


FLOOR = 0.0025      # smallest prob a decoder may give in the search
FOLDS = 5           # every run is scored by a decoder trained on the other 4 folds
CHUNK = 10          # moles per move
THRESHOLD = 0.02    # bits a candidate must gain to be kept: about 2 * the spread of a paired diff (0.010)

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


def legal_moves(symbols, chunk: int = CHUNK) -> list[tuple[tuple[int, int, int], tuple[tuple[int, ...], ...]]]:
    """"every move of 'chunk' moles from 1 slot of 1 symbol to anothe slot, that keeps the codebook legal:
    the source slot has at least 'chunk' moles, and the changed symbol is not the same as another symbol
    each item is (move, new codebook) with move = (symbol, source slot, target slot); butget never changes
    """
    if chunk < 1:
        raise ValueError(f"chunk must be at least 1 moles, got {chunk}")
    codebook = [tuple(v) for v in symbols]
    moves = []
    for u, vector in enumerate(codebook):
        for source, amount in enumerate(vector):
            if amount < chunk:
                continue
            for target in range(len(vector)):
                if target == source:
                    continue
                changed = list(vector)
                changed[source] -= chunk
                changed[target] += chunk
                if tuple(changed) in codebook:
                    continue                    # same as another symbol
                new = codebook[:u] + [tuple(changed)] + codebook[u + 1:]
                moves.append(((u, source, target), tuple(new)))
    return moves


def propose_move(symbols, rng: np.random.Generator, chunk: int=CHUNK):
    """1 legal move picked at random, every legal move equally likely: (move, new codebook).
    or None if no move is legal"""
    moves = legal_moves(symbols, chunk)
    if not moves:
        return None
    return moves[rng.integers(len(moves))]


def search(base_config: Config, start, steps: int, run: int=0, score: Callable = score_codebook,
           chunk: int=CHUNK, threshold: float=THRESHOLD):
    """hill climbing over codebooks: each step propose a legal move, scores the parent and the candidate on
    the same new dataset, and keeps the candidate only if its BA bound is higher by more then 'threshold'
    'run' picks the random streams (moves and datasets), do 2 runs are 2 independent searches.
    return the final codebook and the history, one dict per step"""
    rng = random_stream(base_config.seed, "search_moves", run)
    current = tuple(tuple(v) for v in start)
    history = []
    for step in range(steps):
        proposal = propose_move(current, rng, chunk)
        if proposal is None:                            # no legal move left
            break
        move, candidate = proposal
        seed = dataset_seed(base_config.seed, "search_data", run, step)         # a new dataset, the same for both
        parent_bits = score(base_config, current, seed)["ba_bound_bits"]
        candidate_bits = score(base_config, candidate, seed)["ba_bound_bits"]
        accepted = candidate_bits - parent_bits > threshold
        if accepted:
            current = candidate
        history.append({"step": step, "seed": seed, "move": move, "parent_bits": parent_bits,
                        "candidate_bits": candidate_bits, "difference": candidate_bits - parent_bits,
                        "accepted": accepted, "codebook": current})
    return current, history