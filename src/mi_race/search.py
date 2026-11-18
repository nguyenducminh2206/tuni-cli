"""encoder search: improve the codebook by moving molecules between slots, scored with the decoders of v1"""

from collections.abc import Callable
from dataclasses import replace
from functools import partial

import numpy as np
from sklearn.model_selection import StratifiedKFold

from mi_race.config import DECODERS, Config, _parse_symbols
from mi_race.dataset import simulate_dataset
from mi_race.decoders.cnn import train_cnn
from mi_race.decoders.forest import train_random_forest
from mi_race.decoders.mlp import train_mlp
from mi_race.decoders.rescaling import train_rescaled
from mi_race.metrics import evaluate
from mi_race.streams import random_stream, dataset_seed


FLOOR = 0.0025      # smallest prob a decoder may give in the search
FOLDS = 5           # every run is scored by a decoder trained on the other 4 folds
CHUNK = 10          # moles per move
THRESHOLD = 0.02    # bits a candidate must gain to be kept: about 2 * the spread of a paired diff (0.010)
CHECKS = 5          # check datasets per codebook: their mean has a spread of about 0.01 bits
TRAINERS = {"random_forest": train_random_forest, "mlp": train_mlp, "cnn": train_cnn}   # decoders a codebook can be scored with



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


def score_codebook(base_config: Config, symbols, seed: int, decoder: str="random_forest", rescaled: bool=False) -> dict:
    """every metric of 1 codebook: simulate, score every run out-of-fold with 1 decoder, floor the probabilities, evaluate
    rescaled: each fold's decoder also gets a temperature, fitted on runs held back from its own training runs
    seed is the dataset seed (dataset_seed): 2 codebooks scored with the same seed share their random numbers"""
    if decoder not in TRAINERS:
        raise ValueError(f"unknown decoder {decoder!r}: use one of {','.join(TRAINERS)}")
    encoder = base_config.encoder
    checked = _parse_symbols([list(v) for v in symbols], encoder.n_slots, encoder.budget)
    cfg = replace(base_config, seed=seed, encoder=replace(encoder, symbols=checked))
    rows = simulate_dataset(cfg)                                    # in memory: no csv for a codebook that may be thrown away
    X, y = rows[:, 1:], rows[:, 0]
    settings = cfg.decoders.get(decoder, DECODERS[decoder]())       # the defaults if the config has now section for it
    train_decoder = partial(train_rescaled, TRAINERS[decoder]) if rescaled else TRAINERS[decoder]
    probabilities = out_of_fold_probabilities(X, y, train_decoder, settings, random_stream(seed, "split"), random_stream(seed, decoder))
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
           chunk: int=CHUNK, threshold: float=THRESHOLD, progress=None):
    """hill climbing over codebooks: each step propose a legal move, scores the parent and the candidate on
    the same new dataset, and keeps the candidate only if its BA bound is higher by more then 'threshold'
    'run' picks the random streams (moves and datasets), do 2 runs are 2 independent searches.
    return the final codebook and the history, one dict per step.
    progress (optional) is called after each step with (done, total, stats, detail)"""
    rng = random_stream(base_config.seed, "search_moves", run)
    current = tuple(tuple(v) for v in start)
    history = []
    for step in range(steps):
        proposal = propose_move(current, rng, chunk)
        if proposal is None:                            # no legal move left
            if progress:
                progress(step, step, f"{sum(row['accepted'] for row in history)} kept · no legal move left")
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
        if progress:
            kept = sum(row["accepted"] for row in history)
            progress(step + 1, steps, f"{kept} kept · score {candidate_bits if accepted else parent_bits:.3f}",
                     f"last move {candidate_bits - parent_bits:+.3f} bits · {'kept' if accepted else 'not kept'}")
    return current, history


def check_codebook(base_config: Config, symbols, n_checks: int=CHECKS, score: Callable=score_codebook) -> dict:
    """the BA bound of a codebook on the check datasets (check_data 0 ... n_checks - 1), with their mean and spread
    every codebook is checked on the same datasets, so 2 checks can be compared run by run (paired)
    these are development numbers: use them to choose search settings, never as the final result"""
    if n_checks < 2:
        raise ValueError(f"n_checks must be at least 2 to measure a spread, got {n_checks}")
    seeds = [dataset_seed(base_config.seed, "check_data", i) for i in range(n_checks)]
    bits = [score(base_config, symbols, seed)["ba_bound_bits"] for seed in seeds]
    return {"bits": bits, "mean": float(np.mean(bits)), "sd": float(np.std(bits, ddof=1))}


def learning_curve(base_config: Config, start, history: list, every: int=10, n_checks:int=CHECKS,
                   score: Callable=score_codebook, progress=None) -> list:
    """check the start codebook and the codebook after every 'every' steps of a search history (and after the last step)
    all on the same check datasets. A codebook that was already checked is not checked again
    return 1 dict per point: the steps done, the moves accepted so far, and the check (bits, mean, sd).
    progress (optional) is called after each point with (done, total, stats)"""
    if every < 1:
        raise ValueError(f"every must be at least 1, got {every}")
    codebooks = [start] + [row["codebook"] for row in history]      # codebooks[k]: the codebook after k steps
    points = list(range(0, len(history) + 1, every))
    if points[-1] != len(history):
        points.append(len(history))                                 # always end with the last step
    checks = {}                                                     # codebook -> its check
    curve = []
    for k in points:
        codebook = tuple(tuple(v) for v in codebooks[k])            # tuples, so it can be a dict key
        if codebook not in checks:
            checks[codebook] = check_codebook(base_config, codebook, n_checks, score)
        accepted = sum(row["accepted"] for row in history[:k])
        curve.append({"steps": k, "accepted": accepted, **checks[codebook]})
        if progress:
            progress(len(curve), len(points), f"after {k} steps · {checks[codebook]['mean']:.3f} bits")
    return curve


