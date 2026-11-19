"""compare: codebooks side by side under every decoder, and the noise of a search's feedback
the tested pieces behind the guided search, '[mi-race] compare' and '[mi-race] final'"""

from dataclasses import replace

import numpy as np

from mi_race.config import DECODERS, Config, _parse_symbols
from mi_race.dataset import simulate_dataset, split_dataset
from mi_race.decoders.rescaling import train_rescaled
from mi_race.metrics import evaluate
from mi_race.search import CHECKS, TRAINERS
from mi_race.streams import dataset_seed, random_stream

NOISE_PAIRS = 20        # paired datasets in a noise measurement: the se is the known to about ±16%
NOISE_RUN = 999         # the search run whose datasets measure the noise: no search uses it


def evaluate_codebook(base_config: Config, symbols, seed: int, rescaled: bool = True) -> dict:
    """every decoder of the config (all three if it names none) on one dataset of this codebook: simulate,
    one stratified split, train each decoder on the training runs (rescaled: T fitted on a fifth of them),
    evaluate on the test runs. Returns {decoder: its metrics, with 'temperature' (1.0 when not rescaled)}"""
    encoder = base_config.encoder
    checked = _parse_symbols([list(v) for v in symbols], encoder.n_slots, encoder.budget)     # the config's own rules
    cfg = replace(base_config, seed=seed, encoder=replace(encoder, symbols=checked))
    rows = simulate_dataset(cfg)
    train, test = split_dataset(rows, cfg.data.test_fraction, random_stream(seed, "split"))
    scores = {}
    for name in base_config.decoders or TRAINERS:
        settings = base_config.decoders.get(name, DECODERS[name]())
        if rescaled:
            model = train_rescaled(TRAINERS[name], train[:, 1:], train[:, 0], settings, random_stream(seed, name))
        else:
            model = TRAINERS[name](train[:, 1:], train[:, 0], settings, random_stream(seed, name))
        scores[name] = evaluate(test[:, 0], model.predict_proba(test[:, 1:])) | {
            "temperature": getattr(model, "temperature", 1.0)}
    return scores



def compare_codebooks(base_config: Config, codebooks: dict, job: str = "check_data", n_datasets: int = CHECKS,
                      rescaled: bool = True, progress=None) -> dict:
    """every codebook under every decoder on the same datasets of a job ('check_data' while choosing,
    'final_data' once at the end), so they can be compared dataset by dataset. A codebook given twice is
    scored once. Returns {label: [its scores on dataset 0, 1, ...]}, in the order given.
    progress (optional) is called after each dataset with (done, total, stats, detail)"""
    groups = {}                                                     # codebook -> the labels it was given under
    for label, symbols in codebooks.items():
        groups.setdefault(tuple(tuple(v) for v in symbols), []).append(label)
    scores, done, total = {}, 0, len(groups) * n_datasets
    for symbols, labels in groups.items():
        results = []
        for i in range(n_datasets):
            results.append(evaluate_codebook(base_config, symbols, dataset_seed(base_config.seed, job, i), rescaled))
            done += 1
            if progress:
                progress(done, total, "", f"{labels[0]} · dataset {i + 1} of {n_datasets}")
        for label in labels:
            scores[label] = results
    return {label: scores[label] for label in codebooks}


def paired_gain(scores: dict, label: str, reference: str, decoder: str, metric: str = "ba_bound_bits") -> dict:
    """label minus reference, dataset by dataset (the same datasets, so the difference is paired):
    the mean gain, its standard error, and on how many datasets label is higher"""
    gains = np.array([a[decoder][metric] - b[decoder][metric] for a, b in zip(scores[label], scores[reference])])
    se = float(gains.std(ddof=1) / np.sqrt(len(gains))) if len(gains) > 1 else float("nan")
    return {"mean": float(gains.mean()), "se": se, "higher": int((gains > 0).sum()), "datasets": len(gains)}


def neutral_move(symbols, budget: int) -> tuple:
    """an almost neutral move, to measure noise with: a fifth of one symbol's budget moved one slot along,
    from its biggest release (with budget 50: 10 molecules, as in ENCODER_PLAN section 7). Symbol 1 is tried
    first, then the others, until the moved codebook still has every symbol different. Returns that codebook"""
    symbols = tuple(tuple(v) for v in symbols)
    n_slots, amount = len(symbols[0]), max(1, budget // 5)
    if n_slots < 2:
        raise ValueError("a move needs at least 2 slots")
    for u in dict.fromkeys([1, *range(len(symbols))]):           # symbol 1 first, then 0, 2, 3, ...
        if u >= len(symbols):
            continue
        vector = list(symbols[u])
        k = max(range(n_slots), key=lambda slot: vector[slot])     # its biggest release
        target = k + 1 if k + 1 < n_slots else k - 1
        vector[k] -= amount
        vector[target] += amount
        moved = symbols[:u] + (tuple(vector),) + symbols[u + 1:]
        if vector[k] >= 0 and len(set(moved)) == len(moved):
            return moved
    raise ValueError("no symbol can move without becoming equal to another")


def measure_noise(base_config: Config, score, symbols=None, pairs: int = NOISE_PAIRS, progress=None) -> dict:
    """the noise of one paired comparison under a search's feedback (score): an almost neutral move and its
    parent, scored on the same datasets of a search run no search uses. The proposed threshold is 2 × the
    sd of the differences, to 0.01 bits, so a neutral move passes only about 3% of the time.
    progress (optional) is called after each pair with (done, total, stats)"""
    parent = tuple(tuple(v) for v in (symbols if symbols is not None else base_config.encoder.symbols))
    moved = neutral_move(parent, base_config.encoder.budget)
    differences = []
    for i in range(pairs):
        seed = dataset_seed(base_config.seed, "search_data", NOISE_RUN, i)
        differences.append(score(base_config, moved, seed)["ba_bound_bits"] - score(base_config, parent, seed)["ba_bound_bits"])
        if progress:
            progress(i + 1, pairs, f"sd {np.std(differences, ddof=1):.4f} bits" if i else "")
    sd = float(np.std(differences, ddof=1))
    return {"parent": parent, "moved": moved, "differences": differences, "mean": float(np.mean(differences)),
            "sd": sd, "threshold": round(2 * sd, 2)}