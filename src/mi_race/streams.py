"""random streams: one master seed gives every job its own independent random numbers"""

import numpy as np

# each job has a fixed number. Add new jobs at the end and never renumber, or old experiments could no longer be repeated
JOBS = {"simulation": 0, "split": 1, "random_forest": 2, "mlp": 3, "cnn": 4,
        "search_moves": 5,  # encoder search: which moles to move, one stream per search run
        "search_data": 6,   # encoder search: the dataset that compares parent and candidate, one per step
        "check_data": 7,    # development checks: comparing search settings, learning curves
        "final_data": 8,    # final evaluation: used once, after every choice is frozen
        }


def random_stream(seed: int, job: str, *indices: int) -> np.random.Generator:
    """the random generator of one job: the same seed, job and indices always give the same numbers
    for ex: run 7 of symbol 2: random_stream(cfg.seed, "simulation", 2, 7)"""
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(JOBS[job], *indices)))


def dataset_seed(seed: int, job: str, *indices: int) -> int:
    """the master seed of 1 simulated dataset: its runs, split and decoder all follow from it, as from cfg.seed
    for ex: the dataset of step 7 in search run 2: dataset_seed(cfg.seed, "search_data", 2, 7)"""
    return int(random_stream(seed, job, *indices).integers(2**63))