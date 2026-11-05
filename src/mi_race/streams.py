"""random streams: one master seed gives every job its own independent random numbers"""

import numpy as np

# each job has a fixed number. Add new jobs at the end and never renumber, or old experiments could no longer be repeated
JOBS = {"simulation": 0, "split": 1, "random_forest": 2, "mlp": 3, "cnn": 4}


def random_stream(seed: int, job: str, *indices: int) -> np.random.Generator:
    """the random generator of one job: the same seed, job and indices always give the same numbers
    for ex: run 7 of symbol 2: random_stream(cfg.seed, "simulation", 2, 7)"""
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(JOBS[job], *indices)))