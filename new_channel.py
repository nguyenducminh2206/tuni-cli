"""Custom mi-race channel — edit `simulate` with your own physics.

It must return (times, X):
  times : 1D array of snapshot times, length n_steps
  X     : 2D int array (n_steps, L); X[t, k] = molecules in compartment k at time t

Wire it up in your config:
  "channel": {
    "type": "custom",
    "impl": "new_channel.py:simulate",
    "L": 8, "dt": 0.01, "T": 2.0
  }
"""
import numpy as np


def simulate(schedule, cfg, rng):
    # schedule : list of (release_time, amount) injected into compartment 0
    # cfg      : the channel config dict — read your own params, e.g. cfg["my_param"]
    # rng      : numpy random Generator
    L = int(cfg["L"])
    dt = float(cfg["dt"])
    T = float(cfg["T"])
    n_steps = int(round(T / dt)) + 1
    times = np.linspace(0.0, T, n_steps)

    x = np.zeros(L, dtype=float)
    X = np.zeros((n_steps, L), dtype=int)
    releases = sorted(schedule)
    ptr = 0
    for i, t in enumerate(times):
        while ptr < len(releases) and releases[ptr][0] <= t:   # inject due releases
            x[0] += releases[ptr][1]
            ptr += 1
        # >>> your physics here: move molecules between compartments <<<
        X[i] = np.round(x).astype(int)
    return times, X
