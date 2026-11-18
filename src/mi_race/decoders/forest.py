"""the random forest decoder (scikit-learn only, no PyTorch)"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from mi_race.config import RandomForestConfig


TREES_PER_STEP = 10         # with progress, the forest grows this many trees at a time


def train_random_forest(
        X: np.ndarray, y: np.ndarray, settings: RandomForestConfig, rng: np.random.Generator, progress=None
) -> RandomForestClassifier:
    """the baseline decoder: a forest of decision trees on the raw counts, no scaling.
    progress (optional) is called with (done, total) as the trees grow, a few at a time:
    scikit-learn's warm start gives exactly the same trees as growing them all at once"""
    forest = RandomForestClassifier(
        n_estimators=settings.n_estimators,
        random_state=int(rng.integers(2**32)),          # sklearn wants whole number
    )
    if not progress:
        return forest.fit(X, y)
    forest.set_params(warm_start=True)                  # keep the trees grown so far, add new ones
    for trees in range(TREES_PER_STEP, settings.n_estimators + TREES_PER_STEP, TREES_PER_STEP):
        forest.set_params(n_estimators=min(trees, settings.n_estimators)).fit(X, y)
        progress(len(forest.estimators_), settings.n_estimators)
    return forest.set_params(warm_start=False)