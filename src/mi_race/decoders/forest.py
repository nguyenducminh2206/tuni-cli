"""the random forest decoder (scikit-learn only, no PyTorch)"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from mi_race.config import RandomForestConfig


def train_random_forest(
        X: np.ndarray, y: np.ndarray, settings: RandomForestConfig, rng: np.random.Generator
) -> RandomForestClassifier:
    """the baseline decoder: a forest of decision trees on the raw counts, no scaling"""
    forest = RandomForestClassifier(
        n_estimators=settings.n_estimators,
        random_state=int(rng.integers(2**32)),          # sklearn wants whole number
    )
    return forest.fit(X, y)