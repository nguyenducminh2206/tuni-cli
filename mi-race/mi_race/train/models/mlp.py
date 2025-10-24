from __future__ import annotations

from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier


def run_mlp(feature_df: pd.DataFrame,
            y: np.ndarray,
            train_cfg: dict,
            model_cfg: dict,
            standardize: bool,
            random_state: int,
            stratify,
            full_counts: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """Train an MLP and return (y_test, y_pred)."""
    X = feature_df.to_numpy()

    test_size = float(train_cfg.get("test_size", 0.2))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    train_counts = pd.Series(y_train).value_counts().sort_index()
    test_counts  = pd.Series(y_test).value_counts().sort_index()
    print(f"[mi-race][mlp] Total rows: {feature_df.shape[0]}")
    print(f"[mi-race][mlp] Class distribution (train): {train_counts.to_dict()}")
    print(f"[mi-race][mlp] Class distribution (test):  {test_counts.to_dict()}")

    hidden_layers = tuple(model_cfg.get("hidden_layers", [128, 128]))
    activation = model_cfg.get("activation", "relu")
    alpha = float(model_cfg.get("alpha", 1e-4))
    max_iter = int(train_cfg.get("max_iter", 200))

    steps = []
    if standardize:
        steps.append(("scaler", StandardScaler()))
    steps.append((
        "clf",
        MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation=activation,
            alpha=alpha,
            max_iter=max_iter,
            random_state=random_state
        )
    ))
    pipe = Pipeline(steps)

    pipe.fit(X_train, y_train)

    learned_classes = pipe.named_steps["clf"].classes_.tolist()
    print(f"[mi-race][mlp] Model learned classes: {learned_classes}")
    missing_classes = [c for c in full_counts.index if c not in learned_classes]
    if missing_classes:
        print(f"[mi-race][WARN][mlp] Classes missing from training set: {missing_classes} -> cannot be predicted.")

    y_pred = pipe.predict(X_test)
    return y_test, y_pred
