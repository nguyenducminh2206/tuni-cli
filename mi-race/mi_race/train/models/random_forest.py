from __future__ import annotations

from typing import Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def _normalize_labels_for_sklearn(y: np.ndarray) -> np.ndarray:
    """Coerce labels into a discrete form for sklearn classifiers.

    sklearn rejects non-integer float labels as "continuous".
    - integer-like floats -> int
    - other floats -> string
    """
    arr = np.asarray(y)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    if arr.dtype.kind == "f":
        finite = np.isfinite(arr)
        if finite.all() and np.allclose(arr, np.round(arr)):
            return np.round(arr).astype(int)
        return arr.astype(str)
    return arr


def run_random_forest(
    feature_df: pd.DataFrame,
    y: np.ndarray,
    train_cfg: dict,
    model_cfg: dict,
    standardize: bool,  # Ignored for RF; kept for API parity
    random_state: int,
    stratify,
    quiet: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Train a RandomForest classifier and return (y_test, y_pred, y_proba).

    Notes:
    - RandomForest does not require feature standardization; 'standardize' is ignored.
    - Supports common hyperparameters via model_cfg.
    - ``quiet=True`` suppresses the per-call progress lines (used for per-split runs).
    """
    X = feature_df.to_numpy()
    y_norm = _normalize_labels_for_sklearn(y)
    stratify_norm = _normalize_labels_for_sklearn(stratify) if stratify is not None else None

    test_size = float(train_cfg.get("test_size", 0.2))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_norm, test_size=test_size, random_state=random_state, stratify=stratify_norm
    )

    if not quiet:
        train_counts = pd.Series(y_train).value_counts().sort_index()
        test_counts = pd.Series(y_test).value_counts().sort_index()
        print(f"[rf] {feature_df.shape[0]} rows  ·  train {dict((str(k), int(v)) for k, v in train_counts.items())}")
        print(f"[rf] test  {dict((str(k), int(v)) for k, v in test_counts.items())}")

    # Hyperparameters
    n_estimators = int(model_cfg.get("n_estimators", 200))
    max_depth = model_cfg.get("max_depth")
    if max_depth is not None:
        max_depth = int(max_depth)
    # Handle max_features; 'auto' is invalid in newer sklearn, map to 'sqrt'
    max_features = model_cfg.get("max_features", "sqrt")
    if isinstance(max_features, str):
        mf = max_features.lower()
        if mf == "auto":
            max_features = "sqrt"
        elif mf == "none":
            max_features = None
    min_samples_split = int(model_cfg.get("min_samples_split", 2))
    min_samples_leaf = int(model_cfg.get("min_samples_leaf", 1))
    bootstrap = bool(model_cfg.get("bootstrap", True))
    criterion = str(model_cfg.get("criterion", "gini"))  # 'gini' or 'entropy' or 'log_loss'
    class_weight = model_cfg.get("class_weight", None)  # e.g., 'balanced'
    n_jobs = int(model_cfg.get("n_jobs", -1))

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        bootstrap=bootstrap,
        criterion=criterion,
        class_weight=class_weight,
        n_jobs=n_jobs,
        random_state=random_state,
    )

    clf.fit(X_train, y_train)
    if not quiet:
        learned_classes = getattr(clf, "classes_", None)
        if learned_classes is not None:
            print(f"[rf] learned classes: {[str(c) for c in learned_classes]}")

    y_pred = clf.predict(X_test)
    # RF supports predict_proba when criterion is classification
    try:
        y_proba = clf.predict_proba(X_test)
    except Exception:
        y_proba = None
    return y_test, y_pred, y_proba
