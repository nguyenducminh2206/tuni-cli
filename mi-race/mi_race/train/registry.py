"""Single source of truth for supported model types.

The registry stores lightweight :class:`ModelSpec` records keyed by short name
(``"mlp"``, ``"cnn"``, ``"rnn"``, ``"rf"``). Each spec carries an *adapter*
function with a uniform 10-argument signature so the orchestrator does not
need to branch by model type when calling the runner.

Adapter signature::

    adapter(
        df, feature_df, y, y_discrete,
        train_cfg, model_cfg,
        standardize, random_state,
        stratify, counts,
        quiet=False,
    ) -> (y_test, y_pred, y_proba)

``y_proba`` is ``None`` for models that don't expose calibrated probabilities
(cnn, rnn). Runners themselves are imported lazily inside each adapter to keep
the CLI fast.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd


# Uniform return type from any adapter.
AdapterResult = Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]


def _adapter_mlp(
    df: pd.DataFrame,
    feature_df: pd.DataFrame,
    y: np.ndarray,
    y_discrete: np.ndarray,
    train_cfg: dict,
    model_cfg: dict,
    standardize: bool,
    random_state: int,
    stratify,
    counts: pd.Series,
    quiet: bool = False,
) -> AdapterResult:
    from .models.mlp import run_mlp

    return run_mlp(
        feature_df, y, train_cfg, model_cfg, standardize, random_state, stratify, counts,
        quiet=quiet,
    )


def _adapter_cnn(
    df: pd.DataFrame,
    feature_df: pd.DataFrame,
    y: np.ndarray,
    y_discrete: np.ndarray,
    train_cfg: dict,
    model_cfg: dict,
    standardize: bool,
    random_state: int,
    stratify,
    counts: pd.Series,
    quiet: bool = False,
) -> AdapterResult:
    from .models.cnn import run_cnn

    y_test, y_pred = run_cnn(
        feature_df, y, train_cfg, model_cfg, standardize, random_state, stratify, quiet=quiet,
    )
    return y_test, y_pred, None


def _adapter_rf(
    df: pd.DataFrame,
    feature_df: pd.DataFrame,
    y: np.ndarray,
    y_discrete: np.ndarray,
    train_cfg: dict,
    model_cfg: dict,
    standardize: bool,
    random_state: int,
    stratify,
    counts: pd.Series,
    quiet: bool = False,
) -> AdapterResult:
    from .models.random_forest import run_random_forest

    return run_random_forest(
        feature_df, y_discrete, train_cfg, model_cfg, standardize, random_state, stratify,
        quiet=quiet,
    )


def _adapter_rnn(
    df: pd.DataFrame,
    feature_df: pd.DataFrame,
    y: np.ndarray,
    y_discrete: np.ndarray,
    train_cfg: dict,
    model_cfg: dict,
    standardize: bool,
    random_state: int,
    stratify,
    counts: pd.Series,
    quiet: bool = False,
) -> AdapterResult:
    from .models.rnn import run_rnn

    y_test, y_pred = run_rnn(
        df, y, train_cfg, model_cfg, random_state, stratify, quiet=quiet,
    )
    return y_test, y_pred, None


@dataclass(frozen=True)
class ModelSpec:
    name: str
    runner_path: str               # "module.path:function" — used by get_runner()
    adapter: Callable[..., AdapterResult]
    needs_features: bool           # True for mlp/cnn/rf; False for rnn (raw df)
    default_epochs: Optional[int]  # None means "don't print epochs" (rf)


_SPECS: Tuple[ModelSpec, ...] = (
    ModelSpec(
        name="mlp",
        runner_path="mi_race.train.models.mlp:run_mlp",
        adapter=_adapter_mlp,
        needs_features=True,
        default_epochs=15,
    ),
    ModelSpec(
        name="cnn",
        runner_path="mi_race.train.models.cnn:run_cnn",
        adapter=_adapter_cnn,
        needs_features=True,
        default_epochs=5,
    ),
    ModelSpec(
        name="rnn",
        runner_path="mi_race.train.models.rnn:run_rnn",
        adapter=_adapter_rnn,
        needs_features=False,
        default_epochs=5,
    ),
    ModelSpec(
        name="rf",
        runner_path="mi_race.train.models.random_forest:run_random_forest",
        adapter=_adapter_rf,
        needs_features=True,
        default_epochs=None,
    ),
)


MODEL_REGISTRY: Dict[str, ModelSpec] = {s.name: s for s in _SPECS}
SUPPORTED_MODELS: Tuple[str, ...] = tuple(MODEL_REGISTRY.keys())


def get_runner(mtype: str) -> Callable:
    """Lazy-import and return the runner function for ``mtype``.

    Raises ``KeyError`` if ``mtype`` is not a registered model.
    """
    spec = MODEL_REGISTRY[mtype]
    mod_path, func_name = spec.runner_path.split(":")
    mod = importlib.import_module(mod_path)
    return getattr(mod, func_name)


def resolve_epochs(spec: ModelSpec, model_cfg: dict, train_cfg: dict) -> Optional[int]:
    """Resolve the epochs value used only for the split-summary print line.

    Returns ``None`` for models that don't display an epoch count (e.g. rf).
    """
    if spec.default_epochs is None:
        return None
    if not isinstance(model_cfg, dict):
        return spec.default_epochs
    return int(model_cfg.get("epochs", train_cfg.get("epochs", spec.default_epochs)))
