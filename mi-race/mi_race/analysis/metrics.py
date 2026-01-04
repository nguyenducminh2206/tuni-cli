"""
Information metrics computed from a confusion matrix.

All entropies are in bits (log base 2).
"""
from __future__ import annotations

from typing import Literal, Optional, Dict, Any

import numpy as np
from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors


def _safe_probs(counts: np.ndarray) -> np.ndarray:
    total = counts.sum()
    if total <= 0:
        # Return uniform zero-prob vector/matrix with same shape
        return np.zeros_like(counts, dtype=float)
    return counts.astype(float) / float(total)


def _entropy(p: np.ndarray) -> float:
    """Shannon entropy H(p) in bits. Accepts 1D probabilities; ignores zeros."""
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log2(p)).sum())


def _joint_entropy(P: np.ndarray) -> float:
    """Joint entropy H(X,Y) for joint distribution matrix P."""
    P = np.asarray(P, dtype=float)
    P = P[P > 0]
    if P.size == 0:
        return 0.0
    return float(-(P * np.log2(P)).sum())


def _mutual_information(P: np.ndarray) -> float:
    """Mutual information I(X;Y) for joint distribution matrix P."""
    P = np.asarray(P, dtype=float)
    # Marginals
    Px = P.sum(axis=1, keepdims=True)
    Py = P.sum(axis=0, keepdims=True)

    # Only sum where P>0
    nz = P > 0
    ratio = np.zeros_like(P)
    # Avoid division by zero by masking
    denom = Px @ Py  # outer product using broadcasting
    ratio[nz] = P[nz] / denom[nz]
    # Where denom is zero, ratio stays zero and excluded by nz mask
    return float((P[nz] * np.log2(ratio[nz])).sum())


def _normalized_mi(mi: float, Hx: float, Hy: float, mode: Literal["sqrt", "min", "max"] = "sqrt") -> float:
    if mi <= 0.0:
        return 0.0
    if mode == "sqrt":
        denom = np.sqrt(Hx * Hy)
    elif mode == "min":
        denom = min(Hx, Hy)
    elif mode == "max":
        denom = max(Hx, Hy)
    else:
        raise ValueError("mode must be one of: 'sqrt', 'min', 'max'")
    if denom <= 0.0:
        return 0.0
    val = mi / denom
    # guard for tiny numeric overshoots
    return float(min(max(val, 0.0), 1.0))


def info_from_confusion_matrix(cm: np.ndarray, labels: Optional[list] = None) -> Dict[str, Any]:
    """
    Compute information-theoretic metrics from a confusion matrix.

    Parameters
    ----------
    cm : np.ndarray (shape: [n_classes_true, n_classes_pred])
        Confusion matrix of counts.
    labels : Optional[list]
        Optional list of labels in the order used to build the confusion matrix.

    Returns
    -------
    dict with keys:
      - N: total samples (int)
      - labels: provided labels or None
      - H_true, H_pred, H_joint: entropies in bits (floats)
      - I: mutual information I(true;pred)
      - H_true_given_pred, H_pred_given_true: conditional entropies
      - NMI_sqrt, NMI_min, NMI_max: normalized mutual information variants
      - p_true, p_pred: marginal distributions
      - P: joint distribution matrix
    """
    if not isinstance(cm, np.ndarray):
        cm = np.asarray(cm)
    if cm.ndim != 2:
        raise ValueError("cm must be a 2D array")

    N = int(cm.sum())
    P = _safe_probs(cm)
    p_true = P.sum(axis=1)
    p_pred = P.sum(axis=0)

    H_true = _entropy(p_true)
    H_pred = _entropy(p_pred)
    H_joint = _joint_entropy(P)
    mi = H_true + H_pred - H_joint
    H_true_given_pred = H_joint - H_pred
    H_pred_given_true = H_joint - H_true

    return {
        "N": N,
        "labels": labels,
        "H_true": H_true,
        "H_pred": H_pred,
        "H_joint": H_joint,
        "I": mi,
        "H_true_given_pred": H_true_given_pred,
        "H_pred_given_true": H_pred_given_true,
        "NMI_sqrt": _normalized_mi(mi, H_true, H_pred, mode="sqrt"),
        "NMI_min": _normalized_mi(mi, H_true, H_pred, mode="min"),
        "NMI_max": _normalized_mi(mi, H_true, H_pred, mode="max"),
        "p_true": p_true.tolist(),
        "p_pred": p_pred.tolist(),
        "P": P.tolist(),
    }


def ksg_mi_labels_probs(y: np.ndarray, probs: np.ndarray, k: int = 5) -> float:
    """Kraskov MI between discrete labels and continuous probabilities.

    Notes:
    - KSG is designed for continuous variables; we embed integer labels as 1D continuous values.
    - We use L_inf (Chebyshev) norm for neighbor radius, following the KSG-1 variant.
    - Returns MI in bits.
    """
    y = np.asarray(y).reshape(-1, 1).astype(float)
    probs = np.asarray(probs, dtype=float)
    if y.shape[0] != probs.shape[0]:
        raise ValueError("y and probs must have the same number of samples")
    N = y.shape[0]
    if N <= k:
        return 0.0

    # Joint space
    X = np.concatenate([y, probs], axis=1)
    nn_joint = NearestNeighbors(n_neighbors=k + 1, metric="chebyshev").fit(X)
    d_joint, _ = nn_joint.kneighbors(X)
    eps = d_joint[:, k]

    # Marginals
    nn_x = NearestNeighbors(n_neighbors=N, metric="chebyshev").fit(y)
    nn_y = NearestNeighbors(n_neighbors=N, metric="chebyshev").fit(probs)
    d_x, _ = nn_x.kneighbors(y)
    d_y, _ = nn_y.kneighbors(probs)

    nx = np.empty(N, dtype=int)
    ny = np.empty(N, dtype=int)
    for i in range(N):
        nx[i] = int(np.sum(d_x[i, 1:] <= eps[i]))
        ny[i] = int(np.sum(d_y[i, 1:] <= eps[i]))

    mi_nats = digamma(k) - np.mean(digamma(nx + 1) + digamma(ny + 1)) + digamma(N)
    mi_bits = float(mi_nats / np.log(2))
    return max(mi_bits, 0.0)
