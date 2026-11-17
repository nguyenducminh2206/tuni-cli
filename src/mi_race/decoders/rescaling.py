"""rescaling: soften (or sharpen) a decoder's probabilities with 1 number, the temperature"""

import numpy as np

def rescale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """p^(1/T), normalised per run: T > 1 softens overconfident probabilities, T < 1 sharpens them, T = 1 changes nothing
    the most likely symbol never changes, and a probability of 0 stays 0"""
    if temperature <= 0:
        raise ValueError(f"the temperature must be above 0, got {temperature}")
    with np.errstate(divide="ignore"):              # log(0) = -inf is fine: exp(-inf) = 0
        logs = np.log(probabilities) / temperature
    logs -= logs.max(axis=1, keepdims=True)         # the largest becomes exp(0) = 1, so nothing overflows
    raised = np.exp(logs)
    return raised / raised.sum(axis=1, keepdims=True)