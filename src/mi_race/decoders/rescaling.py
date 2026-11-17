"""rescaling: soften (or sharpen) a decoder's probabilities with 1 number, the temperature"""

import numpy as np

from mi_race.metrics import log_loss

TEMPERATURES = np.geomspace(0.05, 20, 401)          # the temperatures fit_temperature tries: each 1.5% above the last, 1 included

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


def fit_temperature(probabilities: np.ndarray, sent: np.ndarray, temperatures: np.ndarray=TEMPERATURES) -> float:
    """the temperature whose rescaled probabilities have the lowest log-loss on these runs
    use runs the decoder did not train on: on its own training runs a decoder looks surer that it should"""
    losses = [log_loss(sent, rescale(probabilities, T)) for T in temperatures]
    return float(temperatures[int(np.argmin(losses))])