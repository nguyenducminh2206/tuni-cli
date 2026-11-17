"""rescaling: soften (or sharpen) a decoder's probabilities with 1 number, the temperature"""

import numpy as np
from collections.abc import Callable

from mi_race.metrics import log_loss
from mi_race.dataset import split_dataset

TEMPERATURES = np.geomspace(0.05, 20, 401)          # the temperatures fit_temperature tries: each 1.5% above the last, 1 included
HELD_OUT = 0.2                                      # share of the training runs held back to fit the temperature


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


class RescaledDecoder:
    """a trained decoder and its temperature: predict_proba gives the rescaled probabilities"""
    def __init__(self, decoder, temperature: float):
        self.decoder = decoder          # any trained decoder with predict_proba
        self.temperature = temperature  # fitted on runs the decoder did not train on

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """the decoder's probabilities, rescaled: the same guesses, honest confidence"""
        return rescale(self.decoder.predict_proba(X), self.temperature)

def train_rescaled(train_decoder: Callable, X: np.ndarray, y: np.ndarray, settings, rng: np.random.Generator,
                   held_out: float=HELD_OUT) -> RescaledDecoder:
    """hold back a share of each symbol's training runs, train the decoder on the rest,
    fit the temperature on the held-back runs, and return the 2 together.
    the random stream picks the held-back runs first, then trains the decoder"""
    if not 0 < held_out < 1:
        raise ValueError(f"held_out must be between 0 and 1, got {held_out}")
    numbered = np.column_stack([y, np.arange(len(y))])          # symbol, row number: split the row numbers
    fit_rows, held_rows = split_dataset(numbered, held_out, rng)
    fit, held = fit_rows[:, 1], held_rows[:, 1]
    decoder = train_decoder(X[fit], y[fit], settings, rng)
    temperature = fit_temperature(decoder.predict_proba(X[held]), y[held])
    return RescaledDecoder(decoder, temperature)