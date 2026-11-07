"""the CNN decoder: the rx values of a run as one signal over time, read by sliding filters (PyTorch)"""

from torch import nn
import numpy as np


def make_cnn(n_inputs: int, channels: tuple[int, ...], kernel_size: int, 
             n_symbols: int) -> nn.Sequential:
    """(runs, n_inputs) -> one signal per run -> conv layers with ReLU -> one score per symbol"""
    layers = [nn.Unflatten(1, (1, n_inputs))]           # (runs, n_inputs) -> (runs, 1 channel, n_inputs)
    sizes = (1, *channels)
    for size_in, size_out in zip(sizes, sizes[1:]):
        layers += [nn.Conv1d(size_in, size_out, kernel_size, padding="same"), nn.ReLU()]
    layers += [nn.Flatten(), nn.Linear(channels[-1] * n_inputs, n_symbols)]     # every channel at every time step
    return nn.Sequential(*layers)


class GlobalScaler:
    """one mean and one spread for all values together, so the signal keeps its shape over time"""

    def fit(self, X: np.ndarray) -> "GlobalScaler":
        """learn the mean and the spread of all values in X"""
        self.mean_ = float(X.mean())
        self.scale_ = float(X.std()) or 1.0     # all values equal: no spread to divide by, as in StandardScaler
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """the same shift and stretch for every value, whatever its time step"""
        return (X - self.mean_) / self.scale_