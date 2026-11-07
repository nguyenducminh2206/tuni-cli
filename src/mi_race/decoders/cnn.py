"""the CNN decoder: the rx values of a run as one signal over time, read by sliding filters (PyTorch)"""

from torch import nn


def make_cnn(n_inputs: int, channels: tuple[int, ...], kernel_size: int, 
             n_symbols: int) -> nn.Sequential:
    """(runs, n_inputs) -> one signal per run -> conv layers with ReLU -> one score per symbol"""
    layers = [nn.Unflatten(1, (1, n_inputs))]           # (runs, n_inputs) -> (runs, 1 channel, n_inputs)
    sizes = (1, *channels)
    for size_in, size_out in zip(sizes, sizes[1:]):
        layers += [nn.Conv1d(size_in, size_out, kernel_size, padding="same"), nn.ReLU()]
    layers += [nn.Flatten(), nn.Linear(channels[-1] * n_inputs, n_symbols)]     # every channel at every time step
    return nn.Sequential(*layers)