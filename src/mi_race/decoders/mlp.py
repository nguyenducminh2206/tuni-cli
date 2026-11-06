"""the MLP decoder: a fully connected network, each rx_m is one input (PyTorch)"""

from torch import nn


def make_mlp(n_inputs: int, hidden_layers: tuple[int, ...], n_symbols: int) -> nn.Sequential:
    """layers n_inputs -> hidden_layers ... -> n_symbols with ReLU in between;
    the output is one score per symbol"""
    sizes = (n_inputs, *hidden_layers)
    layers = []
    for size_in, size_out in zip(sizes, sizes[1:]):
        layers += [nn.Linear(size_in, size_out), nn.ReLU()]
    layers.append(nn.Linear(sizes[-1], n_symbols))          # no ReLU at the end: scores can be negative
    return nn.Sequential(*layers)