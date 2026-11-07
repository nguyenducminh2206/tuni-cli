"""the MLP decoder: a fully connected network, each rx_m is one input (PyTorch)"""

from torch import nn
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from mi_race.config import MLPConfig
from mi_race.decoders.training import NetworkDecoder, train_network


def make_mlp(n_inputs: int, hidden_layers: tuple[int, ...], n_symbols: int) -> nn.Sequential:
    """layers n_inputs -> hidden_layers ... -> n_symbols with ReLU in between;
    the output is one score per symbol"""
    sizes = (n_inputs, *hidden_layers)
    layers = []
    for size_in, size_out in zip(sizes, sizes[1:]):
        layers += [nn.Linear(size_in, size_out), nn.ReLU()]
    layers.append(nn.Linear(sizes[-1], n_symbols))          # no ReLU at the end: scores can be negative
    return nn.Sequential(*layers)


def train_mlp(X: np.ndarray, y: np.ndarray, settings: MLPConfig, rng: np.random.Generator) -> NetworkDecoder:
    """the MLP decoder: each rx_m scaled on its own, then a fully connected network"""
    scaler = StandardScaler().fit(X)            # mean and spread of each rx_m, from these runs only
    torch.manual_seed(int(rng.integers(2**32))) # the starting weights come from the stream
    network = make_mlp(X.shape[1], settings.hidden_layers, n_symbols=int(y.max()) + 1)
    losses = train_network(network, scaler.transform(X), y, settings, rng)
    return NetworkDecoder(scaler, network, losses)