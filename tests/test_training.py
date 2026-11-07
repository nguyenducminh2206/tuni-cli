"""tests for src/mi_race/decoders/training.py"""

import dataclasses
import numpy as np
import pytest
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler

from mi_race.config import MLPConfig
from mi_race.decoders.mlp import make_mlp
from mi_race.decoders.training import train_network, NetworkDecoder
from mi_race.streams import random_stream

SETTINGS = MLPConfig(hidden_layers=(16,), epochs=20, lr=0.02, batch_size=16)    # small and fast


def easy_data():
    """40 runs of 10 inputs, 10 runs per symbol; input u is about 1 for symbol u, every other input is about 0"""
    y = np.repeat(np.arange(4), 10)
    X = np.random.default_rng(0).normal(0, 0.1, size=(40, 10))
    X[np.arange(40), y] += 1
    return X, y


def trained(settings=SETTINGS, seed=12345):
    """a small network trained on the easy data: (network, losses)"""
    X, y = easy_data()
    torch.manual_seed(0)                                        # the same starting weights every time
    network = make_mlp(10, settings.hidden_layers, 4)
    losses = train_network(network, X, y, settings, random_stream(seed, "mlp"))
    return network, losses


# 5d: the training loop
def test_one_loss_per_epoch():
    _, losses = trained()
    assert len(losses) == 20


def test_loss_goes_down():
    _, losses = trained()
    assert losses[-1] < losses[0] / 10


def test_learns_the_easy_data():
    network, _ = trained()
    X, y = easy_data()
    predicted = network(torch.as_tensor(X, dtype=torch.float32)).argmax(dim=1)
    assert predicted.tolist() == y.tolist()


def test_loss_is_the_mean_over_all_runs():
    """with one batch per epoch, the first loss is the loss of the untrained network on all runs"""
    X, y = easy_data()
    torch.manual_seed(0)
    network = make_mlp(10, (16,), 4)
    with torch.no_grad():
        expected = nn.functional.cross_entropy(network(torch.as_tensor(X, dtype=torch.float32)), torch.as_tensor(y))
    settings = dataclasses.replace(SETTINGS, batch_size=100)    # more than the 40 runs: one batch per epoch
    losses = train_network(network, X, y, settings, random_stream(12345, "mlp"))
    assert losses[0] == pytest.approx(expected.item())


def test_same_stream_gives_the_same_network():
    network_a, losses_a = trained()
    network_b, losses_b = trained()
    assert losses_a == losses_b
    for a, b in zip(network_a.parameters(), network_b.parameters()):
        assert torch.equal(a, b)


def test_another_stream_shuffles_differently():
    _, losses_a = trained(seed=1)
    _, losses_b = trained(seed=2)
    assert losses_a != losses_b


def test_deterministic_mode_is_on():
    trained()
    assert torch.are_deterministic_algorithms_enabled()



# 5d: the decoder around a trained network
def test_decoder_gives_probabilities():
    network, _ = trained()
    X, _ = easy_data()
    decoder = NetworkDecoder(StandardScaler().fit(X), network, losses=[])
    probabilities = decoder.predict_proba(X)
    assert probabilities.shape == (40, 4)
    assert probabilities.dtype == np.float64
    assert (probabilities >= 0).all()
    assert probabilities.sum(axis=1) == pytest.approx(1)


def test_decoder_scales_the_raw_counts():
    """with a network that changes nothing, the probabilities are the softmax of the scaled counts"""
    X = np.array([[0, 0], [2, 20]])                         # scaled: [-1, -1] and [1, 1]
    decoder = NetworkDecoder(StandardScaler().fit(X), nn.Identity(), losses=[])
    assert decoder.predict_proba(X) == pytest.approx(0.5)   # every probability is 0.5