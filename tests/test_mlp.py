"""tests for src/mi_race/decoders/mlp.py"""

import dataclasses
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn

from mi_race.config import MLPConfig, load_config
from mi_race.dataset import simulate_dataset, split_dataset
from mi_race.decoders.mlp import make_mlp, train_mlp
from mi_race.streams import random_stream

EXAMPLE = Path(__file__).parents[1] / "configs" / "example.json"
SMALL_MLP = MLPConfig(hidden_layers=(16,), epochs=20, lr=0.02, batch_size=16)  # small and fast


def linear_sizes(network):
    """(inputs, outputs) of every Linear layer, in order"""
    return [(layer.in_features, layer.out_features) for layer in network if isinstance(layer, nn.Linear)]


# 5d: the MLP network
def test_layer_sizes_of_the_example():
    assert linear_sizes(make_mlp(201, (128, 128), 4)) == [(201, 128), (128, 128), (128, 4)]


def test_one_hidden_layer():
    assert linear_sizes(make_mlp(201, (64,), 4)) == [(201, 64), (64, 4)]


def test_relu_between_layers_but_not_at_the_end():
    kinds = [type(layer) for layer in make_mlp(201, (128, 128), 4)]
    assert kinds == [nn.Linear, nn.ReLU, nn.Linear, nn.ReLU, nn.Linear]


def test_number_of_weights():
    n = sum(p.numel() for p in make_mlp(201, (128, 128), 4).parameters())
    assert n == (201 * 128 + 128) + (128 * 128 + 128) + (128 * 4 + 4)          # 42,884


def test_one_score_per_symbol_for_each_run():
    scores = make_mlp(201, (128, 128), 4)(torch.zeros(5, 201))
    assert scores.shape == (5, 4)


@pytest.fixture(scope="module")
def data():
    """the example config with 20 runs per symbol, split: X_train, y_train, X_test, y_test"""
    cfg = load_config(EXAMPLE)
    cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, runs_per_symbol=20))
    train, test = split_dataset(simulate_dataset(cfg), cfg.data.test_fraction, random_stream(cfg.seed, "split"))
    return train[:, 1:], train[:, 0], test[:, 1:], test[:, 0]


def mlp(data, seed=12345):
    """a small MLP decoder trained on the training rows"""
    X_train, y_train, _, _ = data
    return train_mlp(X_train, y_train, SMALL_MLP, random_stream(seed, "mlp"))


def test_one_probability_per_symbol(data):
    probabilities = mlp(data).predict_proba(data[2])
    assert probabilities.shape == (16, 4)                   # 4 test runs of each of the 4 symbols
    assert probabilities.sum(axis=1) == pytest.approx(1)    # every row adds up to 1


def test_network_comes_from_the_settings(data):
    assert linear_sizes(mlp(data).network) == [(201, 16), (16, 4)]


def test_one_loss_per_epoch(data):
    assert len(mlp(data).losses) == 20


def test_scaler_is_fitted_on_the_training_rows(data):
    X_train = data[0]
    assert mlp(data).scaler.mean_ == pytest.approx(X_train.mean(axis=0))


def test_same_stream_gives_the_same_decoder(data):
    assert np.array_equal(mlp(data).predict_proba(data[2]), mlp(data).predict_proba(data[2]))


def test_another_seed_gives_another_decoder(data):
    assert not np.array_equal(mlp(data, seed=12345).predict_proba(data[2]),
                              mlp(data, seed=54321).predict_proba(data[2]))


def test_learns_a_signal_that_is_easy_to_tell_apart():
    y = np.repeat(np.arange(4), 10)
    X = np.zeros((40, 41), dtype=int)
    X[np.arange(40), 10 * y] = 5                            # symbol u: 5 molecules in interval 10 u, nothing else
    decoder = train_mlp(X, y, SMALL_MLP, random_stream(0, "mlp"))
    assert decoder.predict_proba(X).argmax(axis=1).tolist() == y.tolist()