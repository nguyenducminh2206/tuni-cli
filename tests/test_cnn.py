"""tests for src/mi_race/decoders/cnn.py"""

"""tests for src/mi_race/decoders/cnn.py"""

import dataclasses
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn

from mi_race.config import CNNConfig, load_config
from mi_race.dataset import simulate_dataset, split_dataset
from mi_race.decoders.cnn import GlobalScaler, make_cnn, train_cnn
from mi_race.streams import random_stream

EXAMPLE = Path(__file__).parents[1] / "configs" / "example.json"
SMALL_CNN = CNNConfig(channels=(4,), kernel_size=3, epochs=20, lr=0.02, batch_size=16)    # small and fast


def conv_sizes(network):
    """(input channels, output channels, kernel size) of every Conv1d layer, in order"""
    return [(layer.in_channels, layer.out_channels, layer.kernel_size[0]) for layer in network if isinstance(layer, nn.Conv1d)]


# 5e: the CNN network
def test_layers_of_the_example():
    kinds = [type(layer) for layer in make_cnn(201, (16, 32), 5, 4)]
    assert kinds == [nn.Unflatten, nn.Conv1d, nn.ReLU, nn.Conv1d, nn.ReLU, nn.Flatten, nn.Linear]


def test_conv_sizes_of_the_example():
    assert conv_sizes(make_cnn(201, (16, 32), 5, 4)) == [(1, 16, 5), (16, 32, 5)]


def test_one_conv_layer():
    assert conv_sizes(make_cnn(201, (8,), 3, 4)) == [(1, 8, 3)]


def test_last_layer_sees_every_channel_at_every_time_step():
    last = make_cnn(201, (16, 32), 5, 4)[-1]
    assert (last.in_features, last.out_features) == (32 * 201, 4)


def test_number_of_weights():
    n = sum(p.numel() for p in make_cnn(201, (16, 32), 5, 4).parameters())
    assert n == (1 * 16 * 5 + 16) + (16 * 32 * 5 + 32) + (32 * 201 * 4 + 4)    # 28,420


def test_one_score_per_symbol_for_each_run():
    scores = make_cnn(201, (16, 32), 5, 4)(torch.zeros(5, 201))
    assert scores.shape == (5, 4)


# 5e: the global scaler
def test_one_mean_and_spread_for_all_values():
    scaler = GlobalScaler().fit(np.array([[0, 4], [0, 4]]))
    assert (scaler.mean_, scaler.scale_) == (2, 2)


def test_columns_are_not_scaled_on_their_own():
    """StandardScaler would turn both columns into 0, since each column alone has no spread"""
    X = np.array([[0, 4], [0, 4]])
    assert GlobalScaler().fit(X).transform(X).tolist() == [[-1, 1], [-1, 1]]


def test_the_same_count_gets_the_same_value_at_every_time_step():
    X = np.random.default_rng(0).poisson(1.0, size=(50, 20))
    scaled = GlobalScaler().fit(X).transform(np.full((1, 20), 3))
    assert np.unique(scaled).size == 1


def test_scaled_training_values_have_mean_0_and_spread_1():
    X = np.random.default_rng(0).poisson(1.0, size=(50, 20))
    scaled = GlobalScaler().fit(X).transform(X)
    assert (scaled.mean(), scaled.std()) == (pytest.approx(0), pytest.approx(1))


def test_other_rows_use_the_mean_and_spread_of_the_fitted_rows():
    scaler = GlobalScaler().fit(np.array([[0, 4]]))         # mean 2, spread 2
    assert scaler.transform(np.array([[2, 2], [6, 0]])).tolist() == [[0, 0], [2, -1]]


def test_values_that_are_all_equal_are_not_divided_by_0():
    scaler = GlobalScaler().fit(np.zeros((3, 5), dtype=int))
    assert scaler.scale_ == 1
    assert scaler.transform(np.zeros((2, 5))).tolist() == [[0] * 5] * 2


# 5e: the CNN decoder
@pytest.fixture(scope="module")
def data():
    """the example config with 20 runs per symbol, split: X_train, y_train, X_test, y_test"""
    cfg = load_config(EXAMPLE)
    cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, runs_per_symbol=20))
    train, test = split_dataset(simulate_dataset(cfg), cfg.data.test_fraction, random_stream(cfg.seed, "split"))
    return train[:, 1:], train[:, 0], test[:, 1:], test[:, 0]


def cnn(data, seed=12345):
    """a small CNN decoder trained on the training rows"""
    X_train, y_train, _, _ = data
    return train_cnn(X_train, y_train, SMALL_CNN, random_stream(seed, "cnn"))


def test_one_probability_per_symbol(data):
    probabilities = cnn(data).predict_proba(data[2])
    assert probabilities.shape == (16, 4)                   # 4 test runs of each of the 4 symbols
    assert probabilities.sum(axis=1) == pytest.approx(1)    # every row adds up to 1


def test_network_comes_from_the_settings(data):
    network = cnn(data).network
    assert conv_sizes(network) == [(1, 4, 3)]
    assert network[-1].in_features == 4 * 201


def test_one_loss_per_epoch(data):
    assert len(cnn(data).losses) == 20


def test_one_scaler_for_all_training_values(data):
    X_train = data[0]
    scaler = cnn(data).scaler
    assert isinstance(scaler, GlobalScaler)
    assert scaler.mean_ == pytest.approx(X_train.mean())


def test_same_stream_gives_the_same_decoder(data):
    assert np.array_equal(cnn(data).predict_proba(data[2]), cnn(data).predict_proba(data[2]))


def test_another_seed_gives_another_decoder(data):
    assert not np.array_equal(cnn(data, seed=12345).predict_proba(data[2]),
                              cnn(data, seed=54321).predict_proba(data[2]))


def test_learns_a_signal_that_is_easy_to_tell_apart():
    y = np.repeat(np.arange(4), 10)
    X = np.zeros((40, 41), dtype=int)
    X[np.arange(40), 10 * y] = 5                            # symbol u: 5 molecules in interval 10 u, nothing else
    decoder = train_cnn(X, y, SMALL_CNN, random_stream(0, "cnn"))
    assert decoder.predict_proba(X).argmax(axis=1).tolist() == y.tolist()