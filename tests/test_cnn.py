"""tests for src/mi_race/decoders/cnn.py"""

import torch
from torch import nn
import pytest
import numpy as np

from mi_race.decoders.cnn import make_cnn, GlobalScaler


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