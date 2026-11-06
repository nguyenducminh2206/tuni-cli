"""tests for src/mi_race/decoders/mlp.py"""

import torch
from torch import nn

from mi_race.decoders.mlp import make_mlp


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