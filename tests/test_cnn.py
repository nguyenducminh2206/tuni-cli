"""tests for src/mi_race/decoders/cnn.py"""

import torch
from torch import nn

from mi_race.decoders.cnn import make_cnn


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