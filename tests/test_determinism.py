"""5f: the decoders give the same numbers every time, and they don't affect each other"""

import dataclasses
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import torch

from mi_race.config import CNNConfig, MLPConfig, RandomForestConfig, load_config
from mi_race.dataset import simulate_dataset, split_dataset
from mi_race.decoders.cnn import train_cnn
from mi_race.decoders.forest import train_random_forest
from mi_race.decoders.mlp import train_mlp
from mi_race.streams import random_stream

TESTS = Path(__file__).parent
EXAMPLE = TESTS.parent / "configs" / "example.json"

# each decoder by its name in the config and in JOBS: its train function and small settings
DECODERS = {
    "random_forest": (train_random_forest, RandomForestConfig(n_estimators=10)),
    "mlp": (train_mlp, MLPConfig(hidden_layers=(16,), epochs=5, lr=0.02, batch_size=16)),
    "cnn": (train_cnn, CNNConfig(channels=(4,), kernel_size=3, epochs=5, lr=0.02, batch_size=16)),
}


def small_data():
    """40 runs of 30 random counts, 10 runs per symbol: X_train, y_train, X_test"""
    rng = np.random.default_rng(0)
    return rng.poisson(0.5, size=(40, 30)), np.repeat(np.arange(4), 10), rng.poisson(0.5, size=(8, 30))


def probabilities(name, seed=12345):
    """train one decoder on the small data with its own stream; its probabilities for the test runs"""
    X_train, y_train, X_test = small_data()
    train, settings = DECODERS[name]
    return train(X_train, y_train, settings, random_stream(seed, name)).predict_proba(X_test)


def fingerprint() -> str:
    """simulate, split and train every decoder from the example's seed; one hash of all the numbers"""
    cfg = load_config(EXAMPLE)
    cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, runs_per_symbol=10))
    rows = simulate_dataset(cfg)
    train, test = split_dataset(rows, cfg.data.test_fraction, random_stream(cfg.seed, "split"))
    numbers = hashlib.sha256(rows.tobytes())
    for name, (train_decoder, settings) in DECODERS.items():
        decoder = train_decoder(train[:, 1:], train[:, 0], settings, random_stream(cfg.seed, name))
        numbers.update(decoder.predict_proba(test[:, 1:]).tobytes())
    return numbers.hexdigest()


# 5f: seeding and determinism
def test_training_order_does_not_matter():
    first = {name: probabilities(name) for name in ["random_forest", "mlp", "cnn"]}
    reverse = {name: probabilities(name) for name in ["cnn", "mlp", "random_forest"]}
    for name in DECODERS:
        assert np.array_equal(first[name], reverse[name])


@pytest.mark.parametrize("name", DECODERS)
def test_global_random_numbers_do_not_matter(name):
    """only the decoder's own stream counts, not numpy's or PyTorch's global random numbers"""
    np.random.seed(1)
    torch.manual_seed(1)
    a = probabilities(name)
    np.random.seed(2)
    torch.manual_seed(2)
    np.random.random(100), torch.rand(100)                  # use up some global random numbers
    b = probabilities(name)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("name", ["mlp", "cnn"])
def test_networks_run_on_the_cpu(name):
    X_train, y_train, _ = small_data()
    train, settings = DECODERS[name]
    decoder = train(X_train, y_train, settings, random_stream(12345, name))
    assert {p.device.type for p in decoder.network.parameters()} == {"cpu"}


def test_a_fresh_python_process_gives_the_same_numbers():
    """this process has run other tests first; a new one starts clean, and both must agree to the last bit"""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(TESTS.parent / "src"), str(TESTS)])}
    fresh = subprocess.run([sys.executable, "-c", "import test_determinism; print(test_determinism.fingerprint())"],
                           env=env, capture_output=True, text=True, check=True)
    assert fresh.stdout.strip() == fingerprint()