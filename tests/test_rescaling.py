"""tests for src/mi_race/decoders/rescaling.py"""

import numpy as np
import pytest

from functools import partial

from mi_race.config import CNNConfig, MLPConfig, RandomForestConfig
from mi_race.decoders.cnn import train_cnn
from mi_race.decoders.mlp import train_mlp
from mi_race.decoders.forest import train_random_forest
from mi_race.decoders.rescaling import rescale, TEMPERATURES, fit_temperature, RescaledDecoder, train_rescaled
from mi_race.metrics import log_loss
from mi_race.search import out_of_fold_probabilities

P = np.array([[0.7, 0.2, 0.1],
              [0.98, 0.01, 0.01]])


# 6f part 1: rescale
def test_rescale_by_hand():
    roots = np.sqrt([0.7, 0.2, 0.1])                                    # T = 2: the square root of each, then normalise
    assert rescale(P[:1], 2)[0] == pytest.approx(roots / roots.sum())


def test_temperature_1_changes_nothing():
    assert rescale(P, 1) == pytest.approx(P)


def test_rows_add_up_to_1_and_the_most_likely_symbol_never_changes():
    p = np.random.default_rng(0).dirichlet(np.ones(4), size=50)         # 50 runs, 4 symbols
    for temperature in [0.1, 0.5, 2, 10]:
        q = rescale(p, temperature)
        assert q.sum(axis=1) == pytest.approx(np.ones(50))
        assert (q.argmax(axis=1) == p.argmax(axis=1)).all()


def test_above_1_softens_and_below_1_sharpens():
    assert (rescale(P, 3).max(axis=1) < P.max(axis=1)).all()
    assert (rescale(P, 0.5).max(axis=1) > P.max(axis=1)).all()


def test_extreme_temperatures_give_uniform_or_sure_probabilities():
    assert rescale(P, 1e9) == pytest.approx(np.full((2, 3), 1 / 3))
    assert rescale(P, 1e-4) == pytest.approx(np.array([[1, 0, 0], [1, 0, 0]]))    # 0.7^10000 alone would round to 0: no 0/0


def test_a_probability_of_0_stays_0():
    q = rescale(np.array([[0.6, 0.4, 0.0]]), 5)
    assert q[0, 2] == 0 and q[0, :2].sum() == pytest.approx(1)


def test_same_as_dividing_a_networks_scores_by_the_temperature():
    scores = np.array([[2.0, -1.0, 0.5], [0.0, 3.0, 1.0]])
    softmax = lambda z: np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    assert rescale(softmax(scores), 2.5) == pytest.approx(softmax(scores / 2.5))


def test_the_probabilities_passed_in_are_not_changed():
    p = P.copy()
    rescale(p, 2)
    assert (p == P).all()


@pytest.mark.parametrize("temperature", [0, -1])
def test_a_temperature_of_0_or_below_is_a_bug(temperature):
    with pytest.raises(ValueError, match="above 0"):
        rescale(P, temperature)



# 6f part 2: fit_temperature
def honest_decoder(n_runs=5000, seed=0):
    """probabilities that are exactly right on average: each run's sent symbol is drawn from its own probabilities"""
    rng = np.random.default_rng(seed)
    probabilities = rng.dirichlet(np.ones(4), size=n_runs)
    sent = np.array([rng.choice(4, p=p) for p in probabilities])
    return probabilities, sent


def test_an_honest_decoder_needs_no_rescaling():
    probabilities, sent = honest_decoder()
    assert fit_temperature(probabilities, sent) == pytest.approx(1, rel=0.1)


def test_an_overconfident_decoder_is_softened_by_the_right_amount():
    probabilities, sent = honest_decoder()
    overconfident = rescale(probabilities, 1 / 3)                      # 3 times too sure
    assert fit_temperature(overconfident, sent) == pytest.approx(3, rel=0.1)


def test_an_underconfident_decoder_is_sharpened():
    probabilities, sent = honest_decoder()
    underconfident = rescale(probabilities, 2)                         # half as sure as it should be
    assert fit_temperature(underconfident, sent) == pytest.approx(0.5, rel=0.1)


def test_the_fitted_temperature_is_never_worse_than_none():
    probabilities, sent = honest_decoder(n_runs=300, seed=1)
    overconfident = rescale(probabilities, 0.2)
    T = fit_temperature(overconfident, sent)
    assert log_loss(sent, rescale(overconfident, T)) <= log_loss(sent, overconfident)     # T = 1 is one of the tries


def test_the_temperature_is_one_of_the_tries():
    assert TEMPERATURES[0] == pytest.approx(0.05) and TEMPERATURES[-1] == pytest.approx(20)
    assert 1 in np.round(TEMPERATURES, 12)
    probabilities, sent = honest_decoder(n_runs=300, seed=2)
    assert fit_temperature(probabilities, sent) in TEMPERATURES


def test_a_sure_and_right_decoder_gets_the_sharpest_temperature():
    sure = np.array([[0.9, 0.1], [0.2, 0.8]])                         # right on both runs: sharper is always better
    assert fit_temperature(sure, np.array([0, 1])) == TEMPERATURES[0]


def test_a_decoder_that_is_always_wrong_gets_the_softest_temperature():
    wrong = np.array([[0.9, 0.1], [0.2, 0.8]])                        # wrong on both runs: closer to 50/50 is always better
    assert fit_temperature(wrong, np.array([1, 0])) == TEMPERATURES[-1]


def test_the_temperature_is_a_plain_number():
    probabilities, sent = honest_decoder(n_runs=300, seed=3)
    assert type(fit_temperature(probabilities, sent)) is float


# 6f part 3: train_rescaled
class SureOfZero:
    """a fake decoder: sure and right on the runs it trained on, 90% sure of symbol 0 on any other run"""

    def __init__(self, X, y):
        self.known = {tuple(x): u for x, u in zip(X, y)}

    def predict_proba(self, X):
        p = np.tile([0.9, 0.1 / 3, 0.1 / 3, 0.1 / 3], (len(X), 1))
        for i, x in enumerate(X):
            if tuple(x) in self.known:
                p[i] = np.eye(4)[self.known[tuple(x)]]
        return p


def sure_of_zero(X, y, settings, rng):
    """SureOfZero's train function, with the same arguments as the real ones"""
    return SureOfZero(X, y)


X_UNIQUE = np.arange(200).reshape(-1, 1)                           # 200 runs, no two alike
Y_UNIQUE = np.repeat(np.arange(4), 50)                              # 50 runs of each symbol


def test_the_temperature_is_fitted_on_runs_the_decoder_did_not_train_on():
    model = train_rescaled(sure_of_zero, X_UNIQUE, Y_UNIQUE, None, np.random.default_rng(0))
    assert isinstance(model, RescaledDecoder)
    assert model.temperature > 1            # on its own training runs it is always right, which would give 0.05


def test_20_percent_of_each_symbol_is_held_back():
    trained_on = []
    def record(X, y, settings, rng):
        trained_on.append(np.bincount(y).tolist())
        return SureOfZero(X, y)
    train_rescaled(record, X_UNIQUE, Y_UNIQUE, None, np.random.default_rng(0))
    assert trained_on == [[40, 40, 40, 40]]                         # 10 runs of each symbol held back


def test_the_rescaled_decoder_makes_the_same_guesses_with_rescaled_probabilities():
    model = train_rescaled(sure_of_zero, X_UNIQUE, Y_UNIQUE, None, np.random.default_rng(0))
    X_new = np.arange(200, 220).reshape(-1, 1)
    plain = model.decoder.predict_proba(X_new)
    assert model.predict_proba(X_new) == pytest.approx(rescale(plain, model.temperature))
    assert (model.predict_proba(X_new).argmax(axis=1) == plain.argmax(axis=1)).all()


SMALL = {"random_forest": (train_random_forest, RandomForestConfig(n_estimators=10)),
         "mlp": (train_mlp, MLPConfig(hidden_layers=(16,), epochs=5, lr=0.02, batch_size=16)),
         "cnn": (train_cnn, CNNConfig(channels=(4,), kernel_size=3, epochs=5, lr=0.02, batch_size=16))}


@pytest.mark.parametrize("name", SMALL)
def test_every_real_decoder_can_be_rescaled_and_the_same_stream_repeats(name):
    data = np.random.default_rng(0)
    X, y, X_new = data.poisson(0.5, size=(40, 30)), np.repeat(np.arange(4), 10), data.poisson(0.5, size=(8, 30))
    train, settings = SMALL[name]
    first = train_rescaled(train, X, y, settings, np.random.default_rng(1))
    second = train_rescaled(train, X, y, settings, np.random.default_rng(1))
    assert type(first.temperature) is float and first.temperature == second.temperature
    assert (first.predict_proba(X_new) == second.predict_proba(X_new)).all()
    assert first.predict_proba(X_new).sum(axis=1) == pytest.approx(np.ones(8))


def test_it_fits_where_a_train_function_fits():
    rescaled = partial(train_rescaled, sure_of_zero)                # the same arguments as train_random_forest & co
    p = out_of_fold_probabilities(X_UNIQUE, Y_UNIQUE, rescaled, None, np.random.default_rng(0), np.random.default_rng(1))
    assert p.shape == (200, 4) and p.sum(axis=1) == pytest.approx(np.ones(200))


@pytest.mark.parametrize("held_out", [0, 1, 1.5])
def test_a_held_out_share_outside_0_and_1_is_a_bug(held_out):
    with pytest.raises(ValueError, match="between 0 and 1"):
        train_rescaled(sure_of_zero, X_UNIQUE, Y_UNIQUE, None, np.random.default_rng(0), held_out=held_out)