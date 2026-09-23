"""Tests for the learned encoder (`mi-race optimize`)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from mi_race.encoder.optimize import (
    EncoderPolicy,
    observed_compartments,
    run_optimize,
    train_encoder_decoder,
)

# A tiny, fast channel: 2 symbols, 5 slots, short tube and window.
TINY_CHANNEL = {
    "type": "ssa", "L": 3, "S": 1.0, "D": 3.0, "dt": 0.05, "T": 0.6,
    "runs_per_symbol": 12, "seed": 1,
    "n_slots": 5, "slot_dt": 0.1, "budget": 20,
    "symbols": {"0": [20, 0, 0, 0, 0], "1": [0, 0, 20, 0, 0]},
}


def _tiny_cfg(tmp_path: Path) -> dict:
    return {
        "channel": dict(TINY_CHANNEL),
        "data": {"path": str(tmp_path / "d.csv"), "x_cols": "comp1_0:comp1_12",
                 "y_col": "symbol", "sequence_mode": "split", "balance": False},
        "model": {"rf": {"n_estimators": 10, "random_state": 0}},
        "train": {"test_size": 0.25, "random_state": 0, "standardize": False},
        "output": {"dir": str(tmp_path / "outputs")},
    }


# --- what the decoder observes -------------------------------------------
def test_observed_single_range():
    assert observed_compartments({"data": {"x_cols": "comp3_0:comp3_200"}}, L=8) == ([3], slice(0, 201))


def test_observed_multiple_ranges():
    cfg = {"data": {"x_cols": ["comp4_0:comp4_10", "comp6_0:comp6_10"]}}
    assert observed_compartments(cfg, L=8) == ([4, 6], slice(0, 11))


def test_observed_defaults_to_all():
    assert observed_compartments({"data": {}}, L=5) == ([0, 1, 2, 3, 4], None)


def test_observed_rejects_out_of_range():
    with pytest.raises(SystemExit):
        observed_compartments({"data": {"x_cols": "comp9_0:comp9_10"}}, L=4)


# --- encoder policy -------------------------------------------------------
def test_baseline_init_favours_baseline_slot():
    valid = np.ones(5, dtype=bool)
    init = np.array([[20, 0, 0, 0, 0], [0, 0, 20, 0, 0]], dtype=float)
    p = EncoderPolicy(2, 5, valid, init_vectors=init, init_mix=0.5).probs()
    assert p[0].argmax() == 0 and p[1].argmax() == 2
    np.testing.assert_allclose(p.sum(axis=1), 1.0)


def test_masked_slots_get_zero_probability():
    valid = np.array([True, True, False, False])
    p = EncoderPolicy(3, 4, valid, seed=0).probs()
    assert np.all(p[:, 2:] < 1e-9)


def test_mode_counts_sum_to_quanta():
    policy = EncoderPolicy(3, 6, np.ones(6, dtype=bool), quanta=4, seed=1)
    assert (policy.mode_counts().sum(axis=1) == 4).all()


def test_policy_gradient_raises_probability_of_rewarded_slot():
    policy = EncoderPolicy(1, 4, np.ones(4, dtype=bool), init_vectors=np.ones((1, 4)))
    before = policy.probs()[0, 2]
    opt = torch.optim.SGD([policy.logits], lr=0.5)
    counts = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    (-policy.log_prob(torch.tensor([0]), counts).mean()).backward()  # positive advantage
    opt.step()
    assert policy.probs()[0, 2] > before


# --- training loop --------------------------------------------------------
def test_train_returns_budgeted_codebook_and_history(tmp_path):
    res = train_encoder_decoder(
        _tiny_cfg(tmp_path), {"steps": 4, "per_symbol": 3, "log_every": 2}, progress=False
    )
    assert set(res.codebook) == {0, 1}
    assert all(sum(v) == 20 and len(v) == 5 for v in res.codebook.values())
    assert res.history["step"] == [2, 4]
    assert len(res.history["policy"]) == 2 and len(res.history["policy"][0]) == 5
    assert res.observed == [1]


def test_train_never_uses_slots_after_T(tmp_path):
    cfg = _tiny_cfg(tmp_path)
    cfg["channel"]["n_slots"] = 10          # slots 7..9 (t = 0.7..0.9 s) exceed T = 0.6 s
    cfg["channel"]["symbols"] = {"0": [20] + [0] * 9, "1": [0, 0, 20] + [0] * 7}
    res = train_encoder_decoder(cfg, {"steps": 3, "per_symbol": 2, "log_every": 3}, progress=False)
    policy = np.array(res.history["policy"])
    assert np.all(policy[:, 7:] < 1e-9)
    assert all(sum(v[7:]) == 0 for v in res.codebook.values())


def test_per_symbol_must_allow_a_baseline(tmp_path):
    with pytest.raises(SystemExit):
        train_encoder_decoder(_tiny_cfg(tmp_path), {"steps": 1, "per_symbol": 1}, progress=False)


# --- end to end -----------------------------------------------------------
def test_run_optimize_writes_report_bundle_and_config(tmp_path):
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(_tiny_cfg(tmp_path)))
    out = tmp_path / "exp"
    args = argparse.Namespace(
        config=str(cfg_path), steps=3, quanta=None, per_symbol=2, init=None, seed=None,
        model="rf", eval_runs=12, name="tiny", out=str(out), open=False,
    )
    run_optimize(args)

    html = (out / "report.html").read_text()
    assert "Before / after" in html and "Encoder training" in html
    bundle = json.loads((out / "result.json").read_text())
    assert [r["label"] for r in bundle["results"]] == ["Baseline (hand-picked)", "Optimized (learned)"]
    assert bundle["training"]["step"][-1] == 3
    learned = json.loads((out / "optimized_config.json").read_text())
    assert all(sum(v) == 20 for v in learned["channel"]["symbols"].values())
    assert learned["data"]["path"] == "data/tiny.csv"
