"""Learned encoder — train an encoder and a decoder jointly through the channel.

The channel (e.g. SSA) is a stochastic simulation, not a differentiable
function, so gradients cannot flow from the decoder back into the encoder as
they do in the reference `Optimal_Encoder_for_Noisy_Channel` project. Instead
the encoder learns by trial and error (REINFORCE):

  * **Encoder = stochastic policy.** For each symbol ``s`` it keeps logits
    ``θ_s`` over the ``n_slots`` release slots. A transmission splits the
    symbol's molecule ``budget`` into ``quanta`` equal packets and drops each
    packet into a slot sampled from ``softmax(θ_s)``. ``quanta=1`` → one pulse
    per symbol; larger values let a symbol spread its molecules over slots.
    Slots that would release after ``T`` are masked out.
  * **Decoder = small 1D CNN** reading the observed compartments (``data.x_cols``)
    and predicting the symbol. Trained supervised (cross-entropy).
  * **Reward = log q(s | y)**, the decoder's log-probability of the true symbol
    on freshly simulated traces. Its mean gives the Barber–Agakov lower bound
    on mutual information, ``I(S;Y) ≥ H(S) + E[log q(S|Y)]``, so maximizing the
    reward maximizes a lower bound on the information the channel carries.

Each step: sample releases → simulate → decode → update the decoder
(supervised) and the encoder (policy gradient with a leave-one-out baseline and
an entropy bonus). After training, the most likely release pattern per symbol
is the learned codebook.
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field

import numpy as np

from mi_race.channel.registry import build_channel
from mi_race.encoder.codebook import symbols_as_vectors, vector_to_schedule
from mi_race.encoder.symbols import _round_preserving_sum, normalize_to_budget


OPTIMIZE_DEFAULTS: dict = {
    "steps": 400,            # training steps
    "per_symbol": 8,         # transmissions per symbol per step (batch = N * per_symbol)
    "quanta": 1,             # packets per symbol (1 = single pulse)
    "lr_encoder": 0.03,      # Adam learning rate for the encoder logits
    "lr_decoder": 0.002,     # Adam learning rate for the decoder CNN
    "entropy_coef": 0.03,    # entropy bonus — keeps the encoder exploring early on
    # "random" (default) breaks symmetry across every slot, so the decoder can tell
    # exploring symbols apart. "baseline" starts near the config's codebook but tends
    # to stay stuck there (all symbols explore the same empty slots, so none can win them).
    "init": "random",
    "init_mix": 0.5,         # with init=baseline: share of probability spread uniformly
    "log_every": 20,         # steps per logged point in the training history
    "eval_runs_per_symbol": None,  # runs per symbol for the final evaluation (None = channel value)
    "seed": 0,
}

_XCOL_RE = re.compile(r"^comp(\d+)_(\d+)$")


# ---------------------------------------------------------------------------
# What the decoder observes
# ---------------------------------------------------------------------------
def observed_compartments(cfg: dict, L: int) -> tuple[list[int], slice | None]:
    """Map ``data.x_cols`` to ``(compartment indices, time slice)``.

    ``"comp3_0:comp3_200"`` → ``([3], slice(0, 201))``; a list of ranges observes
    several compartments; no ``x_cols`` observes all ``L`` compartments.
    """
    x_cols = cfg.get("data", {}).get("x_cols")
    if x_cols is None:
        return list(range(L)), None
    items = [x_cols] if isinstance(x_cols, str) else list(x_cols)
    comps: set[int] = set()
    starts: list[int] = []
    ends: list[int] = []
    for item in items:
        parts = [p.strip() for p in str(item).split(":")]
        matches = [_XCOL_RE.match(p) for p in parts]
        if len(parts) > 2 or not all(matches):
            raise SystemExit(
                f"[mi-race] optimize: data.x_cols entry {item!r} must look like "
                "'compK_a:compK_b' or 'compK_t'."
            )
        ks = {int(m.group(1)) for m in matches}
        if len(ks) != 1:
            raise SystemExit(f"[mi-race] optimize: range {item!r} spans two compartments.")
        k = ks.pop()
        if not 0 <= k < L:
            raise SystemExit(f"[mi-race] optimize: compartment {k} out of range [0, {L - 1}].")
        comps.add(k)
        idx = [int(m.group(2)) for m in matches]
        starts.append(min(idx))
        ends.append(max(idx))
    return sorted(comps), slice(min(starts), max(ends) + 1)


# ---------------------------------------------------------------------------
# Encoder policy
# ---------------------------------------------------------------------------
class EncoderPolicy:
    """Per-symbol softmax over release slots (a learnable logit table).

    With one-hot symbol inputs, a neural encoder's first layer *is* a table
    lookup, so a logit table is the simplest exact parameterization.
    """

    def __init__(
        self,
        n_symbols: int,
        n_slots: int,
        valid_slots: np.ndarray,
        quanta: int = 1,
        init_vectors: np.ndarray | None = None,
        init_mix: float = 0.5,
        seed: int = 0,
    ):
        import torch

        self.n_symbols = int(n_symbols)
        self.n_slots = int(n_slots)
        self.quanta = int(quanta)
        valid = np.asarray(valid_slots, dtype=bool)
        self.mask = torch.as_tensor(valid)

        if init_vectors is not None:
            base = np.asarray(init_vectors, dtype=float) * valid
            rows = base.sum(axis=1, keepdims=True)
            base = np.divide(base, rows, out=np.zeros_like(base), where=rows > 0)
            uniform = valid / valid.sum()
            p = (1.0 - init_mix) * base + init_mix * uniform
            p = p / p.sum(axis=1, keepdims=True)
            logits = np.log(np.maximum(p, 1e-12))
        else:
            logits = np.random.default_rng(seed).normal(0.0, 0.5, (self.n_symbols, self.n_slots))
        self.logits = torch.tensor(logits, dtype=torch.float32, requires_grad=True)

    def _masked_logits(self):
        return self.logits.masked_fill(~self.mask, -1e9)

    def probs(self) -> np.ndarray:
        """Current release probabilities, shape ``(n_symbols, n_slots)``."""
        import torch

        with torch.no_grad():
            p = torch.softmax(self._masked_logits(), dim=1).double().numpy()
        return p / p.sum(axis=1, keepdims=True)

    def log_prob(self, symbol_idx, counts):
        """log π(counts | θ_s) up to the multinomial constant (differentiable)."""
        import torch

        logp = torch.log_softmax(self._masked_logits(), dim=1)[symbol_idx]
        return (counts * logp).sum(dim=1)

    def entropy(self):
        """Entropy of each symbol's slot distribution, in nats (differentiable)."""
        import torch

        logp = torch.log_softmax(self._masked_logits(), dim=1)
        return -(logp.exp() * logp).sum(dim=1)

    def mode_counts(self) -> np.ndarray:
        """Most likely packet allocation per symbol, shape ``(n_symbols, n_slots)``."""
        p = self.probs()
        return np.stack([_round_preserving_sum(self.quanta * row, self.quanta) for row in p])


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------
def make_trace_decoder(in_channels: int, n_steps: int, n_classes: int, hidden: int = 64):
    """Small 1D CNN: (batch, compartments, time) → symbol logits."""
    import torch
    from torch import nn

    conv = nn.Sequential(
        nn.Conv1d(in_channels, 16, kernel_size=5, padding=2), nn.ReLU(),
        nn.MaxPool1d(2, ceil_mode=True),
        nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.ReLU(),
        nn.MaxPool1d(2, ceil_mode=True),
    )
    with torch.no_grad():
        flat = conv(torch.zeros(1, in_channels, n_steps)).numel()
    return nn.Sequential(conv, nn.Flatten(), nn.Linear(flat, hidden), nn.ReLU(), nn.Linear(hidden, n_classes))


def _simulate_batch(channel, probs, per_symbol, quanta, budget, slot_dt, observed, t_slice, rng):
    """Sample releases from the policy and push them through the channel.

    Returns ``(labels[B], counts[B, n_slots], features[B, C, T'])`` in
    symbol-major order (all of symbol 0's samples first, then symbol 1, …).
    """
    n_symbols, n_slots = probs.shape
    labels = np.repeat(np.arange(n_symbols), per_symbol)
    counts = np.zeros((n_symbols * per_symbol, n_slots), dtype=np.float32)
    feats = []
    for row, i in enumerate(labels):
        c = rng.multinomial(quanta, probs[i])
        vec = normalize_to_budget(c, budget)
        _times, X = channel(vector_to_schedule(vec, slot_dt), rng)
        X = np.asarray(X)
        if t_slice is not None:
            X = X[t_slice]
        feats.append(np.log1p(X[:, observed].T.astype(np.float32)))  # log1p tames count noise
        counts[row] = c
    return labels, counts, np.stack(feats)


# ---------------------------------------------------------------------------
# Joint training
# ---------------------------------------------------------------------------
_TABLE = "  {:>9}  {:>8}  {:>9}  {:>8}  {:>12}  {:>6}"   # training progress rows (< 70 cols)


def _fmt_secs(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s}s" if s < 60 else f"{s // 60}m{s % 60:02d}s"


@dataclass
class TrainingResult:
    codebook: dict                   # {symbol_id: learned release vector}
    history: dict                    # logged curves + final policy
    observed: list = field(default_factory=list)


def train_encoder_decoder(cfg: dict, opts: dict | None = None, progress: bool = True) -> TrainingResult:
    """Train the encoder policy and a decoder jointly through ``cfg``'s channel."""
    import torch
    from sklearn.metrics import confusion_matrix

    from mi_race.analysis import info_from_confusion_matrix

    o = {**OPTIMIZE_DEFAULTS, **(opts or {})}
    steps, per_symbol, quanta = int(o["steps"]), int(o["per_symbol"]), int(o["quanta"])
    if per_symbol < 2:
        raise SystemExit("[mi-race] optimize: per_symbol must be >= 2 (needed for the baseline).")
    if quanta < 1:
        raise SystemExit("[mi-race] optimize: quanta must be >= 1.")

    ch = cfg["channel"]
    L, T = int(ch["L"]), float(ch["T"])
    slot_dt = float(ch.get("slot_dt", 0.1))
    base = symbols_as_vectors(ch)
    if not base:
        raise SystemExit(
            "[mi-race] optimize: channel.symbols is empty — build a starting "
            "codebook with `mi-race symbols` first (it sets N, n_slots and budget)."
        )
    sids = sorted(base)
    n_symbols = len(sids)
    n_slots = int(ch.get("n_slots") or len(base[sids[0]]))
    budget = int(ch.get("budget") or sum(base[sids[0]]))
    valid = np.array([k * slot_dt <= T + 1e-9 for k in range(n_slots)])
    observed, t_slice = observed_compartments(cfg, L)

    seed = int(o["seed"])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    channel = build_channel(ch)

    init_vectors = None
    if o["init"] == "baseline":
        init_vectors = np.array([(base[s] + [0] * n_slots)[:n_slots] for s in sids], dtype=float)
    elif o["init"] != "random":
        raise SystemExit("[mi-race] optimize: init must be 'baseline' or 'random'.")
    policy = EncoderPolicy(n_symbols, n_slots, valid, quanta, init_vectors, float(o["init_mix"]), seed)

    # Probe the channel once to size the decoder input.
    _t, X0 = channel([(0.0, budget)], np.random.default_rng(seed))
    n_rows = np.asarray(X0).shape[0]
    n_time = len(range(n_rows)[t_slice]) if t_slice is not None else n_rows
    decoder = make_trace_decoder(len(observed), n_time, n_symbols)

    enc_opt = torch.optim.Adam([policy.logits], lr=float(o["lr_encoder"]))
    dec_opt = torch.optim.Adam(decoder.parameters(), lr=float(o["lr_decoder"]))
    entropy_coef = float(o["entropy_coef"])
    log_every = max(1, int(o["log_every"]))
    max_bits = math.log2(n_symbols)
    ln2 = math.log(2.0)

    history: dict = {"step": [], "acc": [], "mi_cm": [], "mi_ba": [], "entropy_bits": [], "mode_slots": []}
    win_true: list[int] = []
    win_pred: list[int] = []
    win_logq: list[float] = []

    # Progress is a plain table (one row per logged window) rather than a live
    # bar: rows never overwrite each other, so narrow terminals and logs stay clean.
    if progress:
        print(f"[mi-race] chance: {100 / n_symbols:.1f}% accuracy · best possible: {max_bits:.2f} bits\n")
        print(_TABLE.format("step", "accuracy", "MI (bits)", "MI bound", "enc. entropy", "time"))
    t0 = time.monotonic()

    for step in range(1, steps + 1):
        labels, counts, feats = _simulate_batch(
            channel, policy.probs(), per_symbol, quanta, budget, slot_dt, observed, t_slice, rng
        )
        x = torch.from_numpy(feats)
        y = torch.from_numpy(labels).long()

        # Decoder: score the fresh batch (this is the reward), then learn from it.
        decoder.train()
        logits = decoder(x)
        logq = torch.log_softmax(logits, dim=1).gather(1, y[:, None]).squeeze(1)
        dec_opt.zero_grad()
        (-logq.mean()).backward()
        dec_opt.step()

        # Encoder: REINFORCE with a leave-one-out baseline per symbol.
        r = logq.detach().view(n_symbols, per_symbol)
        baseline = (r.sum(dim=1, keepdim=True) - r) / (per_symbol - 1)
        adv = (r - baseline).reshape(-1)
        adv = adv / (adv.std() + 1e-8)
        ent = policy.entropy()
        enc_loss = -(adv * policy.log_prob(y, torch.from_numpy(counts))).mean() - entropy_coef * ent.mean()
        enc_opt.zero_grad()
        enc_loss.backward()
        enc_opt.step()

        win_true.extend(labels.tolist())
        win_pred.extend(logits.detach().argmax(dim=1).tolist())
        win_logq.extend(logq.detach().tolist())

        if step % log_every == 0 or step == steps:
            cm = confusion_matrix(win_true, win_pred, labels=list(range(n_symbols)))
            acc = float(np.mean(np.array(win_true) == np.array(win_pred)))
            mi_cm = float(info_from_confusion_matrix(cm)["I"])
            mi_ba = max_bits + float(np.mean(win_logq)) / ln2
            h_bits = float(ent.detach().mean()) / ln2
            history["step"].append(step)
            history["acc"].append(acc)
            history["mi_cm"].append(mi_cm)
            history["mi_ba"].append(mi_ba)
            history["entropy_bits"].append(h_bits)
            history["mode_slots"].append(policy.probs().argmax(axis=1).tolist())
            win_true, win_pred, win_logq = [], [], []
            if progress:
                print(_TABLE.format(
                    f"{step}/{steps}", f"{acc * 100:.1f}%", f"{mi_cm:.3f}", f"{round(mi_ba, 3) + 0.0:.3f}",
                    f"{h_bits:.2f} bits", _fmt_secs(time.monotonic() - t0),
                ), flush=True)
    if progress:
        print(f"\n[mi-race] training done in {_fmt_secs(time.monotonic() - t0)}")

    mode = policy.mode_counts()
    codebook = {sid: normalize_to_budget(mode[i], budget) for i, sid in enumerate(sids)}
    history.update({
        "policy": policy.probs().tolist(),
        "max_bits": max_bits,
        "slot_dt": slot_dt,
        "observed": observed,
        "valid_slots": valid.tolist(),
        "symbol_ids": sids,
        "options": {k: o[k] for k in OPTIMIZE_DEFAULTS},
    })
    return TrainingResult(codebook=codebook, history=history, observed=observed)


# ---------------------------------------------------------------------------
# CLI entry — `mi-race optimize`
# ---------------------------------------------------------------------------
def run_optimize(args) -> None:
    """Train the encoder, then score baseline vs learned codebook and write a report."""
    import copy
    import json
    from pathlib import Path

    from mi_race.encoder.symbols import _compact_number_arrays, preview_symbols
    from mi_race.reporting.experiment_report import (
        _open_in_browser,
        _save_bundle,
        evaluate_codebook,
        render_html,
    )
    from mi_race.train.registry import SUPPORTED_MODELS

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise SystemExit(f"[mi-race] Config not found: {cfg_path}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if "channel" not in cfg or "data" not in cfg:
        raise SystemExit("[mi-race] optimize: config needs both 'channel' and 'data' sections.")

    model_name = getattr(args, "model", None) or "cnn"
    if model_name not in SUPPORTED_MODELS:
        raise SystemExit(
            f"[mi-race] optimize: unknown --model '{model_name}'. Use one of: {', '.join(SUPPORTED_MODELS)}."
        )

    # Precedence: CLI flag > config "optimize" block > defaults.
    opts = {**OPTIMIZE_DEFAULTS, **cfg.get("optimize", {})}
    for key in ("steps", "quanta", "per_symbol", "init", "seed"):
        val = getattr(args, key, None)
        if val is not None:
            opts[key] = val
    if getattr(args, "eval_runs", None) is not None:
        opts["eval_runs_per_symbol"] = args.eval_runs

    name = getattr(args, "name", None) or f"{cfg_path.stem}_optimized"
    out_dir = Path(getattr(args, "out", None) or (Path("experiments") / name))
    out_dir.mkdir(parents=True, exist_ok=True)
    ch = cfg["channel"]

    print(f"[mi-race] optimize: training encoder + decoder · channel={ch.get('type', 'ssa')}")
    print(f"[mi-race] {opts['steps']} steps · {opts['per_symbol']} sends per symbol per step · "
          f"quanta={opts['quanta']}")
    res = train_encoder_decoder(cfg, opts)
    h = res.history

    slot_dt = float(ch.get("slot_dt", 0.1))
    budget = int(ch.get("budget") or sum(next(iter(res.codebook.values()))))
    print("\nLearned codebook")
    preview_symbols(res.codebook, slot_dt, budget)

    # The learned codebook as a ready-to-use config (for generate-data / report).
    opt_cfg = copy.deepcopy(cfg)
    opt_cfg["channel"]["symbols"] = {str(k): [int(x) for x in v] for k, v in sorted(res.codebook.items())}
    opt_cfg["data"] = {**opt_cfg.get("data", {}), "path": f"data/{name}.csv"}
    opt_cfg_path = out_dir / "optimized_config.json"
    opt_cfg_path.write_text(_compact_number_arrays(json.dumps(opt_cfg, indent=2)) + "\n", encoding="utf-8")

    from mi_race.encoder.codebook import symbols_as_vectors

    eval_runs = opts.get("eval_runs_per_symbol")
    print(f"\n[mi-race] optimize: scoring both codebooks (fresh data + fresh {model_name}) …")
    baseline = evaluate_codebook(
        cfg, symbols_as_vectors(ch), model_name, out_dir / "baseline.csv", "Baseline (hand-picked)", eval_runs
    )
    optimized = evaluate_codebook(
        cfg, res.codebook, model_name, out_dir / "optimized.csv", "Optimized (learned)", eval_runs
    )

    results = [baseline, optimized]
    out_html = out_dir / "report.html"
    render_html(results, cfg, out_html, title=name, history=h)
    _save_bundle(results, cfg, out_dir, extra={"training": h})

    d_acc = (optimized.accuracy - baseline.accuracy) * 100
    d_mi = optimized.mi_bits - baseline.mi_bits
    print(f"\n{'':<12}{'baseline':>12}{'learned':>12}{'Δ':>10}")
    print(f"{'accuracy':<12}{baseline.accuracy * 100:>11.2f}%{optimized.accuracy * 100:>11.2f}%{d_acc:>+9.2f}%")
    print(f"{'MI (bits)':<12}{baseline.mi_bits:>12.3f}{optimized.mi_bits:>12.3f}{d_mi:>+10.3f}")
    print(f"\n[mi-race] optimize: wrote {out_html}")
    print(f"[mi-race] optimize: wrote {out_dir / 'result.json'}")
    print(f"[mi-race] optimize: wrote {opt_cfg_path}")
    print("          ↳ the learned codebook, ready for generate-data / report")
    if getattr(args, "open", False):
        _open_in_browser(out_html)
