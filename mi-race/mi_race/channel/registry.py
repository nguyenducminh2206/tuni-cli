"""Pluggable channel layer.

A **channel** is any callable with this one signature::

    simulate(schedule, cfg, rng) -> (times, X)

      schedule : list[(t, amount)]     releases into compartment 0
      cfg      : the `channel` config dict (reads its own params: L, D, dt, T, …)
      rng      : numpy random Generator
      returns  : (times, X)
                   times : 1D array of snapshot times, length n_steps
                   X     : 2D int array (n_steps, L); X[t, k] = molecules in
                           compartment k at time t

Built-in channels are selected by ``channel.type``. ``type: "custom"`` loads a
user function via ``channel.impl = "path/to/file.py:function"`` — so users can
plug in their own physics without touching mi-race source.
"""
from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from typing import Callable

import numpy as np

from .simulation import simulate_ssa_with_schedule


ChannelFn = Callable[[list, dict, "np.random.Generator"], tuple]


# ---------------------------------------------------------------------------
# Built-in channels (thin wrappers over the physics engines)
# ---------------------------------------------------------------------------
def _ssa(schedule, cfg, rng):
    """Exact SSA diffusion with reflecting walls (mass-conserving)."""
    return simulate_ssa_with_schedule(
        schedule,
        int(cfg["L"]), float(cfg["S"]), float(cfg["D"]),
        float(cfg["dt"]), float(cfg["T"]), rng,
    )


def _ssa_absorbing(schedule, cfg, rng):
    """SSA diffusion with an absorbing receiver at the far end (mass drains → symbols finish)."""
    return simulate_ssa_with_schedule(
        schedule,
        int(cfg["L"]), float(cfg["S"]), float(cfg["D"]),
        float(cfg["dt"]), float(cfg["T"]), rng,
        absorbing=True,
    )


# type -> (function, one-line description, param names shown by `mi-race channels`)
_BUILTINS: dict[str, tuple[ChannelFn, str, list[str]]] = {
    "ssa": (_ssa, "exact SSA diffusion, reflecting walls (mass conserved)",
            ["L", "S", "D", "dt", "T"]),
    "ssa_absorbing": (_ssa_absorbing,
                      "SSA diffusion, absorbing receiver at far end (mass drains, symbols finish)",
                      ["L", "S", "D", "dt", "T"]),
}

CHANNEL_REGISTRY: dict[str, ChannelFn] = {name: fn for name, (fn, _d, _p) in _BUILTINS.items()}


def list_channels() -> list[tuple[str, str, list[str]]]:
    """Return ``[(name, description, param_names), ...]`` for built-in channels."""
    return [(name, desc, params) for name, (_fn, desc, params) in _BUILTINS.items()]


# ---------------------------------------------------------------------------
# Custom channel loading
# ---------------------------------------------------------------------------
def _load_custom(impl: str) -> ChannelFn:
    """Load ``"file.py:function"`` or ``"module.path:function"`` as a channel fn."""
    if not impl or ":" not in impl:
        raise SystemExit(
            "[mi-race] channel.type='custom' requires "
            "channel.impl = 'path/to/file.py:function'"
        )
    target, func_name = impl.rsplit(":", 1)
    p = Path(target)
    if p.exists():
        spec = importlib.util.spec_from_file_location("mi_race_custom_channel", p)
        if spec is None or spec.loader is None:
            raise SystemExit(f"[mi-race] could not load custom channel file: {p}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    else:
        try:
            mod = importlib.import_module(target)
        except ImportError as e:
            raise SystemExit(f"[mi-race] custom channel '{target}' not found: {e}")
    if not hasattr(mod, func_name):
        raise SystemExit(f"[mi-race] custom channel '{target}' has no function '{func_name}'")
    return getattr(mod, func_name)


def _validate_output(out, cfg) -> None:
    """Sanity-check a custom channel's return value; clear error on mismatch."""
    if not (isinstance(out, tuple) and len(out) == 2):
        raise SystemExit("[mi-race] channel must return a (times, X) tuple.")
    _times, X = out
    arr = np.asarray(X)
    if arr.ndim != 2:
        raise SystemExit(
            f"[mi-race] channel X must be 2D (n_steps, L); got shape {arr.shape}."
        )
    L = int(cfg.get("L", arr.shape[1]))
    if arr.shape[1] != L:
        raise SystemExit(
            f"[mi-race] channel X has {arr.shape[1]} compartments, expected L={L}."
        )


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------
def build_channel(channel_cfg: dict) -> Callable[[list, "np.random.Generator"], tuple]:
    """Return a bound ``channel(schedule, rng) -> (times, X)`` from a config block.

    ``channel.type`` picks a built-in (default ``"ssa"``); ``"custom"`` loads
    ``channel.impl``. Custom channels are output-validated on every call.
    """
    ctype = str(channel_cfg.get("type", "ssa"))
    if ctype == "custom":
        fn = _load_custom(channel_cfg.get("impl", ""))
        validate = True
    elif ctype in CHANNEL_REGISTRY:
        fn = CHANNEL_REGISTRY[ctype]
        validate = False
    else:
        raise SystemExit(
            f"[mi-race] unknown channel.type '{ctype}'. "
            f"Available: {sorted(CHANNEL_REGISTRY)} or 'custom'."
        )

    def channel(schedule, rng):
        out = fn(schedule, channel_cfg, rng)
        if validate:
            _validate_output(out, channel_cfg)
        return out

    return channel


# ---------------------------------------------------------------------------
# `mi-race channels` — list built-ins / scaffold a custom channel
# ---------------------------------------------------------------------------
_SCAFFOLD_TEMPLATE = '''"""Custom mi-race channel — edit `simulate` with your own physics.

It must return (times, X):
  times : 1D array of snapshot times, length n_steps
  X     : 2D int array (n_steps, L); X[t, k] = molecules in compartment k at time t

Wire it up in your config:
  "channel": {
    "type": "custom",
    "impl": "%(path)s:simulate",
    "L": 8, "dt": 0.01, "T": 2.0
  }
"""
import numpy as np


def simulate(schedule, cfg, rng):
    # schedule : list of (release_time, amount) injected into compartment 0
    # cfg      : the channel config dict — read your own params, e.g. cfg["my_param"]
    # rng      : numpy random Generator
    L = int(cfg["L"])
    dt = float(cfg["dt"])
    T = float(cfg["T"])
    n_steps = int(round(T / dt)) + 1
    times = np.linspace(0.0, T, n_steps)

    x = np.zeros(L, dtype=float)
    X = np.zeros((n_steps, L), dtype=int)
    releases = sorted(schedule)
    ptr = 0
    for i, t in enumerate(times):
        while ptr < len(releases) and releases[ptr][0] <= t:   # inject due releases
            x[0] += releases[ptr][1]
            ptr += 1
        # >>> your physics here: move molecules between compartments <<<
        X[i] = np.round(x).astype(int)
    return times, X
'''


def scaffold_channel(path: str) -> Path:
    """Write a ready-to-edit custom-channel template to ``path``."""
    p = Path(path)
    if p.exists():
        raise SystemExit(f"[mi-race] refusing to overwrite existing file: {p}")
    p.write_text(_SCAFFOLD_TEMPLATE % {"path": str(p)}, encoding="utf-8")
    return p


def run_channels(args) -> None:
    """CLI entry: list built-in channels, or scaffold a custom one with --new."""
    new_path = getattr(args, "new", None)
    if new_path:
        p = scaffold_channel(new_path)
        print(f"[mi-race] wrote channel template → {p}")
        print(f"[mi-race] edit simulate(), then set in your config:")
        print(f'          "channel": {{ "type": "custom", "impl": "{p}:simulate", ... }}')
        return

    print("\nBuilt-in channels  (set channel.type to one of these):\n")
    for name, desc, params in list_channels():
        print(f"  {name:<16}{desc}")
        print(f"  {'':<16}params: {', '.join(params)}\n")
    print("  custom          your own physics via channel.impl = 'file.py:function'")
    print("                  scaffold one with:  mi-race channels --new my_channel.py\n")
