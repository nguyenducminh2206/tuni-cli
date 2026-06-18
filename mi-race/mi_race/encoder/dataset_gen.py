"""Generate a labeled trajectory CSV by running the channel on a codebook.

Also saves per-symbol and combined trajectory plots next to the CSV so users
can eyeball what their generated dataset actually looks like.
"""
from __future__ import annotations
from pathlib import Path
import json
from typing import Sequence

import numpy as np
import pandas as pd

from mi_race.channel.simulation import simulate_ssa_with_schedule
from mi_race.encoder.codebook import codebook_from_config


# ``schedule`` is a list of (time, amount) tuples; alias for readability.
Schedule = Sequence[tuple[float, int]]


def _save_channel_signal_plot(
    times: np.ndarray,
    X: np.ndarray,
    merged_schedule: Schedule,
    out_csv: Path,
    L: int,
) -> None:
    """Save one figure showing every compartment over time for a SINGLE SSA run
    in which all symbols' pulses fire into the same tube and superpose.

    File: ``<stem>_signal.png`` next to ``out_csv``.
    """
    # Lazy import so callers that don't need plotting stay light.
    import matplotlib.pyplot as plt

    stem = out_csv.stem
    out_dir = out_csv.parent

    fig, ax = plt.subplots(figsize=(10, 6))
    for k in range(min(L, X.shape[1])):
        ax.plot(times, X[:, k], label=f"comp {k}")  # 0-indexed to match CSV cols

    sorted_sched = sorted(merged_schedule, key=lambda p: p[0])
    sched_str = " + ".join(f"({t:.2f}s, {a})" for t, a in sorted_sched)
    ax.set_title(f"Channel signal — all pulses in one tube: {sched_str}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Molecule count")
    ax.grid(True, alpha=0.3)
    if L <= 12:
        ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    out_path = out_dir / f"{stem}_signal.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[mi-race] Saved plot: {out_path.name}")


def generate_dataset(cfg: dict, out_csv: Path) -> Path:
    """
    Read `channel` block from cfg, run SSA `runs_per_symbol` times per symbol,
    write a wide CSV with one row per run.

    The full tube is always recorded — columns are
    ``symbol, comp0_0..comp0_{n_steps}, comp1_0..., ..., comp{L-1}_*``. The
    decoder decides which compartments to observe via ``data.x_cols``; generation
    is observation-agnostic, so one dataset serves every observation experiment.

    Plots one diagnostic trajectory alongside the CSV unless
    ``channel.make_plots`` is set to ``false``.
    """
    channel = cfg["channel"]
    L = int(channel["L"])
    S = float(channel["S"])
    D = float(channel["D"])
    dt = float(channel["dt"])
    T = float(channel["T"])
    runs = int(channel["runs_per_symbol"])
    seed = int(channel.get("seed", 12345))
    make_plots = bool(channel.get("make_plots", True))

    codebook = codebook_from_config(channel)
    rng_master = np.random.default_rng(seed)

    print(
        f"[mi-race] generate-data  L={L} T={T} dt={dt} "
        f"runs_per_symbol={runs} symbols={sorted(codebook.keys())}"
    )

    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None
    from mi_race.cli.ui import progress_disabled

    total_runs = len(codebook) * runs
    pbar = None
    if tqdm is not None:
        pbar = tqdm(
            total=total_runs,
            desc="[mi-race] SSA runs",
            unit="run",
            disable=progress_disabled(),
            leave=False,
        )

    rows = []
    n_steps = None
    for sid, schedule in sorted(codebook.items()):
        for _ in range(runs):
            run_seed = int(rng_master.integers(0, 2**32 - 1))
            rng = np.random.default_rng(run_seed)
            times, X = simulate_ssa_with_schedule(schedule, L, S, D, dt, T, rng)
            if n_steps is None:
                n_steps = X.shape[0]
            row: dict = {"symbol": sid}
            for k in range(L):
                for ti in range(n_steps):
                    row[f"comp{k}_{ti}"] = int(X[ti, k])
            rows.append(row)
            if pbar is not None:
                pbar.update(1)

    if pbar is not None:
        pbar.close()

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[mi-race] Wrote {len(df):,} rows × {df.shape[1]} cols to {out_csv}")

    if make_plots:
        # Diagnostic view: ONE tube, ONE SSA run, ALL pulses superposed.
        # Uses a separate RNG so the main training data is unaffected.
        merged_schedule: list[tuple[float, int]] = []
        for sched in codebook.values():
            merged_schedule.extend(sched)
        plot_rng = np.random.default_rng(seed + 1)
        times, X = simulate_ssa_with_schedule(merged_schedule, L, S, D, dt, T, plot_rng)
        _save_channel_signal_plot(times, X, merged_schedule, out_csv, L)

    return out_csv


def run_generate_data(args) -> None:
    """CLI entry point for `mi-race generate-data`."""
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise SystemExit(f"[mi-race] Config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "channel" not in cfg:
        raise SystemExit("[mi-race] Config is missing 'channel' section.")

    out_path = Path(args.out) if getattr(args, "out", None) else Path(
        cfg.get("data", {}).get("path", "data/encoder_dataset.csv")
    )

    generate_dataset(cfg, out_path)
