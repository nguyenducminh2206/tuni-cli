"""Symbol release vectors: generation, rendering, and the `mi-race symbols` CLI.

A symbol is a length-``n_slots`` vector of non-negative molecule counts — how
many molecules to release into compartment 0 at each time slot. This is the
dense analog of the sparse ``[(t, amount), ...]`` schedule and the format the
learned encoder (Phase C) will output.

Total molecules per symbol is fixed by ``budget`` (the molecular analog of the
power constraint in the reference encoder): every symbol releases the same
number of molecules, so the decoder can't be helped by raw signal strength.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
def _round_preserving_sum(arr: np.ndarray, target: int) -> np.ndarray:
    """Round floats to ints whose sum is exactly ``target`` (largest-remainder)."""
    floor = np.floor(arr).astype(int)
    remainder = int(target) - int(floor.sum())
    if remainder > 0:
        order = np.argsort(arr - floor)[::-1][:remainder]
        floor[order] += 1
    elif remainder < 0:
        order = np.argsort(arr)[::-1][: (-remainder)]
        floor[order] -= 1
    return floor


def normalize_to_budget(vec, budget: int) -> list[int]:
    """Scale a non-negative vector so it sums to exactly ``budget`` (ints).

    Negative entries are clamped to 0. A zero vector is returned unchanged.
    """
    arr = np.clip(np.asarray(vec, dtype=float), 0.0, None)
    total = arr.sum()
    if total <= 0:
        return [0] * len(arr)
    scaled = arr * (float(budget) / total)
    return _round_preserving_sum(scaled, int(budget)).tolist()


def generate_initial_symbols(N: int, n_slots: int, budget: int, kind: str = "time") -> dict[int, list[int]]:
    """Return ``{symbol_id: length-n_slots int vector}`` for a starting codebook.

    kinds:
      - ``"time"``      one pulse per symbol, spread across the window (full budget)
      - ``"two-pulse"`` two equal pulses per symbol, shifting with the symbol id
      - ``"uniform"``   budget spread evenly across all slots
    """
    if N < 1 or n_slots < 1:
        raise ValueError("N and n_slots must be >= 1")
    symbols: dict[int, list[int]] = {}
    span = max(1, N - 1)
    for i in range(N):
        vec = [0] * n_slots
        if kind == "time":
            slot = int(round(i / span * (n_slots - 1)))
            vec[slot] = int(budget)
        elif kind == "two-pulse":
            s1 = int(round(i / span * (n_slots - 1) * 0.5))
            s2 = int(round((0.5 + i / span * 0.5) * (n_slots - 1)))
            half = int(budget) // 2
            vec[s1] += half
            vec[s2] += int(budget) - half
        elif kind == "uniform":
            vec = normalize_to_budget(np.ones(n_slots), int(budget))
        else:
            raise ValueError(f"unknown symbol kind: {kind!r}")
        symbols[i] = vec
    return symbols


_BLOCKS = " ▁▂▃▄▅▆▇█"


def render_symbol_bars(vec, budget: int | None = None) -> str:
    """One-line ASCII bar rendering of a release vector (block height ∝ amount)."""
    arr = np.asarray(vec, dtype=float)
    peak = float(budget) if budget else (float(arr.max()) if arr.max() > 0 else 1.0)
    cells = []
    for v in arr:
        if v <= 0:
            cells.append("·")
        else:
            level = int(round(v / peak * (len(_BLOCKS) - 1)))
            cells.append(_BLOCKS[max(1, min(level, len(_BLOCKS) - 1))])
    return "".join(cells)


# ---------------------------------------------------------------------------
# Manual entry parsing
# ---------------------------------------------------------------------------
def parse_pairs(text: str, n_slots: int) -> dict[int, int]:
    """Parse ``"0:30 4:70"`` → ``{0: 30, 4: 70}``. Raises ``ValueError`` on bad input."""
    out: dict[int, int] = {}
    for tok in text.replace(",", " ").split():
        if ":" not in tok:
            raise ValueError(f"expected slot:amount, got {tok!r}")
        s, a = tok.split(":", 1)
        slot, amount = int(s), int(a)
        if not (0 <= slot < n_slots):
            raise ValueError(f"slot {slot} out of range [0, {n_slots - 1}]")
        if amount < 0:
            raise ValueError(f"amount {amount} must be >= 0")
        out[slot] = out.get(slot, 0) + amount
    return out


def vec_from_pairs(pairs: dict[int, int], n_slots: int) -> list[int]:
    vec = [0] * n_slots
    for slot, amount in pairs.items():
        vec[slot] = amount
    return vec


# ---------------------------------------------------------------------------
# Config writing
# ---------------------------------------------------------------------------
def _compact_number_arrays(text: str) -> str:
    """Collapse JSON arrays that contain only numbers onto a single line."""
    def repl(m: re.Match) -> str:
        parts = [p for p in re.split(r"[\s,]+", m.group(1)) if p]
        return "[" + ", ".join(parts) + "]"

    return re.sub(r"\[\s*([\d\s,.\-]*?)\s*\]", repl, text)


def write_symbols_to_config(
    cfg: dict, cfg_path: Path, symbols: dict[int, list[int]],
    n_slots: int, slot_dt: float, budget: int,
) -> None:
    channel = cfg.setdefault("channel", {})
    channel["n_slots"] = int(n_slots)
    channel["slot_dt"] = float(slot_dt)
    channel["budget"] = int(budget)
    channel["symbols"] = {str(k): list(v) for k, v in sorted(symbols.items())}
    text = _compact_number_arrays(json.dumps(cfg, indent=2))
    cfg_path.write_text(text + "\n", encoding="utf-8")


def preview_symbols(symbols: dict[int, list[int]], slot_dt: float, budget: int) -> None:
    n_slots = len(next(iter(symbols.values()))) if symbols else 0
    print(f"\nPreview   (each row = one symbol · {n_slots} slots · █ height ∝ molecules)\n")
    for sid in sorted(symbols):
        vec = symbols[sid]
        nz = [(i, a) for i, a in enumerate(vec) if a > 0]
        desc = ", ".join(f"{a} @ slot {i} (t={i * slot_dt:.1f}s)" for i, a in nz) or "(empty)"
        print(f"  S{sid}  {render_symbol_bars(vec, budget=budget)}   {desc}")
    print()


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------
def _ask(prompt: str, default: str) -> str:
    raw = input(f"{prompt} [{default}]: ").strip()
    return raw or default


def run_symbols(args) -> None:
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise SystemExit(f"[mi-race] Config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    interactive = getattr(args, "type", None) is None

    print("\n╭──────────────────────── mi-race · build symbols ────────────────────────╮")
    print("│ A symbol is a release vector: how many molecules to drop in each slot.   │")
    print("╰──────────────────────────────────────────────────────────────────────────╯\n")

    if interactive:
        N = int(_ask("Number of symbols (N)", str(getattr(args, "n", None) or 4)))
        n_slots = int(_ask("Slots per symbol", str(getattr(args, "slots", None) or 20)))
        budget = int(_ask("Total molecules per symbol", str(getattr(args, "budget", None) or 100)))
        slot_dt = float(_ask("Seconds per slot", str(getattr(args, "slot_dt", None) or 0.1)))
        print("\nTemplate:\n  1) time       one pulse per symbol, spread across the window"
              "\n  2) two-pulse  two pulses per symbol"
              "\n  3) uniform    release evenly across all slots"
              "\n  4) manual     type each symbol's slots yourself")
        choice = _ask("Choose", "1")
        kind = {"1": "time", "2": "two-pulse", "3": "uniform", "4": "manual",
                "time": "time", "two-pulse": "two-pulse", "uniform": "uniform",
                "manual": "manual"}.get(choice, "time")
    else:
        N = int(getattr(args, "n", None) or 4)
        n_slots = int(getattr(args, "slots", None) or 20)
        budget = int(getattr(args, "budget", None) or 100)
        slot_dt = float(getattr(args, "slot_dt", None) or 0.1)
        kind = args.type

    if kind == "manual":
        symbols: dict[int, list[int]] = {}
        print(f"\nBudget = {budget} molecules per symbol.\n")
        for i in range(N):
            while True:
                raw = input(f"Symbol {i} — slot:amount pairs:  ").strip()
                try:
                    pairs = parse_pairs(raw, n_slots)
                except ValueError as e:
                    print(f"  ✗ {e}")
                    continue
                vec = vec_from_pairs(pairs, n_slots)
                total = sum(vec)
                if total != budget:
                    diff = budget - total
                    word = "short by" if diff > 0 else "over by"
                    print(f"  ✗ total = {total} / {budget}  ({word} {abs(diff)})")
                    continue
                desc = ", ".join(f"{a} @ slot {s}" for s, a in sorted(pairs.items()))
                print(f"  ✓ total = {total} / {budget}")
                print(f"  S{i}  {render_symbol_bars(vec, budget=budget)}   {desc}")
                symbols[i] = vec
                break
    else:
        symbols = generate_initial_symbols(N, n_slots, budget, kind=kind)
        preview_symbols(symbols, slot_dt, budget)

    if not getattr(args, "yes", False):
        confirm = input(f"Write these {N} symbols to {cfg_path}? [y/N]: ").strip().lower()
        if confirm not in {"y", "yes"}:
            print("[mi-race] symbols: aborted, nothing written.")
            return

    write_symbols_to_config(cfg, cfg_path, symbols, n_slots, slot_dt, budget)
    print(f"[mi-race] wrote channel.symbols (N={N}, slots={n_slots}, budget={budget}) → {cfg_path}")
