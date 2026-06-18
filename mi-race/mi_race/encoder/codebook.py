"""Symbol -> release schedule mapping.

Symbols may be written in either format and are auto-detected:

  - **dense vector** (current): a flat list of per-slot molecule counts, e.g.
    ``[50, 0, 0, 20, 0, ...]``. Slot ``i`` releases at time ``i * slot_dt``.
  - **sparse schedule** (legacy): a list of ``[t, amount]`` pairs, e.g.
    ``[[0.0, 100], [0.5, 100]]``.

Both are converted to the canonical ``[(t, amount), ...]`` schedule that the
SSA simulator consumes.
"""
from __future__ import annotations


def vector_to_schedule(vec, slot_dt: float) -> list[tuple[float, int]]:
    """Convert a dense per-slot release vector to a sparse ``[(t, amount)]`` schedule."""
    schedule: list[tuple[float, int]] = []
    for i, amount in enumerate(vec):
        a = int(round(float(amount)))
        if a > 0:
            schedule.append((i * float(slot_dt), a))
    return schedule


def _is_dense_vector(events) -> bool:
    """True if ``events`` is a flat list of numbers (dense vector), not ``[[t, a], ...]``."""
    if not events:
        return True  # empty → treat as a zero (dense) vector
    return not isinstance(events[0], (list, tuple))


def codebook_from_config(channel_cfg: dict) -> dict[int, list[tuple[float, int]]]:
    """Parse ``channel.symbols`` into ``{symbol_id: [(t, amount), ...]}``.

    Dense vectors use ``channel.slot_dt`` (default 0.1s) to place each slot in time.
    """
    raw = channel_cfg.get("symbols")
    if not raw:
        raise SystemExit("[mi-race] channel.symbols missing in config.")
    slot_dt = float(channel_cfg.get("slot_dt", 0.1))

    out: dict[int, list[tuple[float, int]]] = {}
    for k, events in raw.items():
        sid = int(k)
        if _is_dense_vector(events):
            out[sid] = vector_to_schedule(events, slot_dt)
        else:
            out[sid] = [(float(t), int(a)) for (t, a) in events]
    return out
