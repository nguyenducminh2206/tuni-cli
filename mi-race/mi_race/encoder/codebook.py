"""Symbol -> release schedule mapping. Phase 1 reads from config."""
from __future__ import annotations


def codebook_from_config(channel_cfg: dict) -> dict[int, list[tuple[float, int]]]:
    """
    Parse `channel.symbols` into a {symbol_id: [(t, a), ...]} dict.

    Expected config shape:
      "channel": {
        "symbols": {
          "0": [[0.0, 100]],
          "1": [[0.5, 100]],
          "2": [[1.0, 100]]
        }
      }
    """
    raw = channel_cfg.get("symbols")
    if not raw:
        raise SystemExit("[mi-race] channel.symbols missing in config.")
    out: dict[int, list[tuple[float, int]]] = {}
    for k, events in raw.items():
        sid = int(k)
        out[sid] = [(float(t), int(a)) for (t, a) in events]
    return out
