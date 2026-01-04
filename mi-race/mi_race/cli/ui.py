from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass(frozen=True)
class Box:
    title: Optional[str]
    lines: tuple[str, ...]


def _visible_len(s: str) -> int:
    # For now we assume no ANSI codes; keep it simple.
    return len(s)


def render_box(lines: Iterable[str], *, title: Optional[str] = None, min_width: int = 60) -> str:
    """Render a clean Unicode box for terminal output."""
    lines_t = tuple(str(x) for x in lines)
    inner_width = max((_visible_len(x) for x in lines_t), default=0)
    inner_width = max(inner_width, min_width)

    if title:
        t = f" {title} "
        remaining = max(inner_width - len(t), 0)
        left = remaining // 2
        right = remaining - left
        top = f"╭{'─' * left}{t}{'─' * right}╮"
    else:
        top = f"╭{'─' * inner_width}╮"

    body = [f"│ {line.ljust(inner_width - 2)} │" for line in lines_t]
    bottom = f"╰{'─' * inner_width}╯"
    return "\n".join([top, *body, bottom])
