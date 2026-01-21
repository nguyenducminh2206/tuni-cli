from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import textwrap


@dataclass(frozen=True)
class Box:
    title: Optional[str]
    lines: tuple[str, ...]


def _visible_len(s: str) -> int:
    # For now we assume no ANSI codes; keep it simple.
    return len(s)


def _wrap_for_box(line: str, *, width: int) -> list[str]:
    if width <= 0:
        return [""]
    if _visible_len(line) <= width:
        return [line]

    # Try a simple hanging-indent format for key/value lines like: "Columns  a, b, c"
    sep_idx = line.find("  ")
    if sep_idx != -1:
        prefix = line[: sep_idx + 2]
        rest = line[sep_idx + 2 :].strip()
        if prefix.strip() and rest:
            avail = max(width - _visible_len(prefix), 1)
            chunks = textwrap.wrap(
                rest,
                width=avail,
                break_long_words=True,
                break_on_hyphens=False,
            )
            out = [prefix + chunks[0]]
            indent = " " * _visible_len(prefix)
            out.extend(indent + c for c in chunks[1:])
            return out

    # Generic wrap
    return textwrap.wrap(
        line,
        width=width,
        break_long_words=True,
        break_on_hyphens=False,
    )


def render_box(
    lines: Iterable[str],
    *,
    title: Optional[str] = None,
    min_width: int = 60,
    max_width: int | None = None,
) -> str:
    """Render a clean Unicode box for terminal output.

    Notes:
      - If `max_width` is provided, long lines are wrapped to fit and the box width is capped.
      - Title length always wins (box will widen as needed to avoid broken borders).
    """
    raw_lines = [str(x) for x in lines]
    title_text = f" {title} " if title else None

    measured = max((_visible_len(x) for x in raw_lines), default=0)
    inner_width = max(measured, min_width, _visible_len(title_text) if title_text else 0)
    if max_width is not None:
        inner_width = min(inner_width, max_width)
        inner_width = max(inner_width, _visible_len(title_text) if title_text else 0)

    content_width = max(inner_width - 2, 1)

    if max_width is not None:
        wrapped: list[str] = []
        for line in raw_lines:
            wrapped.extend(_wrap_for_box(line, width=content_width))
        lines_t = tuple(wrapped)
    else:
        lines_t = tuple(raw_lines)

    if title_text:
        t = title_text
        remaining = max(inner_width - len(t), 0)
        left = remaining // 2
        right = remaining - left
        top = f"╭{'─' * left}{t}{'─' * right}╮"
    else:
        top = f"╭{'─' * inner_width}╮"

    body = [f"│ {line.ljust(content_width)} │" for line in lines_t]
    bottom = f"╰{'─' * inner_width}╯"
    return "\n".join([top, *body, bottom])
