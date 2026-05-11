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


def render_confusion_matrix(
    cm,
    labels: Iterable,
    *,
    indent: int = 0,
    cell_width: int = 7,
) -> str:
    """Render a confusion matrix as plain text, aligned by columns.

    Used both for per-split inline display and inside the final result box.
    ``indent`` is the number of leading spaces on every line. ``cell_width``
    controls column width (large counts may need 8+).
    """
    pad = " " * indent
    label_strs = [str(l) for l in labels]
    label_w = max((len(s) for s in label_strs), default=1)

    header = pad + " " * (label_w + 2) + "".join(
        f"{s:>{cell_width}}" for s in label_strs
    )
    rows = [header]
    for i, row_label in enumerate(label_strs):
        cells = "".join(f"{int(cm[i][j]):>{cell_width}}" for j in range(len(label_strs)))
        rows.append(f"{pad}{row_label:>{label_w}}  {cells}")
    return "\n".join(rows)


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
    # +2 accounts for the single-space padding on each side of every body line
    # ("│ <content> │"). Without it, a line exactly min_width-1 long would
    # overflow the right border by 1 column.
    inner_width = max(measured + 2, min_width, _visible_len(title_text) if title_text else 0)
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
