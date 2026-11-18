"""ui: what mi-race prints. The [mi-race] prefix is added here and nowhere else;
the rest of the package raises plain messages"""


import textwrap
import re
import unicodedata


PREFIX = "[mi-race]"
WIDTH = 79                  # every line stays under 80 columns
INDENT = " " * (len(PREFIX) + 1)


def _message(head: str, text: str) -> str:
    """head + text, wrapped at WIDTH: later lines are indented by the prefix width
    a line break in text starts a new line; a word is never cut, so a long path stays whole"""
    lines = text.splitlines()
    return "\n".join(textwrap.fill(line, width=WIDTH, initial_indent=head if i == 0 else INDENT,
                                   subsequent_indent=INDENT, break_long_words=False, break_on_hyphens=False)
                     for i, line in enumerate(lines))


def info(text: str) -> str:
    """'[mi-race] text'"""
    return _message(f"{PREFIX} ", text)


def warning(text: str) -> str:
    """'[mi-race] warning: text'"""
    return _message(f"{PREFIX} warning: ", text)


def error(text: str) -> str:
    """'[mi-race] error: text'"""
    return _message(f"{PREFIX} error: ", text)


def display_width(text: str) -> int:
    """the columns a text takes in a terminal: a wide East Asian character takes 2, every other character 1"""
    return sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in text)


def _fit(line: str, width: int) -> list[str]:
    """a box line cut into pieces of at most width columns, at spaces only;
    in a 'key  value' line the later pieces line up under the value"""
    if display_width(line) <= width:
        return [line]
    key = re.match(r"(.*?\S)\s{2,}", line)               # 'Config   /path': the value starts after 2+ spaces
    indent = " " * key.end() if key and key.end() < width // 2 else ""
    return textwrap.wrap(line, width=width, subsequent_indent=indent, break_long_words=False, break_on_hyphens=False)


def box(lines: list[str], title: str = "") -> str:
    """lines in a box with rounded corners and an optional title in the top border;
    every line of the box has the same width: at most WIDTH, unless a single word is longer.
    an item with line breaks (a table) becomes several lines"""
    label = f" {title} " if title else ""
    body = [piece for item in lines for line in item.split("\n") for piece in _fit(line, WIDTH - 4)]   # 4: '│ ' and ' │'
    inner = max([display_width(line) + 2 for line in body] + [display_width(label)])
    left = (inner - display_width(label)) // 2                                                      # the title in the middle
    top = "╭" + "─" * left + label + "─" * (inner - left - display_width(label)) + "╮"
    middle = [f"│ {line}{' ' * (inner - 2 - display_width(line))} │" for line in body]
    return "\n".join([top, *middle, "╰" + "─" * inner + "╯"])