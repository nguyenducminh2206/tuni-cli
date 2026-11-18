"""ui: what mi-race prints. The [mi-race] prefix is added here and nowhere else;
the rest of the package raises plain messages"""


import textwrap
import re
import unicodedata
import os


PREFIX = "[mi-race]"
WIDTH = 79                                  # every line stays under 80 columns
INDENT = " " * (len(PREFIX) + 1)
HIGHLIGHT = "\033[1;31m"                    # bold red, for warnings and errors
RESET = "\033[0m"                           # back to normal text
ANSI = re.compile(r"\033\[[0-9;]*m")        # a colour code: it takes no columns on the screen
BOLD = "\033[1m"
CORRECT = "\033[1;32m"                      # bold green: a correct guess


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


def use_color(stream) -> bool:
    """colour only on a terminal, and never when the NO_COLOR environment variable is set (no-color.org);
    a file or a pipe gets plain text"""
    try:
        terminal = stream.isatty()
    except (AttributeError, ValueError, OSError):       # no isatty, or a closed stream
        terminal = False
    return terminal and not os.environ.get("NO_COLOR")


def _highlight(text: str, color: bool) -> str:
    """text in bold red, line by line, if color"""
    if not color:
        return text
    return "\n".join(f"{HIGHLIGHT}{line}{RESET}" for line in text.split("\n"))


def warning(text: str, color: bool = False) -> str:
    """'[mi-race] warning: text', in bold red if color"""
    return _highlight(_message(f"{PREFIX} warning: ", text), color)


def error(text: str, color: bool = False) -> str:
    """'[mi-race] error: text', in bold red if color"""
    return _highlight(_message(f"{PREFIX} error: ", text), color)


def display_width(text: str) -> int:
    """the columns a text takes in a terminal: a wide East Asian character takes 2, a colour code 0,
    every other character 1"""
    return sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in ANSI.sub("", text))


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


def confusion_table(result: dict, color: bool = False, width: int = WIDTH - 4) -> str:
    """one decoder's confusion matrix and summary, from evaluate(): 'predicted' across, 'sent' down,
    a zero shown as '·', then accuracy, macro F1 and the BA bound. If color (a terminal), the correct guesses
    (the diagonal) are bold green. Many symbols are split into blocks of columns that fit in width"""
    rows = [[int(count) for count in row] for row in result["confusion_matrix"]]
    n = len(rows)
    digits = len(str(n - 1))
    number = max(len(str(value)) for value in [n - 1, *(count for row in rows for count in row)])
    cell = number + 6                                                # a number in the middle, 3 spaces each side
    left = len("sent  ") + digits + 2                                # 'sent  1  ' in front of each row
    per_block = max(1, (width - left) // cell)

    def paint(text: str, style: str) -> str:
        return f"{style}{text}{RESET}" if color else text

    def count(u: int, v: int) -> str:
        text = f"{rows[u][v] or '·':>{number}}".center(cell)          # units under units
        return paint(text, CORRECT) if u == v else text

    lines = []
    for first in range(0, n, per_block):
        columns = range(first, min(first + per_block, n))
        lines += [" " * left + "predicted".center(cell * len(columns)),
                  " " * left + "".join(f"{v:>{number}}".center(cell) for v in columns), ""]
        lines += [f"{'sent' if u == (n - 1) // 2 else '':>4}  {u:>{digits}}  " + "".join(count(u, v) for v in columns)
                  for u in range(n)]
        lines.append("")
    summary = [(f"{result['accuracy']:.1%}", "accuracy"), (f"{result['macro_f1']:.3f}", "macro F1"),
               (f"{result['ba_bound_bits']:.3f} bits", "BA bound")]
    lines.append("    ".join(f"{paint(value, BOLD)} {name}" for value, name in summary))
    full = max(display_width(line) for line in lines)
    return "\n".join(line + " " * (full - display_width(line)) for line in lines)      # every line the same width