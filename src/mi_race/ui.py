"""ui: what mi-race prints, in one style for every element:
- everything is indented 2 spaces, and every line stays under 80 columns;
- bold for the main thing, faint for what is secondary, colour only with a meaning
  (bold green: a correct guess; red: a warning or an error);
- ' · ' between facts, a faint rule under every title;
- a sign for every kind of line: ✓ done, ! warning, ✗ error, ? a question;
- a bold cyan [mi-race] starts the tool's own messages, questions and headings; the banner is grey.
The prefix is added here and nowhere else; the rest of the package raises plain messages.
Styles are only used on a terminal (color=True): a file or a pipe gets plain text."""

import os
import re
import sys
import textwrap
import time
import unicodedata

from mi_race import __version__

PREFIX = "[mi-race]"
WIDTH = 79                                  # every line stays under 80 columns
INDENT = "  "                               # every element starts 2 columns in
TEXT_INDENT = " " * len(INDENT + PREFIX + " ")        # where a message's text starts
RULE = WIDTH - len(INDENT)                  # the width of the rule under a title
DESCRIPTION = ("molecular communication through a diffusion channel: exact simulation, learned decoders, "
               "and a search for the most informative symbols")

BOLD = "\033[1m"
DIM = "\033[2m"                             # faint, for what is secondary
RED = "\033[31m"                            # a warning or an error
CORRECT = "\033[1;32m"                      # bold green: a correct guess
GRAY = "\033[90m"                           # grey: the banner, the version and the config under it
SOFT = "\033[37m"                           # light grey, normal weight: the description under the banner
ART = GRAY                                  # the large MI-RACE banner
BRAND = "\033[1;36m"                        # cyan, not bold: the [mi-race] prefix
RESET = "\033[0m"                           # back to normal text
ANSI = re.compile(r"\033\[[0-9;]*m")        # a style code: it takes no columns on the screen

UP = "\033[{}A"                             # move the cursor up n lines
CLEAR_DOWN = "\033[J"                       # clear from the cursor to the end of the screen
HIDE_CURSOR, SHOW_CURSOR = "\033[?25l", "\033[?25h"

BAR_WIDTH = 32
LABEL_WIDTH = 13                            # progress labels padded to this, so stacked bars line up
ESTIMATE_AFTER = 5                          # steps before the time left is estimated


# --- terminal and style --------------------------------------------------------------------------

def is_terminal(stream) -> bool:
    """True for a terminal; False for a file, a pipe, or anything without isatty"""
    try:
        return stream.isatty()
    except (AttributeError, ValueError, OSError):       # no isatty, or a closed stream
        return False


def use_color(stream) -> bool:
    """styles only on a terminal, and never when the NO_COLOR environment variable is set (no-color.org)"""
    return is_terminal(stream) and not os.environ.get("NO_COLOR")


def paint(text: str, style: str, color: bool) -> str:
    """text in a style, if color; plain text otherwise"""
    return f"{style}{text}{RESET}" if color and style and text else text


def display_width(text: str) -> int:
    """the columns a text takes in a terminal: a wide East Asian character takes 2, a style code 0,
    every other character 1"""
    return sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in ANSI.sub("", text))


def pad(lines: list[str]) -> list[str]:
    """every line padded with spaces to the width of the widest, so a block lines up"""
    width = max((display_width(line) for line in lines), default=0)
    return [line + " " * (width - display_width(line)) for line in lines]


# --- 7a: messages --------------------------------------------------------------------------------

def _wrap(head: str, text: str) -> list[str]:
    """head + text, wrapped at WIDTH: later lines start at TEXT_INDENT, under the sign.
    A line break in text starts a new line; a word is never cut, so a long path stays whole"""
    lines = []
    for i, paragraph in enumerate(text.splitlines() or [""]):
        lines += textwrap.wrap(paragraph, width=WIDTH, initial_indent=head if i == 0 else TEXT_INDENT,
                               subsequent_indent=TEXT_INDENT, break_long_words=False,
                               break_on_hyphens=False) or [head if i == 0 else ""]
    return lines


def _message(label: str, text: str, style: str, color: bool) -> str:
    """'  [mi-race] label · text': the label (sign and word) in bold, the text in style"""
    head = INDENT + PREFIX + " " + (label + " · " if label else "")
    lines = _wrap(head, text)
    first = INDENT + paint(PREFIX, BRAND, color) + " "
    first += (paint(label, BOLD + style, color) + paint(" · " + lines[0][len(head):], style, color)) if label \
        else lines[0][len(head):]
    return "\n".join([first] + [TEXT_INDENT + paint(line[len(TEXT_INDENT):], style, color) for line in lines[1:]])


def info(text: str, color: bool = False) -> str:
    """'  [mi-race] text'"""
    return _message("", text, "", color)


def warning(text: str, color: bool = False) -> str:
    """'  [mi-race] ! warning · text', in red"""
    return _message("! warning", text, RED, color)


def error(text: str, color: bool = False) -> str:
    """'  [mi-race] ✗ error · text', in red"""
    return _message("✗ error", text, RED, color)


def question(text: str, yes: bool = True, color: bool = False) -> str:
    """the prompt of a yes/no question: '  [mi-race] ? text [Y/n] ', the default in capitals.
    Only the prompt: asking and reading the answer come with the questions part (7f)"""
    choice = "[Y/n]" if yes else "[y/N]"
    lines = _wrap(INDENT + PREFIX + " ? ", f"{text} {choice}")
    lines[0] = INDENT + paint(PREFIX, BRAND, color) + " " + paint("?", BOLD, color) + lines[0][len(INDENT + PREFIX + " ?"):]
    lines[-1] = lines[-1][:-len(choice)] + paint(choice, DIM, color)
    return "\n".join(lines) + " "


def value_prompt(label: str, text: str, default: str, width: int = 0, color: bool = False) -> str:
    """the prompt of a value question, set out like a fact: '  label  text [default] › '.
    The label is faint and padded to width, so the questions of a section line up; Enter keeps the default"""
    return f"{INDENT}{paint(label.ljust(width), DIM, color)}  {text} {paint(f'[{default}]', DIM, color)} › "


def done(what: str, *facts: str, color: bool = False) -> str:
    """'  ✓ what · fact · fact': something finished, what in bold"""
    return INDENT + "✓ " + " · ".join([paint(what, BOLD, color), *[fact for fact in facts if fact]])


# --- 7b: headings, sections and facts ------------------------------------------------------------

def heading(title: str, subtitle: str = "", color: bool = False) -> str:
    """the start of a command or stage: '[mi-race] / title' with the title in bold, a faint rule,
    and a faint subtitle"""
    lines = [INDENT + paint(PREFIX, BRAND, color) + " / " + paint(title, BOLD, color), INDENT + paint("─" * RULE, DIM, color)]
    return "\n".join(lines + ([INDENT + paint(subtitle, DIM, color)] if subtitle else []))


def section(title: str, note: str = "", body: str = "", color: bool = False) -> str:
    """a block of results: the title in bold, a faint note, a faint rule, then the body
    (already indented, as every element is)"""
    head = INDENT + paint(title, BOLD, color) + (paint(f" · {note}", DIM, color) if note else "")
    return "\n".join([head, INDENT + paint("─" * RULE, DIM, color)] + (["", body] if body else []))


def facts(pairs: list[tuple[str, str]], color: bool = False) -> str:
    """settings or results as 'label  value' lines: faint labels, padded so the values line up"""
    width = max(display_width(label) for label, _ in pairs)
    return "\n".join(INDENT + paint(label.ljust(width), DIM, color) + "  " + value for label, value in pairs)


# --- 7c: the confusion matrix --------------------------------------------------------------------

def confusion_table(result: dict, color: bool = False, width: int = WIDTH) -> str:
    """one decoder's confusion matrix and summary, from evaluate(): 'predicted' across, 'sent' down,
    labels and zeros ('·') faint, the correct guesses (the diagonal) bold green, then accuracy, macro F1
    and the BA bound. Many symbols are split into blocks of columns that fit in width"""
    rows = [[int(count) for count in row] for row in result["confusion_matrix"]]
    n = len(rows)
    digits = len(str(n - 1))
    number = max(len(str(value)) for value in [n - 1, *(count for row in rows for count in row)])
    cell = number + 6                                                # a number in the middle, 3 spaces each side
    left = len(INDENT + "sent  ") + digits + 2                       # '  sent  1  ' in front of each row
    per_block = max(1, (width - left) // cell)

    def count(u: int, v: int) -> str:
        text = f"{rows[u][v] or '·':>{number}}".center(cell)          # units under units
        return paint(text, CORRECT if u == v else DIM if rows[u][v] == 0 else "", color)

    lines = []
    for first in range(0, n, per_block):
        columns = range(first, min(first + per_block, n))
        lines += [" " * left + paint("predicted".center(cell * len(columns)), DIM, color),
                  " " * left + paint("".join(f"{v:>{number}}".center(cell) for v in columns), DIM, color), ""]
        lines += [INDENT + paint(f"{'sent' if u == (n - 1) // 2 else '':>4}  {u:>{digits}}", DIM, color) + "  "
                  + "".join(count(u, v) for v in columns) for u in range(n)]
        lines.append("")
    summary = [(f"{result['accuracy']:.1%}", "accuracy"), (f"{result['macro_f1']:.3f}", "macro F1"),
               (f"{result['ba_bound_bits']:.3f} bits", "BA bound")]
    lines.append(INDENT + paint(" · ", DIM, color).join(f"{paint(value, BOLD, color)} {name}" for value, name in summary))
    return "\n".join(pad(lines))                                     # every line the same width


# --- 7d: banner ----------------------------------------------------------------------------------

BANNER = """\
███╗   ███╗██╗      ██████╗  █████╗  ██████╗███████╗
████╗ ████║██║      ██╔══██╗██╔══██╗██╔════╝██╔════╝
██╔████╔██║██║█████╗██████╔╝███████║██║     █████╗
██║╚██╔╝██║██║╚════╝██╔══██╗██╔══██║██║     ██╔══╝
██║ ╚═╝ ██║██║      ██║  ██║██║  ██║╚██████╗███████╗
╚═╝     ╚═╝╚═╝      ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚══════╝"""


def banner(config: str | None = None, version: str = __version__, color: bool = False) -> str:
    """the MI-RACE art in grey, then '[mi-race] version · config' (the prefix cyan, the rest grey) and what
    the tool does; the CLI prints it only on a terminal"""
    art = [INDENT + paint(line, ART, color) for line in BANNER.split("\n")]
    line = (
        INDENT
        + paint(PREFIX, BRAND, color)
        + " "
        + paint(
            " · ".join([version] + ([config] if config else [])),
            GRAY,
            color,
        )
    )
    about = [INDENT + paint(line, SOFT, color) for line in textwrap.wrap(DESCRIPTION, width=WIDTH - len(INDENT))]
    return "\n".join(art + ["", line] + about)


# --- 7e: progress --------------------------------------------------------------------------------

def clock(seconds: float) -> str:
    """seconds as mm:ss, or h:mm:ss from an hour on"""
    minutes, seconds = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


def progress_lines(label: str, done_: int, total: int, stats: str = "", detail: str = "", elapsed: float = 0.0) -> list[str]:
    """the lines of a running progress bar, styled for a terminal:
      label          ━━━━━━━━━━━━━━━━━━━─────────────  124/200   62%
                     00:04 elapsed · ~00:02 left · stats
                     detail                                 (only if there is one)"""
    filled = BAR_WIDTH * done_ // total if total else BAR_WIDTH
    percent = 100 * done_ // total if total else 100
    bar = paint("━" * filled, BOLD, True) + paint("─" * (BAR_WIDTH - filled), DIM, True)
    left = f"~{clock(elapsed / done_ * (total - done_))} left" if done_ >= ESTIMATE_AFTER else "estimating time"
    under = " " * len(INDENT + " " * LABEL_WIDTH + "  ")              # the lines below start under the bar
    lines = [f"{INDENT}{label:<{LABEL_WIDTH}}  {bar}  {done_:>{len(str(total))}}/{total}  {percent:3d}%",
             under + paint(" · ".join([f"{clock(elapsed)} elapsed", left] + ([stats] if stats else [])), DIM, True)]
    return lines + ([under + paint(detail, DIM, True)] if detail else [])


class Progress:
    """a progress bar redrawn in place, for the progress hooks of the core: call it with
    (done, total, stats, detail) after each step; when done == total it becomes one ✓ line.
    It draws only on a terminal (stderr by default). Use it in a with block, so the cursor
    comes back even if the work stops early"""

    def __init__(self, label: str, finished: str = "", unit: str = "steps", stream=None, now=time.monotonic):
        self.label, self.finished, self.unit = label, finished or f"{label} done", unit
        self.stream = stream or sys.stderr
        self.now = now
        self.start = now()
        self.show = is_terminal(self.stream)
        self.height = 0                                             # lines drawn last time
        self.over = False

    def __call__(self, done_: int, total: int, stats: str = "", detail: str = "") -> None:
        if not self.show or self.over:
            return
        elapsed = self.now() - self.start
        out = UP.format(self.height) + "\r" + CLEAR_DOWN if self.height else HIDE_CURSOR    # back to the top of the bar
        if done_ >= total:
            out += done(self.finished, f"{total} {self.unit}", stats, clock(elapsed), color=True) + "\n" + SHOW_CURSOR
            self.height, self.over = 0, True
        else:
            lines = progress_lines(self.label, done_, total, stats, detail, elapsed)
            out += "".join(line + "\n" for line in lines)
            self.height = len(lines)
        self.stream.write(out)
        self.stream.flush()

    def __enter__(self):
        return self

    def __exit__(self, *error) -> None:
        if self.show and not self.over:                             # stopped early: give the cursor back
            self.stream.write(SHOW_CURSOR)
            self.stream.flush()


# --- 7f: questions -------------------------------------------------------------------------------

ANSWERS = {"y": True, "yes": True, "n": False, "no": False}


def answer(text: str, yes: bool = True, accept_defaults: bool = False, read=input, stdin=None,
           out=None) -> tuple[bool, str]:
    """a yes/no question, answered on the terminal: Enter takes the default (the capital letter),
    y/yes/n/no are accepted in any case, anything else asks again. Ctrl-D also takes the default.
    With accept_defaults (--yes), or without a terminal to answer on (a batch job), the default is taken
    and written after the question, so the output shows what was decided and why.
    Returns the answer and how it was given: 'typed', 'Enter', 'Ctrl-D', '--yes' or 'no terminal'"""
    stdin, out = stdin or sys.stdin, out or sys.stdout
    color = use_color(out)
    prompt = question(text, yes, color)
    if accept_defaults or not is_terminal(stdin):
        how = "--yes" if accept_defaults else "no terminal"
        note = f"{'yes' if yes else 'no'} · the default ({how})"
        room = WIDTH - display_width(prompt.split("\n")[-1])        # does the note fit after the question?
        out.write((prompt if len(note) <= room else prompt.rstrip() + "\n" + TEXT_INDENT) + paint(note, DIM, color) + "\n")
        out.flush()
        return yes, how
    while True:
        try:
            reply = read(prompt).strip().lower()
        except EOFError:                                            # Ctrl-D: the default
            out.write("\n")
            return yes, "Ctrl-D"
        if not reply:
            return yes, "Enter"
        if reply in ANSWERS:
            return ANSWERS[reply], "typed"
        out.write(TEXT_INDENT + paint(f"answer y or n (Enter: {'yes' if yes else 'no'})", DIM, color) + "\n")


def ask(text: str, yes: bool = True, accept_defaults: bool = False, read=input, stdin=None, out=None) -> bool:
    """a yes/no question (see answer), when only the answer matters"""
    return answer(text, yes, accept_defaults, read, stdin, out)[0]
