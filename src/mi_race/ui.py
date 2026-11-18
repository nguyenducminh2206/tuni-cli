"""ui: what mi-race prints. The [mi-race] prefix is added here and nowhere else;
the rest of the package raises plain messages"""
import textwrap

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