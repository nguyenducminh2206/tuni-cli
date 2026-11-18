from pathlib import Path
import io

import numpy as np

from mi_race import __version__
from mi_race.ui import (ANSI, BANNER, CORRECT, PREFIX, RESET, WIDTH, banner, box, confusion_table, display_width,
                         error, info, is_terminal, warning, use_color)


def test_each_kind_of_message_has_its_prefix():
    assert info("dataset written") == "[mi-race] dataset written"
    assert warning("only 50 runs per symbol") == "[mi-race] warning: only 50 runs per symbol"
    assert error("config file not found") == "[mi-race] error: config file not found"


def test_a_line_is_wrapped_before_80_columns():
    text = "x" * 60 + " " + "y" * 9                  # with the prefix: exactly 80 characters on one line
    assert info(text) == "[mi-race] " + "x" * 60 + "\n" + " " * 10 + "y" * 9


def test_a_long_message_keeps_every_word_and_the_prefix_once():
    text = " ".join(f"word{i}" for i in range(60))
    lines = warning(text).split("\n")
    assert len(lines) > 1 and all(len(line) < 80 for line in lines)
    assert all(line.startswith(" " * 10) for line in lines[1:])          # later lines: indented by the prefix width
    assert warning(text).count(PREFIX) == 1
    assert " ".join(warning(text).replace("[mi-race] warning:", "").split()) == text


def test_a_long_path_is_never_cut():
    path = "/Users/someone/projects/" + "a-very-long-folder-name/" * 4 + "dataset.csv"
    lines = error(f"cannot read {path}").split("\n")
    assert lines == ["[mi-race] error: cannot read", " " * 10 + path]


def test_a_line_break_starts_a_new_indented_line():
    assert error("2 problems:\nfirst\nsecond") == "[mi-race] error: 2 problems:\n" + " " * 10 + "first\n" + " " * 10 + "second"


def test_no_other_module_adds_the_prefix():
    """SPEC: the prefix is added in one place, the rest of the package raises plain messages"""
    package = Path(__file__).parents[1] / "src" / "mi_race"
    others = [path for path in package.rglob("*.py") if path.name != "ui.py"]
    assert others and not [path.name for path in others if PREFIX in path.read_text()]


# 7b: boxes
def widths(text):
    """the set of display widths of the lines of a text: one value if they all line up"""
    return {display_width(line) for line in text.split("\n")}


def test_display_width_counts_terminal_columns():
    assert display_width("abc") == 3
    assert display_width("→ − ±") == 5                 # arrows and math signs: 1 column each
    assert display_width("日本") == 4                   # wide characters: 2 columns each


def test_a_box_by_hand():
    assert box(["a", "bcd"], "t") == "╭─ t ─╮\n│ a   │\n│ bcd │\n╰─────╯"


def test_every_line_of_a_box_has_the_same_width():
    boxes = [box(["forest  1.030 → 1.007   −0.024 ± 0.014", "MLP     1.223 → 1.224   +0.001 ± 0.019"], "start → found"),
             box(["short"], "a title much longer than the text inside"),
             box(["", "after an empty line"]),
             box(["日本語 symbols", "ascii"], "幅"),
             box([])]
    for text in boxes:
        assert len(widths(text)) == 1, text


def test_the_title_sits_in_the_middle():
    top = box(["x" * 20], "results").split("\n")[0]
    left, right = top[1:-1].split(" results ")
    assert set(left + right) == {"─"} and abs(len(left) - len(right)) <= 1


def test_a_long_title_widens_the_box():
    assert box(["x"], "a long title").split("\n")[:2] == ["╭ a long title ╮", "│ x" + " " * 11 + " │"]


def test_a_box_is_at_most_79_columns():
    fits = box(["x" * 70 + " " + "y" * 4])                     # 75 characters: the widest line that fits
    assert fits.count("\n") == 2 and widths(fits) == {WIDTH}
    wraps = box(["x" * 70 + " " + "y" * 5])                    # 76 characters: one too many
    assert wraps.split("\n")[1:3] == ["│ " + "x" * 70 + " │", "│ " + "y" * 5 + " " * 65 + " │"]


def test_a_long_key_value_line_is_wrapped_under_the_value():
    words = [f"part{i}" for i in range(30)]
    lines = box(["Notes    " + " ".join(words)]).split("\n")
    assert len(lines) > 3 and len(widths("\n".join(lines))) == 1
    assert all(line.startswith("│ " + " " * 9 + "part") for line in lines[2:-1])     # under the value, not the key
    assert [word for line in lines[1:-1] for word in line[1:-1].split()] == ["Notes", *words]


def test_a_word_too_long_for_a_box_is_never_cut():
    path = "/Users/someone/" + "very-long-folder-name/" * 4 + "report.html"
    text = box([f"Report  {path}"])
    assert path in text and len(widths(text)) == 1             # the box grows instead


def test_a_table_given_as_one_text_becomes_several_lines():
    table = "   0   1\n0  28   2\n1   3  27"
    assert box([table]).split("\n")[1:-1] == ["│    0   1  │", "│ 0  28   2 │", "│ 1   3  27 │"]

# warnings and errors in bold red
RED, RESET = "\033[1;31m", "\033[0m"


def test_warnings_and_errors_can_be_bold_red():
    assert error("config file not found", color=True) == RED + "[mi-race] error: config file not found" + RESET
    assert warning("only 50 runs", color=True) == RED + "[mi-race] warning: only 50 runs" + RESET


def test_plain_text_is_the_default():
    assert "\033" not in error("x") + warning("x") + info("x")


def test_every_line_of_a_long_message_is_coloured_and_the_text_is_unchanged():
    text = " ".join(f"word{i}" for i in range(40)) + "\nsecond line"
    lines = error(text, color=True).split("\n")
    assert len(lines) > 2 and all(line.startswith(RED) and line.endswith(RESET) for line in lines)
    assert ANSI.sub("", error(text, color=True)) == error(text)


def test_colour_codes_take_no_columns():
    assert display_width(error("x", color=True)) == len("[mi-race] error: x")
    assert len(widths(box([error("inside a box", color=True), "plain"]))) == 1


class Terminal(io.StringIO):
    def isatty(self):
        return True


def test_colour_only_on_a_terminal_and_never_with_no_color(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert use_color(Terminal()) and not use_color(io.StringIO()) and not use_color(object())
    monkeypatch.setenv("NO_COLOR", "1")
    assert not use_color(Terminal())
    monkeypatch.setenv("NO_COLOR", "")                     # no-color.org: an empty value does not count
    assert use_color(Terminal())


# 7c: the confusion matrix
SMALL = {"confusion_matrix": [[28, 2], [3, 27]], "accuracy": 55 / 60, "macro_f1": 0.9166, "ba_bound_bits": 0.5}
FOREST = {"confusion_matrix": [[40, 17, 3, 0], [8, 39, 13, 0], [0, 6, 43, 11], [0, 0, 4, 56]],       # the example forest
          "accuracy": 0.742, "macro_f1": 0.740, "ba_bound_bits": 0.859}


def read_table(table, n):
    """the counts back from a table ('·' is 0), block by block"""
    lines = ANSI.sub("", table).split("\n")
    matrix = {}
    for i, line in enumerate(lines):
        if line.strip() == "predicted":
            columns = [int(v) for v in lines[i + 1].split()]
            for row in lines[i + 3:i + 3 + n]:
                u, *counts = row.replace("sent", "").split()
                matrix.update({(int(u), v): 0 if c == "·" else int(c) for v, c in zip(columns, counts)})
    return [[matrix[u, v] for v in range(n)] for u in range(n)]


def test_a_confusion_table_by_hand():
    assert [line.rstrip() for line in confusion_table(SMALL).split("\n")] == [
        "            predicted",
        "             0       1",
        "",
        "sent  0     28       2",
        "      1      3      27",
        "",
        "91.7% accuracy    0.917 macro F1    0.500 bits BA bound"]


def test_every_line_of_a_confusion_table_has_the_same_width():
    for color in (False, True):
        table = confusion_table(FOREST, color=color)
        assert len(widths(table)) == 1 and len(widths(box([table], "random forest"))) == 1
        assert read_table(table, 4) == FOREST["confusion_matrix"]


def test_only_the_correct_guesses_are_bold_green():
    assert CORRECT == "\033[1;32m"                                      # bold (1), green (32)
    lines = confusion_table({**SMALL, "confusion_matrix": [[30, 0], [3, 27]]}, color=True).split("\n")
    assert lines[3].startswith("sent  0  " + CORRECT + "   30   " + RESET + "    ·   ")
    assert lines[4].startswith("      1  " + "    3   " + CORRECT + "   27   " + RESET)
    codes = ANSI.findall(confusion_table(FOREST, color=True))
    assert codes.count(CORRECT) == 4 and len(codes) == 2 * (4 + 3)      # the diagonal and the 3 bold numbers, each reset


def test_colour_changes_nothing_but_the_colour():
    assert ANSI.sub("", confusion_table(FOREST, color=True)) == confusion_table(FOREST)
    assert "\033" not in confusion_table(FOREST)


def test_sent_is_written_on_the_middle_row():
    lines = confusion_table({**SMALL, "confusion_matrix": np.diag([9] * 5)}).split("\n")
    assert [line.startswith("sent") for line in lines[3:8]] == [False, False, True, False, False]


def test_many_symbols_are_split_into_blocks_that_fit_in_a_box():
    confusion = np.random.default_rng(0).integers(0, 100, size=(20, 20))
    table = confusion_table({**SMALL, "confusion_matrix": confusion})
    assert table.count("predicted") == 3                               # 3 blocks of columns
    assert max(widths(table)) <= WIDTH - 4 and max(widths(box([table]))) <= WIDTH
    assert read_table(table, 20) == confusion.tolist()


# 7d: banner and version box
def test_a_box_can_be_made_wider():
    assert widths(box(["a"], "t", min_width=10)) == {10}
    assert widths(box(["wider than ten"], min_width=10)) == {18}           # the content still wins


def test_the_banner_is_the_art_with_a_version_box_as_wide_as_it():
    lines = banner("configs/example.json", version="1.2.3").split("\n")
    art = BANNER.split("\n")
    assert lines[:len(art)] == art
    assert widths("\n".join(lines[len(art):])) == {max(display_width(line) for line in art)}
    assert "│ Version  1.2.3" in lines[len(art) + 1] and "│ Config   configs/example.json" in lines[len(art) + 2]
    assert all(display_width(line) < 80 for line in lines)


def test_the_banner_without_a_config_shows_only_the_version():
    text = banner()
    assert f"Version  {__version__}" in text and "Config" not in text


def test_is_terminal():
    assert is_terminal(Terminal()) and not is_terminal(io.StringIO()) and not is_terminal(object())