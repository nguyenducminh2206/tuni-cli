from pathlib import Path

from mi_race.ui import PREFIX, WIDTH, box, display_width, error, info, warning


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