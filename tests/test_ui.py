from pathlib import Path

from mi_race.ui import PREFIX, error, info, warning


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