"""tests for src/mi_race/symbols.py"""

from mi_race.symbols import SymbolError, parse_vector, ask_symbols, write_symbols
from mi_race.config import ConfigError, load_config, read_config_file
import pytest
import io
from pathlib import Path

EXAMPLE = Path(__file__).parents[1] / "configs" / "example.json"

# 3a: one typed line -> a release vector
@pytest.mark.parametrize("line, expected", [
    ("50 0 0 0", (50, 0, 0, 0)),                    # spaces
    ("50,0,0,0", (50, 0, 0, 0)),                    # commas
    ("30, 0, 20, 0", (30, 0, 20, 0)),               # both
    ("  0   0 0 50  ", (0, 0, 0, 50)),              # extra spaces
])
def test_valid_line_becomes_a_vector(line, expected):
    assert parse_vector(line, n_slots=4, budget=50) == expected


@pytest.mark.parametrize("line, message", [
    ("", r"expected 4 numbers .* got 0"),                 # empty line
    ("50 0 0", r"expected 4 numbers .* got 3"),             # too few
    ("50 0 0 0 0", r"expected 4 numbers .* got 5"),         # too many
    ("50 0 0 x", r"'x' is not a valid amount"),             # not a number
    ("51 0 0 -1", r"'-1' is not a valid amount"),           # negative
    ("49.5 0.5 0 0", r"'49.5' is not a valid amount"),      # decimal
    ("49 0 0 0", r"add up to 49, but the budget is 50"),    # below the budget
    ("51 0 0 0", r"add up to 51, but the budget is 50"),    # above the budget
])
def test_invalid_line_is_rejected(line, message):
    with pytest.raises(SymbolError, match=message):
        parse_vector(line, n_slots=4, budget=50)


# 3b: the prompt loop
class FakeKeyboard:
    """answers every question with the next prepared line, and remembers the questions"""
    def __init__(self, *lines):
        self.lines = list(lines)
        self.questions = []

    def ask(self, question):
        self.questions.append(question)
        return self.lines.pop(0)

def test_one_good_line_per_symbol():
    keys = FakeKeyboard("50 0 0 0", "0 0 0 50")
    screen = []
    symbols = ask_symbols(n_symbols=2, n_slots=4, budget=50, ask=keys.ask, show=screen.append)
    assert symbols == ((50, 0, 0, 0), (0, 0, 0, 50))
    assert keys.questions == ["symbol 0: ", "symbol 1: "]
    assert screen == ["type 4 numbers for each symbol, adding up to 50"]    # no complaints

def test_wrong_line_shows_the_problem_and_asks_again():
    keys = FakeKeyboard("50 0 0", "50 0 0 0", "0 0 0 50")
    screen = []
    symbols = ask_symbols(n_symbols=2, n_slots=4, budget=50, ask=keys.ask, show=screen.append)
    assert symbols == ((50, 0, 0, 0), (0, 0, 0, 50))                        # the wrong line is not kept
    assert keys.questions == ["symbol 0: ", "symbol 0: ", "symbol 1: "]     # symbol 0 was asked twice
    assert "  expected 4 numbers (one per slot), got 3" in screen

def test_budget_is_checked_at_the_prompt():
    keys = FakeKeyboard("49 0 0 0", "51 0 0 0", "50 0 0 0", "0 0 0 50")
    screen = []
    ask_symbols(n_symbols=2, n_slots=4, budget=50, ask=keys.ask, show=screen.append)
    assert keys.questions.count("symbol 0: ") == 3
    assert "  the molecules add up to 49, but the budget is 50" in screen
    assert "  the molecules add up to 51, but the budget is 50" in screen

def test_stops_after_the_last_symbol():
    keys = FakeKeyboard("50 0 0 0", "0 0 0 50", "25 25 0 0")
    screen = []
    symbols = ask_symbols(n_symbols=2, n_slots=4, budget=50, ask=keys.ask, show=screen.append)
    assert len(symbols) == 2
    assert keys.lines == ["25 25 0 0"]                                      # never asked for

def test_uses_the_real_keyboard_and_screen_by_default(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("50 0 0 0\n0 0 0 50\n"))
    assert ask_symbols(n_symbols=2, n_slots=4, budget=50) == ((50, 0, 0, 0), (0, 0, 0, 50))
    assert "symbol 1: " in capsys.readouterr().out

def test_same_vector_twice_is_rejected():
    keys = FakeKeyboard("50 0 0 0", "50, 0, 0, 0", "0 50 0 0")      # the same vector, typed differently
    screen = []
    symbols = ask_symbols(n_symbols=2, n_slots=4, budget=50, ask=keys.ask, show=screen.append)
    assert symbols == ((50, 0, 0, 0), (0, 50, 0, 0))
    assert keys.questions == ["symbol 0: ", "symbol 1: ", "symbol 1: "]
    assert "  symbol 1 is the same as symbol 0: no decoder could tell them apart" in screen


# 3c: write the symbols into a config
NEW_SYMBOLS = (
    (25, 25, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 25, 25, 0, 0, 0)
)

@pytest.fixture
def config_file(tmp_path):
    """a copy of the example config that a test is free to change"""
    path = tmp_path / "exp.json"
    path.write_text(EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    return path

def test_new_symbols_are_saved(config_file):
    write_symbols(config_file, NEW_SYMBOLS)
    assert load_config(config_file).encoder.symbols == NEW_SYMBOLS

def test_nothing_else_in_the_file_changes(config_file):
    before = read_config_file(config_file)
    write_symbols(config_file, NEW_SYMBOLS)
    after = read_config_file(config_file)
    del before["encoder"]["symbols"], after["encoder"]["symbols"]
    assert after == before


def test_each_new_symbol_stays_on_one_line(config_file):
    write_symbols(config_file, NEW_SYMBOLS)
    assert "[25, 25, 0, 0, 0, 0, 0, 0, 0, 0],\n" in config_file.read_text(encoding="utf-8")


@pytest.mark.parametrize("symbols, message", [
    (NEW_SYMBOLS[:1], r"at least 2"),                                       # only one symbol
    (((50, 0, 0), (0, 0, 50)), r"has 3 entries"),                           # wrong number of slots
    ((NEW_SYMBOLS[0], (40, 0, 0, 0, 0, 0, 0, 0, 0, 0)), r"releases 40"),    # wrong budget
    ((NEW_SYMBOLS[0], NEW_SYMBOLS[0]), r"is the same as"),                  # duplicate
])
def test_bad_symbols_leave_the_file_untouched(config_file, symbols, message):
    before = config_file.read_text(encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        write_symbols(config_file, symbols)
    assert config_file.read_text(encoding="utf-8") == before


def test_a_problem_elsewhere_leaves_the_file_untouched(config_file):
    config_file.write_text(config_file.read_text(encoding="utf-8").replace('"T": 2.0', '"T": -2.0'), encoding="utf-8")
    before = config_file.read_text(encoding="utf-8")
    with pytest.raises(ConfigError, match=r"'channel.T'"):
        write_symbols(config_file, NEW_SYMBOLS)
    assert config_file.read_text(encoding="utf-8") == before


def test_old_symbols_do_not_have_to_be_valid(config_file):
    # n_slots changed from 10 to 4: the old symbols no longer fit, the new ones replace them
    config_file.write_text(config_file.read_text(encoding="utf-8").replace('"n_slots": 10', '"n_slots": 4'), encoding="utf-8")
    new = ((50, 0, 0, 0), (0, 0, 0, 50))
    write_symbols(config_file, new)
    assert load_config(config_file).encoder.symbols == new


def test_missing_file_is_a_config_error(tmp_path):
    with pytest.raises(ConfigError, match=r"not found"):
        write_symbols(tmp_path / "nope.json", NEW_SYMBOLS)


@pytest.mark.parametrize("text", ['{"schema_version": 1}', '{"encoder": 5}', "[1, 2]"])
def test_config_without_an_encoder_section_is_a_config_error(tmp_path, text):
    path = tmp_path / "exp.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match=r"has no 'encoder' section"):
        write_symbols(path, NEW_SYMBOLS)
    assert path.read_text(encoding="utf-8") == text


# 3d: the whole prompt, from the keyboard to the config file
def test_from_the_keyboard_to_the_config_file(config_file):
    encoder = load_config(config_file).encoder
    keys = FakeKeyboard(
        "25 25 0 0",                                                        # too few numbers
        "25 25 0 0 0 0 0 0 0 0",
        "25, 25, 0, 0, 0, 0, 0, 0, 0, 0",                                   # the same vector again
        "0 0 0 0 0 25 25 0 0 1",                                            # over the budget
        "0 0 0 0 0 25 25 0 0 0",
    )
    screen = []
    symbols = ask_symbols(2, encoder.n_slots, encoder.budget, ask=keys.ask, show=screen.append)
    write_symbols(config_file, symbols)
    assert load_config(config_file).encoder.symbols == NEW_SYMBOLS
    assert len(screen) == 4                                                 # the hint and 3 complaints