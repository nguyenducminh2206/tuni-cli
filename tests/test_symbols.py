"""tests for src/mi_race/symbols.py"""

from mi_race.symbols import SymbolError, parse_vector, ask_symbols
import pytest
import io

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