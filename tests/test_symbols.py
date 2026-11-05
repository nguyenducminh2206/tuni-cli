"""tests for src/mi_race/symbols.py"""

from mi_race.symbols import SymbolError, parse_vector
import pytest


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