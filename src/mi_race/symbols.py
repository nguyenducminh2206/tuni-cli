"""symbols: the logic of the interactive prompt (typed lines -> release vectors -> config)"""

from collections.abc import Callable
from pathlib import Path
from mi_race.config import ConfigError, read_config_file, parse_config, write_config_file

class SymbolError(ValueError):
    """a typed release vector breaks a rule (the prompt shows the message and asks again)"""


def parse_pairs(parts: list[str], n_slots: int) -> tuple[int, ...]:
    """'amount:slot' pairs -> a release vector; slots not named release nothing.
    For ex, with n_slots=4: ["25:1", "25:2"] -> (0, 25, 25, 0)"""
    vector = [0] * n_slots
    named = set()
    for part in parts:
        amount, colon, slot = part.partition(":")
        if not (colon and amount.isdecimal() and slot.isdecimal()):
            raise SymbolError(f"{part!r} is not amount:slot, e.g. 50:3 for 50 molecules at slot 3")
        if int(slot) >= n_slots:
            raise SymbolError(f"there is no slot {int(slot)}: the slots are 0 ... {n_slots - 1}")
        if int(slot) in named:
            raise SymbolError(f"slot {int(slot)} is named twice")
        named.add(int(slot))
        vector[int(slot)] = int(amount)
    return tuple(vector)


def parse_vector(line: str, n_slots: int, budget: int) -> tuple[int, ...]:
    """turn one typed line into a release vector, or SymbolError. Two ways to type it, with n_slots=4, budget=50:
    amount:slot pairs, only where molecules go: "50:0" -> (50, 0, 0, 0), "25:1 25:2" -> (0, 25, 25, 0)
    or one number per slot, separated by spaces or commas: "50 0 0 0" -> (50, 0, 0, 0)
    """
    parts = line.replace(",", " ").split()
    if any(":" in part for part in parts):
        vector = parse_pairs(parts, n_slots)
    else:
        if len(parts) != n_slots:
            raise SymbolError(f"expected {n_slots} numbers (one per slot), got {len(parts)!r}")
        for part in parts:
            if not part.isdecimal():
                raise SymbolError(f"{part!r} is not a valid amount: use a whole number, 0 or more")
        vector = tuple(int(part) for part in parts)
    if sum(vector) != budget:
        raise SymbolError(f"the molecules add up to {sum(vector)!r}, but the budget is {budget!r}")
    return vector


def ask_symbols(n_symbols: int, n_slots:int, budget:int, 
                ask: Callable[[str], str]=input, show: Callable[[str], None]=print) -> tuple[tuple[int, ...], ...]:
    """ask for one release vector per symbol; a wrong line shows the problem and asks again
    ask and show are the keyboard and the screen by default; tests pass their own
    """
    show(f"type {n_slots} numbers for each symbol, adding up to {budget}")
    vectors = []
    for u in range(n_symbols):
        while True:
            line = ask(f"symbol {u}: ")
            try:
                vector = parse_vector(line, n_slots, budget)
                if vector in vectors:
                    raise SymbolError(f"symbol {u} is the same as symbol {vectors.index(vector)}: no decoder could tell them apart")
                vectors.append(vector)
                break               # this symbol is done: go to the next one
            except SymbolError as e:
                show(f"  {e}")      # say what is wrong, then ask again
    return tuple(vectors)


def write_symbols(path: str | Path, symbols: tuple[tuple[int, ...], ...]) -> None:
    """put new symbols into a config file; the file only changes if the whole config is still valid"""
    raw = read_config_file(path)
    if not isinstance(raw, dict) or not isinstance(raw.get("encoder"), dict):
        raise ConfigError(f"{path} has no 'encoder' section")
    raw["encoder"]["symbols"] = [list(vector) for vector in symbols]        # json has lists, not tuple
    parse_config(raw)                                                       # a problem stops here, before the file is touched
    write_config_file(path, raw)
