"""symbols: the logic of the interactive prompt (typed lines -> release vectors -> config)"""

from collections.abc import Callable


class SymbolError(ValueError):
    """a typed release vector breaks a rule (the prompt shows the message and asks again)"""


def parse_vector(line: str, n_slots: int, budget: int) -> tuple[int, ...]:
    """turn one typed line into a release vector, or SymbolError
    numbers are separated by spaces or commas. For ex, with n_slots=4 and budget=50:
    "50 0 0 0" -> (50, 0, 0, 0)
    """
    parts = line.replace(",", " ").split()
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
