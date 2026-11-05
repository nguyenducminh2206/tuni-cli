"""symbols: the logic of the interactive prompt (typed lines -> release vectors -> config)"""

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