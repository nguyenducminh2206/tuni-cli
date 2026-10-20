from mi_race.config import ConfigError, _parse_symbols
print(_parse_symbols([[50, 0, 0], [0, 20, 30]], n_slots=3, budget=50))
tries = [
    [[50, 0, 0]],
    [[50, 0, 0], "anc"]
]

for symbols in tries:
    try: 
        _parse_symbols(symbols, n_slots=3, budget=50)
    except ConfigError as e:
        print("ConfigError:", e)
