"""
Load a config file and check it against the rules in SPEC.md
(config validation)
"""

from dataclasses import dataclass, field
from pathlib import Path
import json
import difflib
import math

RECOMMENDED_RUNS = 300
_TOL = 1e-9

class ConfigError(ValueError):
    """the config breaks a rule in SPEC.md"""


@dataclass(frozen=True)
class ChannelConfig:
    """the channel section: a tube of compartments with an absorbing receiver"""
    L: int      # positions: comp 0 ... L-2, receiver at L-1
    h: float    # comp width
    D: float    # diffusion coefficient
    dt: float   # snapshot interval
    T: float    # duration

    @property
    def jump_rate(self) -> float:
        """λ = D/h²: jump rate of 1 molecule to each neighbour"""
        return self.D / self.h**2

    @property
    def n_intervals(self) -> int:
        """M = T/dt, so the receiver signal is rx_0, rx_M"""
        return round(self.T / self.dt)


@dataclass(frozen=True)
class EncoderConfig:
    """the encoder section: when each symbol releases its molecules"""
    n_slots: int                            # number of release slots
    slot_dt: float                          # time between slots: slot k releases at t = k . slot_dt
    budget: int                             # molecules per symbol
    symbols: tuple[tuple[int, ...], ...]    # symbols[u][k] = molecules symbol u releases in slot k 

    @property
    def n_symbols(self) -> int:
        """N, the number of symbols (one release vector each)"""  
        return len(self.symbols)


@dataclass(frozen=True)
class DataConfig:
    """the data section: where the dataset goes and how it is split"""
    path: str                                   # where the simulated dataset is saved
    runs_per_symbol: int = RECOMMENDED_RUNS     # simulations per symbol
    test_fraction: float = 0.2                  # share of runs kept asidde for testing


@dataclass(frozen=True)
class OutputConfig:
    """the output section: where results are written"""
    dir: str = "outputs"

@dataclass(frozen=True)
class Config:
    """a complete, checked config: one field per section of the json file"""
    schema_version: int
    channel: ChannelConfig
    encoder: EncoderConfig
    data: DataConfig
    seed: int = 0                                   # master seed for all randomness
    decoders: dict = field(default_factory=dict)    
    output: OutputConfig = field(default_factory=OutputConfig)


def read_config_file(path: str | Path) -> dict:
    """read json config file into a dict (values are not checked yet)"""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {path}") from None
    try: 
        return json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as e:
        raise ConfigError(f"{path} is not valid json: {e}") from None


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    """not allow a key appears twice in json"""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ConfigError(f"key {key!r} appears twice in the same section")
        result[key] = value
    return result

def _check_keys(
        section: object, where: str, required: tuple[str, ...], optional: tuple[str, ...] = ()
) -> None:
    """check that 'section' is a json object with every required key and no unknown"""
    if not isinstance(section, dict):
        raise ConfigError(f"{where!r} must be a json object, not {type(section).__name__}")
    allowed = list(required) + list(optional)
    for key in section:
        if key not in allowed:
            guess = _closest(key, allowed)
            hint = f" Did you mean '{guess}'?" if guess else f" Allowed: {', '.join(allowed)}"
            raise ConfigError(f"unknown key '{key}' in {where!r}.{hint}")
    missing = [key for key in required if key not in section]
    if missing:
        raise ConfigError(f"{where!r} is missing: {', '.join(missing)}")

def _closest(key: str, allowed: list[str]) -> str | None:
    """the allowed key that 'key' was probably meant to be, or None"""
    for name in allowed:
        if name.lower() == key.lower():
            return name
    close = difflib.get_close_matches(key, allowed, n=1)
    return close[0] if close else None

def _whole(value: object, where: str, minimum: int) -> int:
    """a whole number of at least 'minimum' or ConfigError"""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(
            f"{where!r} must be a whole number, got {value!r}"
        )
    if value < minimum:
        raise ConfigError(f"{where!r} must be at least {minimum}, got {value}")
    return value

def _positive(value: object, where: str) -> float:
    """a finite number greater than 0, returned as a float, or ConfigError"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where!r} must be a number, got {value!r}")
    if not math.isfinite(value) or value <= 0:
        raise ConfigError(f"{where!r} must be a finite number greater than 0, got {value}")
    return float(value)


def _parse_channel(raw:object) -> ChannelConfig:
    """check the channel section and build a ChannelConfig"""
    _check_keys(raw, "channel", required=("L", "h", "D", "dt", "T"))
    L = _whole(raw["L"], "channel.L", minimum=2)
    h = _positive(raw["h"], "channel.h")
    D = _positive(raw["D"], "channel.D")
    dt = _positive(raw["dt"], "channel.dt")
    T = _positive(raw["T"], "channel.T")

    steps = T / dt
    nearest = round(steps)
    if nearest < 1 or abs(steps - nearest) > _TOL * steps:
        raise ConfigError(
            f"'channel.T' / 'channel.dt' must be a whole number of steps, got {T:g} / {dt:g} = {steps:g}"
        )
    return ChannelConfig(L=L, h=h, D=D, dt=dt, T=T)


def _parse_symbols(raw: object, n_slots: int, budget: int) -> tuple[tuple[int, ...], ...]:
    """check the release vectors and return them as tuples (symbol u = u-th vector)"""
    if not isinstance(raw, list) or len(raw) < 2:
        raise ConfigError("'encoder.symbols' must be a list of at least 2 release vectors")
    vectors = []
    for u, vector in enumerate(raw):
        where = f"encoder.symbols[{u}]"
        if not isinstance(vector, list):
            raise ConfigError(f"{where!r} must be a list of {n_slots} whole numbers")
        if len(vector) != n_slots:
            raise ConfigError(f"{where!r} has {len(vector)} entries, but 'n_slots' = {n_slots}")
        amounts = tuple(_whole(x, f'{where}[{k}]', minimum=0) for k, x in enumerate(vector))
        if sum(amounts) != budget:
            raise ConfigError(f"{where!r} releases {sum(amounts)} molecules, but the 'budget' is {budget}")
        vectors.append(amounts)
    return tuple(vectors)
        