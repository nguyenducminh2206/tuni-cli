"""
Load a config file and check it against the rules in SPEC.md
(config validation)
"""

from dataclasses import dataclass, field
from pathlib import Path
import json
import difflib
import math
import warnings

SCHEMA_VERSION = 1
RECOMMENDED_RUNS = 300
_TOL = 1e-9
DECODER_FIELDS = {
    "random_forest": ("n_estimators",),
    "mlp": ("hidden_layers", "epochs", "lr", "batch_size"),
    "cnn": ("channels", "kernel_size", "epochs", "lr", "batch_size")
}


class ConfigError(ValueError):
    """the config breaks a rule in SPEC.md"""


@dataclass(frozen=True)
class ChannelConfig:
    """the channel section: a tube of compartments with an absorbing receiver"""
    L: int      # positions: comp 0 ... L-2, receiver at L-1
    h: float    # comp width -> wider comp mean a molecule needs loger to cross one -> fewer jumps | unit: µm
    D: float    # diffusion coefficient -> bigger D means faster spreading -> more jumps | unit: µ²m/s
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
    test_fraction: float = 0.2                  # share of runs kept aside for testing


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


def _parse_encoder(raw: object, channel: ChannelConfig) -> EncoderConfig:
    """check the encoder section and build an EncoderConfig"""
    _check_keys(raw, "encoder", required=("n_slots", "slot_dt", "budget", "symbols"))
    n_slots = _whole(raw["n_slots"], "encoder.n_slots", minimum=1)
    slot_dt = _positive(raw["slot_dt"], "encoder.slot_dt")
    budget = _whole(raw["budget"], "encoder.budget", minimum=1)

    last_release = (n_slots - 1) * slot_dt
    if last_release >= channel.T * (1 - _TOL):
        raise ConfigError(
            f"the last slot releases at t = {last_release:g}s, but every release must happen before 'channel.T' = {channel.T:g}s. "
            f"Use fewer 'n_slots' or a smaller 'slot_dt'")
    symbols = _parse_symbols(raw["symbols"], n_slots, budget)
    return EncoderConfig(n_slots=n_slots, slot_dt=slot_dt, budget=budget, symbols=symbols)


def _parse_data(raw: object) -> DataConfig:
    """check the data section and build DataConfig"""
    _check_keys(raw, "data", required=("path",), optional=("runs_per_symbol", "test_fraction"))
    path = raw["path"]
    if not isinstance(path, str) or not path.strip():
        raise ConfigError(f"'data.path' must be a file path like 'data/example.csv', got {path!r}")
    runs = _whole(raw.get("runs_per_symbol", DataConfig.runs_per_symbol), "data.runs_per_symbol", minimum=1)
    fraction = _positive(raw.get("test_fraction", DataConfig.test_fraction), "data.test_fraction")
    if fraction >= 1:
        raise ConfigError(f"'data.test_fraction' must be between 0 and 1, got {fraction:g}")

    test_runs = runs * fraction
    if test_runs < 1 - _TOL or runs - test_runs < 1 - _TOL:
        raise ConfigError(
            f"'data.runs_per_symbol' = {runs} is too small for 'data_test_fraction' = {fraction:g} "
            "every symbol needs at least one training run and one test run"
        )
    if runs < RECOMMENDED_RUNS:
        warnings.warn(
            f"'data.runs_per_symbol' = {runs} is below {RECOMMENDED_RUNS}: fine for quick test, "
            "but results (especially MI) will be noisy"
        )
    
    return DataConfig(path=path, runs_per_symbol=runs, test_fraction=fraction)


def _parse_output(raw: object) -> OutputConfig:
    """check the output section and build and OutputConfig"""
    _check_keys(raw, "output", required=(), optional=("dir",))
    directory = raw.get("dir", OutputConfig.dir)
    if not isinstance(directory, str) or not directory.strip():
        raise ConfigError(f"'output.dir' must be a folder path like 'outputs', got {directory!r}")
    return OutputConfig(dir=directory)


def _parse_decoders(raw: object) -> dict:
    """check decoder names and their setting names; the values are checked in step 5"""
    _check_keys(raw, "decoders", required=(), optional=tuple(DECODER_FIELDS))
    for name, settings in raw.items():
        _check_keys(settings, f"decoders.{name}", required=(), optional=DECODER_FIELDS[name])
    return {name: dict(settings) for name, settings in raw.items()}


def load_config(path: str | Path) -> Config:
    """read a config gile and check everything: the one way to get a Config"""
    raw = read_config_file(path)
    try:
        return parse_config(raw)
    except ConfigError as e:
        raise ConfigError(f"{path}: {e}") from None

def parse_config(raw: object) -> Config:
    """check a whole config and build a Config"""
    if not isinstance(raw, dict):
        raise ConfigError(f"the config must be a json object, not {type(raw).__name__}")
    version = raw.get("schema_version")
    if isinstance(version, bool) or version != SCHEMA_VERSION:
        raise ConfigError(f"'schema_version' must be {SCHEMA_VERSION}, got {version!r}")
    _check_keys(
        raw, "config", 
        required=("schema_version", "channel", "encoder", "data"),
        optional=("seed", "decoders", "output"),
    )
    channel = _parse_channel(raw["channel"])
    return Config(
        schema_version=SCHEMA_VERSION,
        channel=channel,
        encoder=_parse_encoder(raw["encoder"], channel),
        data=_parse_data(raw["data"]),
        seed=_whole(raw.get("seed", Config.seed), "seed", minimum=0),
        decoders=_parse_decoders(raw.get("decoders", {})),
        output=_parse_output(raw.get("output", {}))
    )