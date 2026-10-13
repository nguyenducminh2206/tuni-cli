"""
Load a config file and check it against the rules in SPEC.md
(config validation)
"""

from dataclasses import dataclass, field

RECOMMENDED_RUNS = 300

class ConfigError(ValueError):
    """The config breaks a rule in SPEC.md"""


@dataclass(frozen=True)
class ChannelConfig:
    """The channel section: a tube of compartments with an absorbing receiver"""
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
    """The encoder section: when each symbol releases its molecules"""
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
    """The data section: where the dataset goes and how it is split"""
    path: str                                   # where the simulated dataset is saved
    runs_per_symbol: int = RECOMMENDED_RUNS     # simulations per symbol
    test_fraction: float = 0.2                  # share of runs kept asidde for testing


@dataclass(frozen=True)
class OutputConfig:
    """The output section: where results are written"""
    dir: str = "outputs"

@dataclass(frozen=True)
class Config:
    """A complete, checked config: one field per section of the json file"""
    schema_version: int
    channel: ChannelConfig
    encoder: EncoderConfig
    data: DataConfig
    seed: int = 0                                   # master seed for all randomness
    decoders: dict = field(default_factory=dict)    
    output: OutputConfig = field(default_factory=OutputConfig)