"""
Load a config file and check it against the rules in SPEC.md
(config validation)
"""

from dataclasses import dataclass

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