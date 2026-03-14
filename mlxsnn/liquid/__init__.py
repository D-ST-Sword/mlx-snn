"""Liquid State Machine (LSM) module.

Provides reservoir computing with spiking neurons:
- :class:`LiquidReservoir` — random sparse recurrent network
- :class:`LSM` — complete reservoir + readout system
- Topology generators for reservoir connectivity
"""

from __future__ import annotations

from mlxsnn.liquid.reservoir import LiquidReservoir
from mlxsnn.liquid.lsm import LSM
from mlxsnn.liquid.topology import (
    erdos_renyi,
    small_world,
    scale_free,
)

__all__ = [
    "LiquidReservoir",
    "LSM",
    "erdos_renyi",
    "small_world",
    "scale_free",
]
