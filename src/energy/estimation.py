"""Energy *estimation* interface: Energy = Power x Time.

IMPORTANT EPISTEMIC BOUNDARY
----------------------------
This module produces ESTIMATES, not physical measurements.

Energy cannot be measured directly in Prototype V0.1: we do not attach power
meters to the CPU/RAM. Instead we combine a *measured* execution time with a
documented, configurable power convention (typical TDP of the machine class)
to yield an estimate in Joules.

To keep the distinction crisp:

* measured  -> ``src.energy.measurement`` (time, CPU %, RSS memory).
* estimated -> this module (energy derived from measured time x power model).

Callers MUST label results from this module as estimates in any output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.config import DEFAULT_CPU_POWER_WATTS, DEFAULT_RAM_POWER_WATTS
from src.energy.measurement import Measurement

logger = logging.getLogger(__name__)

_JOULES_PER_WATT_SECOND = 1.0


@dataclass(frozen=True)
class PowerModel:
    """Power-convention assumed for the host machine."""

    cpu_watts: float
    ram_watts: float
    description: str

    @classmethod
    def default(cls) -> "PowerModel":
        """Return the default host power convention documented in config.

        The values are *typical* TDP figures for a mid-range desktop CPU and
        DIMM, chosen as a reproducible convention - they are NOT measured.
        """
        return cls(
            cpu_watts=DEFAULT_CPU_POWER_WATTS,
            ram_watts=DEFAULT_RAM_POWER_WATTS,
            description=(
                "Typical TDP convention (not measured): "
                f"{DEFAULT_CPU_POWER_WATTS} W CPU + {DEFAULT_RAM_POWER_WATTS} W RAM"
            ),
        )


@dataclass(frozen=True)
class EnergyEstimate:
    """Result of applying a power model to a measurement."""

    energy_joules: float
    cpu_energy_joules: float
    ram_energy_joules: float
    wall_time_seconds: float
    power_model: PowerModel
    is_estimated: bool = True

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict with explicit 'estimated' flag."""
        return {
            "energy_joules_ESTIMATED": round(self.energy_joules, 6),
            "cpu_energy_joules_ESTIMATED": round(self.cpu_energy_joules, 6),
            "ram_energy_joules_ESTIMATED": round(self.ram_energy_joules, 6),
            "wall_time_seconds_measured": round(self.wall_time_seconds, 6),
            "power_model": self.power_model.description,
            "is_estimated": self.is_estimated,
        }


def estimate_energy_consumption(
    measurement: Measurement,
    power_model: PowerModel | None = None,
) -> EnergyEstimate:
    """Estimate energy (Joules) from a measured execution using Energy = P x T.

    Parameters
    ----------
    measurement : object produced by ``src.energy.measurement``.
    power_model : optional power convention; defaults to :meth:`PowerModel.default`.

    Returns
    -------
    EnergyEstimate labelled as an estimate. The estimate is a research proxy,
    NOT physical energy consumption.
    """
    if power_model is None:
        power_model = PowerModel.default()

    wall_time = measurement.wall_time_seconds
    cpu_energy = power_model.cpu_watts * wall_time * _JOULES_PER_WATT_SECOND
    ram_energy = power_model.ram_watts * wall_time * _JOULES_PER_WATT_SECOND
    total = cpu_energy + ram_energy

    logger.info(
        "Estimated energy %.4f J from %.4f s x power model %.1f/%.1f W",
        total,
        wall_time,
        power_model.cpu_watts,
        power_model.ram_watts,
    )
    return EnergyEstimate(
        energy_joules=total,
        cpu_energy_joules=cpu_energy,
        ram_energy_joules=ram_energy,
        wall_time_seconds=wall_time,
        power_model=power_model,
    )