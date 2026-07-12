"""Linear non-permanent magnetic material model."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .base import MU_0


@dataclass(frozen=True, slots=True)
class LinearMagneticMaterial:
    id: int
    name: str
    relative_permeability: float

    def __post_init__(self) -> None:
        if self.id < 0:
            raise ValueError("material id must be non-negative.")
        if not self.name:
            raise ValueError("material name must be non-empty.")
        if not isfinite(float(self.relative_permeability)) or float(self.relative_permeability) <= 0.0:
            raise ValueError("relative_permeability must be positive and finite.")

    @property
    def permeability(self) -> float:
        return MU_0 * float(self.relative_permeability)

    @property
    def is_permanent_magnet(self) -> bool:
        return False
