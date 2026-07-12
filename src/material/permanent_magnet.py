"""Linear recoil permanent-magnet material model."""
from __future__ import annotations

from dataclasses import dataclass, field
from math import isclose, isfinite

from .base import MU_0


@dataclass(frozen=True, slots=True)
class LinearPermanentMagnetMaterial:
    id: int
    name: str
    relative_permeability: float
    remanence: float
    coercive_field_strength: float | None = None
    consistency_relative_tolerance: float = field(default=1e-6, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.id < 0:
            raise ValueError("material id must be non-negative.")
        if not self.name:
            raise ValueError("material name must be non-empty.")
        mu_r = float(self.relative_permeability)
        br = float(self.remanence)
        if not isfinite(mu_r) or mu_r <= 0.0:
            raise ValueError("relative_permeability must be positive and finite.")
        if not isfinite(br) or br <= 0.0:
            raise ValueError("remanence must be positive and finite.")
        expected = br / (MU_0 * mu_r)
        hc = expected if self.coercive_field_strength is None else float(self.coercive_field_strength)
        if not isfinite(hc) or hc <= 0.0:
            raise ValueError("coercive_field_strength must be positive and finite.")
        if self.coercive_field_strength is not None and not isclose(hc, expected, rel_tol=self.consistency_relative_tolerance, abs_tol=0.0):
            raise ValueError("coercive_field_strength is inconsistent with remanence and relative_permeability.")
        object.__setattr__(self, "coercive_field_strength", hc)

    @property
    def permeability(self) -> float:
        return MU_0 * float(self.relative_permeability)

    @property
    def is_permanent_magnet(self) -> bool:
        return True
