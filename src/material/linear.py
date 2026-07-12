"""Linear isotropic magnetic material definitions."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping

MU_0 = 4.0e-7 * 3.141592653589793


@dataclass(frozen=True, slots=True)
class LinearMagneticMaterial:
    """Linear isotropic material used by the first physical MRN model.

    The model intentionally supports only a constant relative permeability.
    Nonlinear B-H curves and permanent-magnet source terms belong to later
    solver phases.
    """

    id: int
    name: str
    relative_permeability: float

    def __post_init__(self) -> None:
        if self.id < 0:
            raise ValueError("material id must be non-negative.")
        if not self.name.strip():
            raise ValueError("material name must not be empty.")
        if not isfinite(self.relative_permeability) or self.relative_permeability <= 0.0:
            raise ValueError("relative_permeability must be positive and finite.")

    @property
    def permeability(self) -> float:
        """Return absolute permeability in H/m."""

        return MU_0 * self.relative_permeability


def validate_material_library(
    materials: Mapping[int, LinearMagneticMaterial], required_material_ids: set[int]
) -> None:
    """Validate that every required material id has one consistent entry."""

    missing = required_material_ids.difference(materials)
    if missing:
        raise ValueError(f"missing material definitions for ids: {tuple(sorted(missing))}.")
    inconsistent = tuple(sorted(key for key, material in materials.items() if key != material.id))
    if inconsistent:
        raise ValueError(f"material mapping keys do not match material ids: {inconsistent}.")
