"""Magnetization directions in the axisymmetric r-z plane.

Positive radial direction means increasing r; positive axial direction means increasing z.
No circumferential component is represented in the current two-dimensional model.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from math import hypot, isfinite


class MagnetizationAxis(Enum):
    RADIAL_POSITIVE = auto()
    RADIAL_NEGATIVE = auto()
    AXIAL_POSITIVE = auto()
    AXIAL_NEGATIVE = auto()


@dataclass(frozen=True, slots=True)
class MagnetizationDirection:
    radial: float
    axial: float

    def __post_init__(self) -> None:
        r, z = float(self.radial), float(self.axial)
        if not isfinite(r) or not isfinite(z):
            raise ValueError("magnetization components must be finite.")
        if hypot(r, z) == 0.0:
            raise ValueError("magnetization vector must be nonzero.")
        object.__setattr__(self, "radial", r)
        object.__setattr__(self, "axial", z)

    @property
    def unit_vector(self) -> tuple[float, float]:
        mag = hypot(self.radial, self.axial)
        return (self.radial / mag, self.axial / mag)

    @classmethod
    def from_axis(cls, axis: MagnetizationAxis) -> "MagnetizationDirection":
        return {
            MagnetizationAxis.RADIAL_POSITIVE: cls(1.0, 0.0),
            MagnetizationAxis.RADIAL_NEGATIVE: cls(-1.0, 0.0),
            MagnetizationAxis.AXIAL_POSITIVE: cls(0.0, 1.0),
            MagnetizationAxis.AXIAL_NEGATIVE: cls(0.0, -1.0),
        }[axis]


def coerce_magnetization_direction(value: MagnetizationDirection | MagnetizationAxis) -> MagnetizationDirection:
    if isinstance(value, MagnetizationDirection):
        return value
    if isinstance(value, MagnetizationAxis):
        return MagnetizationDirection.from_axis(value)
    raise TypeError("magnetization must be a MagnetizationDirection or MagnetizationAxis.")
