"""Magnetization directions in the axisymmetric r-z plane.

Positive radial direction means increasing r; positive axial direction means increasing z.
The optional circumferential component is used by unwrapped phi-z models.
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
    CIRCUMFERENTIAL_POSITIVE = auto()
    CIRCUMFERENTIAL_NEGATIVE = auto()


@dataclass(frozen=True, slots=True)
class MagnetizationDirection:
    radial: float
    axial: float
    circumferential: float = 0.0

    def __post_init__(self) -> None:
        r, z, c = float(self.radial), float(self.axial), float(self.circumferential)
        if not isfinite(r) or not isfinite(z) or not isfinite(c):
            raise ValueError("magnetization components must be finite.")
        if hypot(hypot(r, z), c) == 0.0:
            raise ValueError("magnetization vector must be nonzero.")
        object.__setattr__(self, "radial", r)
        object.__setattr__(self, "axial", z)
        object.__setattr__(self, "circumferential", c)

    @property
    def unit_vector(self) -> tuple[float, float]:
        mag = hypot(self.radial, self.axial)
        if mag == 0.0:
            return (0.0, 0.0)
        return (self.radial / mag, self.axial / mag)

    @property
    def unit_vector_3d(self) -> tuple[float, float, float]:
        mag = hypot(hypot(self.radial, self.axial), self.circumferential)
        return (self.radial / mag, self.circumferential / mag, self.axial / mag)

    @classmethod
    def from_axis(cls, axis: MagnetizationAxis) -> "MagnetizationDirection":
        return {
            MagnetizationAxis.RADIAL_POSITIVE: cls(1.0, 0.0),
            MagnetizationAxis.RADIAL_NEGATIVE: cls(-1.0, 0.0),
            MagnetizationAxis.AXIAL_POSITIVE: cls(0.0, 1.0),
            MagnetizationAxis.AXIAL_NEGATIVE: cls(0.0, -1.0),
            MagnetizationAxis.CIRCUMFERENTIAL_POSITIVE: cls(0.0, 0.0, 1.0),
            MagnetizationAxis.CIRCUMFERENTIAL_NEGATIVE: cls(0.0, 0.0, -1.0),
        }[axis]


def coerce_magnetization_direction(value: MagnetizationDirection | MagnetizationAxis) -> MagnetizationDirection:
    if isinstance(value, MagnetizationDirection):
        return value
    if isinstance(value, MagnetizationAxis):
        return MagnetizationDirection.from_axis(value)
    raise TypeError("magnetization must be a MagnetizationDirection or MagnetizationAxis.")
