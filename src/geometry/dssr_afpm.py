"""Parametric geometry description for a DSSR axial-flux PM machine.

The geometry is represented in an ``r-z`` cross-section.  It is intentionally
independent of the mesh resolution so that the same physical model can be
filled by a coarse or fine non-parametric mesh.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class MaterialRegion(IntEnum):
    """Integer material identifiers stored in mesh cells."""

    OUTER_AIR = 0
    STATOR_LEFT = 1
    WINDING_LEFT = 2
    AIR_GAP_LEFT = 3
    ROTOR_CORE = 4
    PERMANENT_MAGNET = 5
    AIR_GAP_RIGHT = 6
    WINDING_RIGHT = 7
    STATOR_RIGHT = 8


@dataclass(frozen=True, slots=True)
class DSSRAFPMGeometry:
    """Simplified symmetric DSSR-AFPM geometry in the ``r-z`` plane.

    All dimensions are expressed in metres.  The rotor mid-plane is ``z=0``.
    The first milestone uses annular axial layers.  Tooth, slot and spoke
    segmentation can be added later without changing the mesh generator.
    """

    inner_radius: float = 0.090
    outer_radius: float = 0.1505
    rotor_thickness: float = 0.012
    magnet_thickness: float = 0.008
    air_gap: float = 0.001
    winding_thickness: float = 0.012
    stator_back_iron_thickness: float = 0.015
    radial_air_margin: float = 0.005
    axial_air_margin: float = 0.005

    def __post_init__(self) -> None:
        positive = {
            "inner_radius": self.inner_radius,
            "outer_radius": self.outer_radius,
            "rotor_thickness": self.rotor_thickness,
            "magnet_thickness": self.magnet_thickness,
            "air_gap": self.air_gap,
            "winding_thickness": self.winding_thickness,
            "stator_back_iron_thickness": self.stator_back_iron_thickness,
        }
        for name, value in positive.items():
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.outer_radius <= self.inner_radius:
            raise ValueError("outer_radius must be greater than inner_radius")
        if self.magnet_thickness > self.rotor_thickness:
            raise ValueError("magnet_thickness cannot exceed rotor_thickness")
        if self.radial_air_margin < 0.0 or self.axial_air_margin < 0.0:
            raise ValueError("air margins cannot be negative")

    @property
    def rotor_half_thickness(self) -> float:
        return self.rotor_thickness / 2.0

    @property
    def model_r_min(self) -> float:
        return max(0.0, self.inner_radius - self.radial_air_margin)

    @property
    def model_r_max(self) -> float:
        return self.outer_radius + self.radial_air_margin

    @property
    def stator_outer_z(self) -> float:
        return (
            self.rotor_half_thickness
            + self.air_gap
            + self.winding_thickness
            + self.stator_back_iron_thickness
        )

    @property
    def model_z_min(self) -> float:
        return -self.stator_outer_z - self.axial_air_margin

    @property
    def model_z_max(self) -> float:
        return self.stator_outer_z + self.axial_air_margin

    def classify(self, r: float, z: float) -> MaterialRegion:
        """Return the material region containing point ``(r, z)``."""

        if not (self.inner_radius <= r <= self.outer_radius):
            return MaterialRegion.OUTER_AIR

        abs_z = abs(z)
        rotor_half = self.rotor_half_thickness
        gap_outer = rotor_half + self.air_gap
        winding_outer = gap_outer + self.winding_thickness
        stator_outer = winding_outer + self.stator_back_iron_thickness

        if abs_z <= rotor_half:
            if abs_z <= self.magnet_thickness / 2.0:
                return MaterialRegion.PERMANENT_MAGNET
            return MaterialRegion.ROTOR_CORE

        if abs_z <= gap_outer:
            return (
                MaterialRegion.AIR_GAP_LEFT
                if z < 0.0
                else MaterialRegion.AIR_GAP_RIGHT
            )

        if abs_z <= winding_outer:
            return (
                MaterialRegion.WINDING_LEFT
                if z < 0.0
                else MaterialRegion.WINDING_RIGHT
            )

        if abs_z <= stator_outer:
            return (
                MaterialRegion.STATOR_LEFT
                if z < 0.0
                else MaterialRegion.STATOR_RIGHT
            )

        return MaterialRegion.OUTER_AIR
