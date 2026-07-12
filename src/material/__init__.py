"""Magnetic material models for UniversalMRN."""

from .linear import MU_0, LinearMagneticMaterial, validate_material_library

__all__ = ["MU_0", "LinearMagneticMaterial", "validate_material_library"]
