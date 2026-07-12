"""Magnetic material models for UniversalMRN."""
from .base import MU_0, MagneticMaterial
from .linear import LinearMagneticMaterial
from .magnetization import MagnetizationAxis, MagnetizationDirection
from .permanent_magnet import LinearPermanentMagnetMaterial

__all__ = [
    "MU_0",
    "MagneticMaterial",
    "LinearMagneticMaterial",
    "LinearPermanentMagnetMaterial",
    "MagnetizationAxis",
    "MagnetizationDirection",
]
