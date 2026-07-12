"""Solver components for UniversalMRN."""

from .excitation import MagneticExcitation, build_magnetic_excitation
from .linear import solve_linear_mrn
from .solution import LinearMagneticSolution

__all__ = [
    "LinearMagneticSolution",
    "MagneticExcitation",
    "build_magnetic_excitation",
    "solve_linear_mrn",
]
