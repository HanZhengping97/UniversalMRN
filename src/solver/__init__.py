"""Solver components for UniversalMRN."""

from .excitation import MagneticExcitation, build_magnetic_excitation
from .linear import solve_linear_mrn
from .solution import LinearMagneticSolution

__all__ = [
    "LinearMagneticSolution",
    "MagneticExcitation",
    "build_magnetic_excitation",
    "solve_linear_mrn",
    "SegmentFieldState",
    "BranchFieldState",
    "MagneticFieldSolution",
    "recover_magnetic_field_solution",
    "solve_physical_linear_mrn",
    "solve_permanent_magnet_linear_mrn",
]

from .field_state import (
    SegmentFieldState,
    BranchFieldState,
    MagneticFieldSolution,
    recover_magnetic_field_solution,
    solve_physical_linear_mrn,
    solve_permanent_magnet_linear_mrn,
)
