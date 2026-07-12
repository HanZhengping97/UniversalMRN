"""Result model for linear magnetic scalar-potential solves."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .validation import validate_finite_vector


@dataclass(frozen=True, slots=True)
class LinearMagneticSolution:
    """Immutable solution vectors for a linear MRN solve.

    Sign convention: ``branch_potential_drop = A.T @ psi`` and
    ``branch_flux = Gb @ (branch_mmf - branch_potential_drop)``. Positive
    branch flux follows the stored branch direction.
    """

    node_potential: np.ndarray
    branch_mmf: np.ndarray
    branch_potential_drop: np.ndarray
    branch_flux: np.ndarray
    nodal_flux_residual: np.ndarray
    reference_node_id: int
    reference_row: int
    reference_potential: float

    def __post_init__(self) -> None:
        for name in ("node_potential", "branch_mmf", "branch_potential_drop", "branch_flux", "nodal_flux_residual"):
            array = validate_finite_vector(name, getattr(self, name)).copy()
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        if self.branch_mmf.shape != self.branch_potential_drop.shape or self.branch_mmf.shape != self.branch_flux.shape:
            raise ValueError("branch_mmf, branch_potential_drop, and branch_flux must have matching shapes.")
        if self.node_potential.shape != self.nodal_flux_residual.shape:
            raise ValueError("node_potential and nodal_flux_residual must have matching shapes.")

    @property
    def max_abs_flux(self) -> float:
        return float(np.max(np.abs(self.branch_flux))) if self.branch_flux.size else 0.0

    @property
    def max_abs_residual(self) -> float:
        return float(np.max(np.abs(self.nodal_flux_residual))) if self.nodal_flux_residual.size else 0.0

    @property
    def number_of_nodes(self) -> int:
        return int(self.node_potential.size)

    @property
    def number_of_branches(self) -> int:
        return int(self.branch_flux.size)
