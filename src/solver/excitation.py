"""Excitation containers for magnetic scalar-potential network solves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from src.mesh import Mesh
from src.topology import build_topology_index
from .validation import validate_conserved_nodal_flux, validate_finite_vector


@dataclass(frozen=True, slots=True)
class MagneticExcitation:
    """Branch MMF sources and nodal flux injections in deterministic matrix order."""

    branch_mmf: np.ndarray
    nodal_flux_injection: np.ndarray

    def __post_init__(self) -> None:
        branch_mmf = validate_finite_vector("branch_mmf", self.branch_mmf).copy()
        nodal_flux = validate_finite_vector("nodal_flux_injection", self.nodal_flux_injection).copy()
        validate_conserved_nodal_flux(nodal_flux, 1e-12)
        branch_mmf.setflags(write=False)
        nodal_flux.setflags(write=False)
        object.__setattr__(self, "branch_mmf", branch_mmf)
        object.__setattr__(self, "nodal_flux_injection", nodal_flux)

    @classmethod
    def zeros(cls, number_of_nodes: int, number_of_branches: int) -> "MagneticExcitation":
        """Create a zero-source excitation for the given topology sizes."""

        if number_of_nodes < 0 or number_of_branches < 0:
            raise ValueError("number_of_nodes and number_of_branches must be non-negative.")
        return cls(np.zeros(number_of_branches, dtype=float), np.zeros(number_of_nodes, dtype=float))


def build_magnetic_excitation(
    mesh: Mesh,
    *,
    branch_mmf_by_id: Mapping[int, float] | None = None,
    nodal_flux_by_id: Mapping[int, float] | None = None,
    conservation_tolerance: float = 1e-12,
) -> MagneticExcitation:
    """Build an excitation from sparse branch/node ID mappings."""

    index = build_topology_index(mesh)
    branch_mmf = np.zeros(mesh.number_of_branches, dtype=float)
    nodal_flux = np.zeros(mesh.number_of_nodes, dtype=float)
    for branch_id, value in (branch_mmf_by_id or {}).items():
        if branch_id not in index.branch_id_to_column:
            raise ValueError(f"unknown branch ID {branch_id} in branch_mmf_by_id.")
        branch_mmf[index.branch_id_to_column[branch_id]] = float(value)
    for node_id, value in (nodal_flux_by_id or {}).items():
        if node_id not in index.node_id_to_row:
            raise ValueError(f"unknown node ID {node_id} in nodal_flux_by_id.")
        nodal_flux[index.node_id_to_row[node_id]] = float(value)
    validate_finite_vector("branch_mmf", branch_mmf)
    validate_finite_vector("nodal_flux_injection", nodal_flux)
    validate_conserved_nodal_flux(nodal_flux, conservation_tolerance)
    return MagneticExcitation(branch_mmf, nodal_flux)
