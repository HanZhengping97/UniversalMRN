"""Linear magnetic scalar-potential solver for reluctance networks."""

from __future__ import annotations

import warnings
from math import isfinite
from typing import Mapping

import numpy as np
from scipy.sparse.linalg import MatrixRankWarning, spsolve

from src.mesh import Mesh
from src.mrn import apply_reference_node, build_branch_permeance_matrix, build_nodal_permeance_matrix
from src.topology import build_incidence_matrix, build_topology_index, validate_connected_mesh
from .excitation import MagneticExcitation
from .solution import LinearMagneticSolution
from .validation import validate_excitation_dimensions, validate_reference_node, validate_solution_residual


def solve_linear_mrn(
    mesh: Mesh,
    branch_permeances: Mapping[int, float],
    excitation: MagneticExcitation | None = None,
    *,
    reference_node_id: int | None = None,
    reference_potential: float = 0.0,
    residual_tolerance: float = 1e-9,
) -> LinearMagneticSolution:
    """Solve ``A Gb A.T psi = A Gb F_b - q_n`` for nodal magnetic potential."""

    if not isfinite(reference_potential):
        raise ValueError(f"reference_potential must be finite; got {reference_potential}.")
    validate_connected_mesh(mesh)
    index = build_topology_index(mesh)
    if not index.row_to_node_id:
        raise ValueError("mesh must contain at least one node.")
    reference_node_id = index.row_to_node_id[0] if reference_node_id is None else reference_node_id
    reference_row = validate_reference_node(reference_node_id, index)

    incidence = build_incidence_matrix(mesh).tocsr()
    gb = build_branch_permeance_matrix(mesh, branch_permeances, index).tocsr()
    gn = build_nodal_permeance_matrix(incidence, gb).tocsr()

    if excitation is None:
        excitation = MagneticExcitation.zeros(mesh.number_of_nodes, mesh.number_of_branches)
    validate_excitation_dimensions(
        excitation.branch_mmf,
        excitation.nodal_flux_injection,
        number_of_nodes=mesh.number_of_nodes,
        number_of_branches=mesh.number_of_branches,
    )

    rhs = incidence @ (gb @ excitation.branch_mmf) - excitation.nodal_flux_injection
    keep = np.ones(mesh.number_of_nodes, dtype=bool)
    keep[reference_row] = False
    reduced_matrix = apply_reference_node(gn, reference_row)
    reference_column = gn[:, reference_row]
    if hasattr(reference_column, "toarray"):
        reference_coupling = np.asarray(reference_column.toarray()).ravel()[keep]
    else:
        reference_coupling = np.asarray(reference_column, dtype=float).ravel()[keep]
    rhs_reduced = np.asarray(rhs, dtype=float)[keep] - reference_coupling * float(reference_potential)

    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=MatrixRankWarning)
        try:
            reduced_solution = spsolve(reduced_matrix, rhs_reduced)
        except MatrixRankWarning as exc:
            raise ValueError("reduced nodal permeance matrix is singular; check mesh connectivity and permeances.") from exc
    reduced_solution = np.asarray(reduced_solution, dtype=float)
    if not np.all(np.isfinite(reduced_solution)):
        raise ValueError("linear MRN solve produced non-finite nodal potentials.")

    node_potential = np.empty(mesh.number_of_nodes, dtype=float)
    node_potential[reference_row] = float(reference_potential)
    node_potential[keep] = reduced_solution

    branch_potential_drop = incidence.transpose() @ node_potential
    branch_flux = gb @ (excitation.branch_mmf - branch_potential_drop)
    nodal_flux_balance = incidence @ branch_flux
    residual = nodal_flux_balance - excitation.nodal_flux_injection
    validate_solution_residual(nodal_flux_balance, excitation.nodal_flux_injection, residual, residual_tolerance)

    return LinearMagneticSolution(
        node_potential=node_potential,
        branch_mmf=excitation.branch_mmf,
        branch_potential_drop=branch_potential_drop,
        branch_flux=branch_flux,
        nodal_flux_residual=residual,
        reference_node_id=reference_node_id,
        reference_row=reference_row,
        reference_potential=float(reference_potential),
    )
