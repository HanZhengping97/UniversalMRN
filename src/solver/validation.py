"""Validation helpers for the linear magnetic scalar-potential solver."""

from __future__ import annotations

from math import isfinite

import numpy as np

from src.topology import TopologyIndex


def validate_finite_vector(name: str, vector: np.ndarray) -> np.ndarray:
    """Return ``vector`` as a one-dimensional float array with finite entries."""

    array = np.asarray(vector, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")
    return array


def validate_reference_node(reference_node_id: int, index: TopologyIndex) -> int:
    """Return the row for a valid reference node identifier."""

    if reference_node_id not in index.node_id_to_row:
        raise ValueError(f"unknown reference node ID {reference_node_id}.")
    return index.node_id_to_row[reference_node_id]


def validate_excitation_dimensions(branch_mmf: np.ndarray, nodal_flux_injection: np.ndarray, *, number_of_nodes: int, number_of_branches: int) -> None:
    """Validate excitation vector dimensions against mesh topology counts."""

    if branch_mmf.shape != (number_of_branches,):
        raise ValueError(f"branch_mmf shape {branch_mmf.shape} is incompatible with {number_of_branches} branches.")
    if nodal_flux_injection.shape != (number_of_nodes,):
        raise ValueError(
            f"nodal_flux_injection shape {nodal_flux_injection.shape} is incompatible with {number_of_nodes} nodes."
        )


def validate_conserved_nodal_flux(nodal_flux_injection: np.ndarray, tolerance: float) -> None:
    """Require zero net flux injection for a connected closed magnetic network."""

    if not isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(f"conservation_tolerance must be finite and non-negative; got {tolerance}.")
    total = float(np.sum(nodal_flux_injection))
    if abs(total) > tolerance:
        raise ValueError(f"nodal flux injection is not globally conserved: sum={total}, tolerance={tolerance}.")


def validate_solution_residual(nodal_flux_balance: np.ndarray, nodal_flux_injection: np.ndarray, residual: np.ndarray, tolerance: float) -> None:
    """Validate the scaled nodal flux residual on every node, including reference."""

    if not isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(f"residual_tolerance must be finite and non-negative; got {tolerance}.")
    max_abs_residual = float(np.max(np.abs(residual))) if residual.size else 0.0
    scale = max(
        1.0,
        float(np.max(np.abs(nodal_flux_balance))) if nodal_flux_balance.size else 0.0,
        float(np.max(np.abs(nodal_flux_injection))) if nodal_flux_injection.size else 0.0,
    )
    allowed = tolerance * scale
    if max_abs_residual > allowed:
        raise ValueError(
            f"linear MRN residual {max_abs_residual} exceeds scaled tolerance {allowed} "
            f"(residual_tolerance={tolerance}, scale={scale})."
        )
