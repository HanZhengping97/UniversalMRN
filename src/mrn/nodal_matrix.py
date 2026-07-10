"""Nodal permeance matrix assembly helpers.

For a connected network the full nodal matrix is singular because magnetic
scalar potential has an arbitrary reference.  Remove one reference node before a
future solve; this module does not solve the magnetic field.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse


def build_nodal_permeance_matrix(
    incidence_matrix: sparse.spmatrix, branch_permeance_matrix: sparse.spmatrix
) -> sparse.csr_matrix:
    """Assemble ``Gn = A @ Gb @ A.T`` and return CSR format."""

    if incidence_matrix.ndim != 2 or branch_permeance_matrix.ndim != 2:
        raise ValueError("incidence and branch permeance matrices must be two-dimensional.")
    node_count, branch_count = incidence_matrix.shape
    if branch_permeance_matrix.shape != (branch_count, branch_count):
        raise ValueError(
            "branch permeance matrix shape "
            f"{branch_permeance_matrix.shape} is incompatible with incidence shape {incidence_matrix.shape}."
        )
    return (incidence_matrix.tocsr() @ branch_permeance_matrix.tocsr() @ incidence_matrix.tocsr().transpose()).tocsr()


def apply_reference_node(nodal_matrix: sparse.spmatrix, reference_row: int) -> sparse.csr_matrix:
    """Remove the reference-node row and column from a square nodal matrix."""

    if nodal_matrix.ndim != 2 or nodal_matrix.shape[0] != nodal_matrix.shape[1]:
        raise ValueError("nodal matrix must be square.")
    size = nodal_matrix.shape[0]
    if reference_row < 0 or reference_row >= size:
        raise ValueError(f"reference_row {reference_row} is outside matrix size {size}.")
    keep = np.ones(size, dtype=bool)
    keep[reference_row] = False
    return nodal_matrix.tocsr()[keep, :][:, keep].tocsr()


def remove_reference_entry(vector, reference_row: int):
    """Remove the entry corresponding to a reference node from a nodal vector."""

    array = np.asarray(vector)
    if array.ndim != 1:
        raise ValueError("vector must be one-dimensional.")
    if reference_row < 0 or reference_row >= array.shape[0]:
        raise ValueError(f"reference_row {reference_row} is outside vector length {array.shape[0]}.")
    return np.delete(array, reference_row)
