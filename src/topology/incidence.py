"""Incidence-matrix construction for magnetic-network branch topology."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from src.mesh import Mesh


@dataclass(frozen=True, slots=True)
class TopologyIndex:
    """Deterministic mappings between mesh IDs and matrix indices."""

    node_id_to_row: dict[int, int]
    row_to_node_id: tuple[int, ...]
    branch_id_to_column: dict[int, int]
    column_to_branch_id: tuple[int, ...]


def build_topology_index(mesh: Mesh) -> TopologyIndex:
    """Build deterministic node-row and branch-column mappings for ``mesh``."""

    row_to_node_id = tuple(sorted(mesh.nodes))
    column_to_branch_id = tuple(sorted(mesh.branches))
    return TopologyIndex(
        node_id_to_row={node_id: row for row, node_id in enumerate(row_to_node_id)},
        row_to_node_id=row_to_node_id,
        branch_id_to_column={branch_id: column for column, branch_id in enumerate(column_to_branch_id)},
        column_to_branch_id=column_to_branch_id,
    )


def build_incidence_matrix(mesh: Mesh) -> sparse.csr_matrix:
    """Build the sparse node-branch incidence matrix using start=-1, end=+1."""

    index = build_topology_index(mesh)
    rows: list[int] = []
    columns: list[int] = []
    data: list[int] = []
    for branch_id in index.column_to_branch_id:
        branch = mesh.branches[branch_id]
        if branch.start_node_id not in index.node_id_to_row:
            raise ValueError(f"branch {branch.id} references missing start node {branch.start_node_id}.")
        if branch.end_node_id not in index.node_id_to_row:
            raise ValueError(f"branch {branch.id} references missing end node {branch.end_node_id}.")
        column = index.branch_id_to_column[branch.id]
        rows.extend((index.node_id_to_row[branch.start_node_id], index.node_id_to_row[branch.end_node_id]))
        columns.extend((column, column))
        data.extend((-1, 1))
    return sparse.coo_matrix(
        (data, (rows, columns)), shape=(len(index.row_to_node_id), len(index.column_to_branch_id)), dtype=np.int8
    ).tocsr()


def validate_incidence_matrix(mesh: Mesh, incidence_matrix: sparse.spmatrix, index: TopologyIndex) -> None:
    """Validate incidence shape, signs, ID mappings, and branch endpoint rows."""

    expected_nodes = set(mesh.nodes)
    expected_branches = set(mesh.branches)
    if set(index.row_to_node_id) != expected_nodes or set(index.node_id_to_row) != expected_nodes:
        raise ValueError("topology index node IDs do not match mesh nodes.")
    if set(index.column_to_branch_id) != expected_branches or set(index.branch_id_to_column) != expected_branches:
        raise ValueError("topology index branch IDs do not match mesh branches.")
    expected_shape = (mesh.number_of_nodes, mesh.number_of_branches)
    if incidence_matrix.shape != expected_shape:
        raise ValueError(f"incidence matrix shape {incidence_matrix.shape} does not match {expected_shape}.")

    csc = incidence_matrix.tocsc()
    for branch_id in index.column_to_branch_id:
        branch = mesh.branches.get(branch_id)
        if branch is None:
            raise ValueError(f"branch {branch_id} in index does not exist in mesh.")
        if branch.start_node_id not in mesh.nodes or branch.end_node_id not in mesh.nodes:
            raise ValueError(f"branch {branch.id} references a missing node.")
        column = index.branch_id_to_column[branch_id]
        start_ptr, end_ptr = csc.indptr[column], csc.indptr[column + 1]
        rows = csc.indices[start_ptr:end_ptr]
        values = csc.data[start_ptr:end_ptr]
        if len(values) != 2:
            raise ValueError(f"branch column {column} for branch {branch_id} has {len(values)} nonzero values.")
        if set(values.tolist()) != {-1, 1}:
            raise ValueError(f"branch column {column} must contain exactly one -1 and one +1.")
        value_by_row = dict(zip(rows.tolist(), values.tolist(), strict=True))
        start_row = index.node_id_to_row[branch.start_node_id]
        end_row = index.node_id_to_row[branch.end_node_id]
        if value_by_row.get(start_row) != -1:
            raise ValueError(f"branch {branch_id} start node row {start_row} does not contain -1.")
        if value_by_row.get(end_row) != 1:
            raise ValueError(f"branch {branch_id} end node row {end_row} does not contain +1.")
        if any(value not in (-1, 1) for value in values):
            raise ValueError(f"branch column {column} contains an unexpected nonzero value.")
