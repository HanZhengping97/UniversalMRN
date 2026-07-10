"""Tests for Phase 3 MRN incidence and matrix assembly."""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.mesh import Branch, BranchOrientation, Mesh, MeshGenerator, Node
from src.mrn import apply_reference_node, build_branch_permeance_matrix, build_nodal_permeance_matrix
from src.topology import (
    build_incidence_matrix,
    build_topology_index,
    find_connected_components,
    validate_connected_mesh,
    validate_incidence_matrix,
)


def test_incidence_for_1_by_1_cell_mesh() -> None:
    mesh = MeshGenerator.generate_rectangular_mesh(nx=1, ny=1, width=1.0, height=1.0)
    index = build_topology_index(mesh)
    incidence = build_incidence_matrix(mesh)

    assert mesh.number_of_nodes == 4
    assert mesh.number_of_branches == 4
    assert incidence.shape == (4, 4)
    validate_incidence_matrix(mesh, incidence, index)
    for column in range(incidence.shape[1]):
        values = incidence.tocsc().data[incidence.tocsc().indptr[column] : incidence.tocsc().indptr[column + 1]]
        assert sorted(values.tolist()) == [-1, 1]


def test_incidence_for_3_by_2_cell_mesh() -> None:
    mesh = MeshGenerator.generate_rectangular_mesh(nx=3, ny=2, width=3.0, height=2.0)
    incidence = build_incidence_matrix(mesh)

    assert mesh.number_of_nodes == 12
    assert mesh.number_of_branches == 17
    assert incidence.shape == (12, 17)


def test_incidence_matches_every_branch_endpoint() -> None:
    mesh = MeshGenerator.generate_rectangular_mesh(nx=3, ny=2, width=3.0, height=2.0)
    index = build_topology_index(mesh)
    incidence = build_incidence_matrix(mesh)

    for branch in mesh.branches.values():
        start_row = index.node_id_to_row[branch.start_node_id]
        end_row = index.node_id_to_row[branch.end_node_id]
        column = index.branch_id_to_column[branch.id]
        assert incidence[start_row, column] == -1
        assert incidence[end_row, column] == 1


def test_topology_index_sorts_ids_independent_of_insertion_order() -> None:
    mesh = Mesh()
    for node_id in (10, 2, 7):
        mesh.add_node(Node(id=node_id, x=float(node_id), y=0.0))
    mesh.add_branch(Branch(5, 10, 2, BranchOrientation.RADIAL, 1.0, 0.0, 0.0))
    mesh.add_branch(Branch(1, 2, 7, BranchOrientation.RADIAL, 1.0, 0.0, 0.0))

    index = build_topology_index(mesh)

    assert index.row_to_node_id == (2, 7, 10)
    assert index.column_to_branch_id == (1, 5)
    assert index.node_id_to_row == {2: 0, 7: 1, 10: 2}
    assert index.branch_id_to_column == {1: 0, 5: 1}


def test_branch_permeance_matrix_validation_and_order() -> None:
    mesh = MeshGenerator.generate_rectangular_mesh(nx=1, ny=1, width=1.0, height=1.0)
    index = build_topology_index(mesh)
    values = {branch_id: branch_id + 1.0 for branch_id in mesh.branches}

    gb = build_branch_permeance_matrix(mesh, values, index)

    assert gb.shape == (4, 4)
    assert_allclose(gb.diagonal(), [values[branch_id] for branch_id in index.column_to_branch_id])

    with pytest.raises(ValueError, match="missing"):
        build_branch_permeance_matrix(mesh, {0: 1.0, 1: 1.0, 2: 1.0}, index)
    with pytest.raises(ValueError, match="unknown"):
        build_branch_permeance_matrix(mesh, values | {99: 1.0}, index)
    for invalid in (0.0, -1.0, math.nan, math.inf, -math.inf):
        bad_values = dict(values)
        bad_values[0] = invalid
        with pytest.raises(ValueError, match="positive and finite"):
            build_branch_permeance_matrix(mesh, bad_values, index)


def test_nodal_matrix_symmetry_row_sums_and_reference_reduction() -> None:
    mesh = MeshGenerator.generate_rectangular_mesh(nx=3, ny=2, width=3.0, height=2.0)
    index = build_topology_index(mesh)
    incidence = build_incidence_matrix(mesh)
    gb = build_branch_permeance_matrix(mesh, {branch_id: 1.0 for branch_id in mesh.branches}, index)

    gn = build_nodal_permeance_matrix(incidence, gb)
    reduced = apply_reference_node(gn, reference_row=0)

    assert gn.shape == (mesh.number_of_nodes, mesh.number_of_nodes)
    assert_allclose((gn - gn.transpose()).toarray(), np.zeros(gn.shape))
    assert_allclose(np.asarray(gn.sum(axis=1)).ravel(), np.zeros(mesh.number_of_nodes), atol=1e-12)
    assert reduced.shape == (mesh.number_of_nodes - 1, mesh.number_of_nodes - 1)


def test_connectivity_validation() -> None:
    mesh = MeshGenerator.generate_rectangular_mesh(nx=3, ny=2, width=3.0, height=2.0)

    assert find_connected_components(mesh) == (tuple(range(12)),)
    validate_connected_mesh(mesh)

    disconnected = Mesh()
    disconnected.add_node(Node(id=0, x=0.0, y=0.0))
    disconnected.add_node(Node(id=1, x=1.0, y=0.0))
    disconnected.add_node(Node(id=2, x=10.0, y=0.0))
    disconnected.add_branch(Branch(0, 0, 1, BranchOrientation.RADIAL, 1.0, 0.5, 0.0))

    assert find_connected_components(disconnected) == ((0, 1), (2,))
    with pytest.raises(ValueError, match="disconnected"):
        validate_connected_mesh(disconnected)
