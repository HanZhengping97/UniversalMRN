"""Tests for structured r-z branch topology generation and validation."""

import pytest

from src.mesh import Branch, BranchOrientation, Mesh, MeshGenerator, Node, validate_branch_topology


def test_branch_count_for_1_by_1_mesh() -> None:
    mesh = MeshGenerator.generate_rectangular_mesh(nx=1, ny=1, width=2.0, height=3.0)

    assert mesh.number_of_nodes == 4
    assert mesh.number_of_cells == 1
    assert mesh.number_of_radial_branches == 2
    assert mesh.number_of_axial_branches == 2
    assert mesh.number_of_branches == 4


def test_branch_count_for_3_by_2_mesh() -> None:
    mesh = MeshGenerator.generate_rectangular_mesh(nx=3, ny=2, width=3.0, height=2.0)

    assert mesh.number_of_nodes == 12
    assert mesh.number_of_cells == 6
    assert mesh.number_of_radial_branches == 9
    assert mesh.number_of_axial_branches == 8
    assert mesh.number_of_branches == 17


def test_radial_branch_orientation_and_length() -> None:
    mesh = MeshGenerator.generate_rectangular_mesh(nx=1, ny=1, width=2.0, height=3.0)
    branch = mesh.get_branch(0)
    start = mesh.get_node(branch.start_node_id)
    end = mesh.get_node(branch.end_node_id)

    assert branch.orientation is BranchOrientation.RADIAL
    assert start.r < end.r
    assert start.z == end.z
    assert branch.length == pytest.approx(2.0)
    assert branch.center_r == pytest.approx(1.0)
    assert branch.center_z == pytest.approx(0.0)


def test_axial_branch_orientation_and_length() -> None:
    mesh = MeshGenerator.generate_rectangular_mesh(nx=1, ny=1, width=2.0, height=3.0)
    branch = mesh.get_branch(2)
    start = mesh.get_node(branch.start_node_id)
    end = mesh.get_node(branch.end_node_id)

    assert branch.orientation is BranchOrientation.AXIAL
    assert start.z < end.z
    assert start.r == end.r
    assert branch.length == pytest.approx(3.0)
    assert branch.center_r == pytest.approx(0.0)
    assert branch.center_z == pytest.approx(1.5)


def test_no_duplicate_node_pairs() -> None:
    mesh = MeshGenerator.generate_rectangular_mesh(nx=3, ny=2, width=3.0, height=2.0)
    pairs = [frozenset((branch.start_node_id, branch.end_node_id)) for branch in mesh.branches.values()]

    assert len(pairs) == len(set(pairs))


def test_duplicate_branch_id_rejection() -> None:
    mesh = Mesh()
    mesh.add_node(Node(id=0, x=0.0, y=0.0))
    mesh.add_node(Node(id=1, x=1.0, y=0.0))
    branch = Branch(
        id=0,
        start_node_id=0,
        end_node_id=1,
        orientation=BranchOrientation.RADIAL,
        length=1.0,
        center_r=0.5,
        center_z=0.0,
    )
    mesh.add_branch(branch)

    with pytest.raises(ValueError, match="Branch with id 0"):
        mesh.add_branch(branch)


def test_topology_validation_success() -> None:
    mesh = MeshGenerator.generate_rectangular_mesh(nx=3, ny=2, width=3.0, height=2.0)

    validate_branch_topology(mesh)


def test_topology_validation_detects_invalid_orientation() -> None:
    mesh = MeshGenerator.generate_rectangular_mesh(nx=1, ny=1, width=1.0, height=1.0)
    mesh.branches[0] = Branch(
        id=0,
        start_node_id=0,
        end_node_id=2,
        orientation=BranchOrientation.RADIAL,
        length=1.0,
        center_r=0.0,
        center_z=0.5,
    )

    with pytest.raises(ValueError, match="radial branch 0"):
        validate_branch_topology(mesh)
