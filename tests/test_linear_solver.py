"""Tests for the linear magnetic scalar-potential solver."""

from __future__ import annotations

import numpy as np
import pytest

from src.mesh import Branch, BranchOrientation, Mesh, MeshGenerator, Node
from src.solver import MagneticExcitation, build_magnetic_excitation, solve_linear_mrn
from src.topology import build_topology_index


def _mesh() -> Mesh:
    return MeshGenerator.generate_rectangular_mesh(nx=1, ny=1, width=1.0, height=1.0)


def _permeances(mesh: Mesh) -> dict[int, float]:
    return {branch_id: 1.0 for branch_id in mesh.branches}


def _source_excitation(mesh: Mesh) -> MagneticExcitation:
    return build_magnetic_excitation(mesh, branch_mmf_by_id={0: 2.0})


def test_zero_excitation_solution_is_zero() -> None:
    mesh = _mesh()
    solution = solve_linear_mrn(mesh, _permeances(mesh))
    np.testing.assert_allclose(solution.node_potential, 0.0)
    np.testing.assert_allclose(solution.branch_flux, 0.0)
    np.testing.assert_allclose(solution.nodal_flux_residual, 0.0)


def test_nonzero_branch_mmf_produces_flux() -> None:
    mesh = _mesh()
    solution = solve_linear_mrn(mesh, _permeances(mesh), _source_excitation(mesh))
    assert np.max(np.abs(solution.branch_flux)) > 0.0
    assert solution.max_abs_residual <= 1e-12


def test_reference_node_invariance() -> None:
    mesh = _mesh()
    excitation = _source_excitation(mesh)
    first = solve_linear_mrn(mesh, _permeances(mesh), excitation, reference_node_id=0)
    second = solve_linear_mrn(mesh, _permeances(mesh), excitation, reference_node_id=1)
    np.testing.assert_allclose(first.branch_flux, second.branch_flux, atol=1e-12)
    difference = second.node_potential - first.node_potential
    np.testing.assert_allclose(difference, difference[0], atol=1e-12)


def test_nonzero_reference_potential_shifts_only_potentials() -> None:
    mesh = _mesh()
    excitation = _source_excitation(mesh)
    zero = solve_linear_mrn(mesh, _permeances(mesh), excitation)
    shifted = solve_linear_mrn(mesh, _permeances(mesh), excitation, reference_potential=7.5)
    np.testing.assert_allclose(zero.branch_flux, shifted.branch_flux, atol=1e-12)
    np.testing.assert_allclose(shifted.node_potential - zero.node_potential, 7.5, atol=1e-12)


def test_invalid_branch_id_rejected() -> None:
    with pytest.raises(ValueError, match="unknown branch ID 99"):
        build_magnetic_excitation(_mesh(), branch_mmf_by_id={99: 1.0})


def test_invalid_node_id_rejected() -> None:
    with pytest.raises(ValueError, match="unknown node ID 99"):
        build_magnetic_excitation(_mesh(), nodal_flux_by_id={99: 1.0})


def test_unbalanced_nodal_flux_rejected() -> None:
    with pytest.raises(ValueError, match="not globally conserved"):
        build_magnetic_excitation(_mesh(), nodal_flux_by_id={0: 1e-6})


def test_balanced_nodal_source_sink_solves() -> None:
    mesh = _mesh()
    excitation = build_magnetic_excitation(mesh, nodal_flux_by_id={0: 1e-6, 3: -1e-6})
    solution = solve_linear_mrn(mesh, _permeances(mesh), excitation)
    assert solution.max_abs_residual <= 1e-12


def test_disconnected_mesh_rejected() -> None:
    mesh = Mesh()
    mesh.add_node(Node(id=0, x=0.0, y=0.0))
    mesh.add_node(Node(id=1, x=1.0, y=0.0))
    mesh.add_node(Node(id=2, x=2.0, y=0.0))
    mesh.add_branch(Branch(id=0, start_node_id=0, end_node_id=1, orientation=BranchOrientation.RADIAL, length=1.0, center_r=0.5, center_z=0.0))
    with pytest.raises(ValueError, match="disconnected"):
        solve_linear_mrn(mesh, {0: 1.0})


def test_incompatible_excitation_dimensions_rejected() -> None:
    mesh = _mesh()
    excitation = MagneticExcitation(branch_mmf=np.zeros(mesh.number_of_branches + 1), nodal_flux_injection=np.zeros(mesh.number_of_nodes))
    with pytest.raises(ValueError, match="branch_mmf shape"):
        solve_linear_mrn(mesh, _permeances(mesh), excitation)


def test_builder_uses_deterministic_topology_order() -> None:
    mesh = _mesh()
    index = build_topology_index(mesh)
    excitation = build_magnetic_excitation(mesh, branch_mmf_by_id={2: 3.0}, nodal_flux_by_id={0: 1e-9, 1: -1e-9})
    assert excitation.branch_mmf[index.branch_id_to_column[2]] == pytest.approx(3.0)
    assert excitation.nodal_flux_injection[index.node_id_to_row[0]] == pytest.approx(1e-9)
