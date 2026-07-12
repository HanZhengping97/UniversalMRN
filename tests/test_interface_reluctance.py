"""Tests for cell-centered material-interface reluctance."""

from __future__ import annotations

from math import log, pi, isfinite

import numpy as np
import pytest

from src.mesh import BranchOrientation, generate_cell_centered_axisymmetric_mesh
from src.mrn import MU0, calculate_branch_reluctance_segments, calculate_cell_centered_branch_permeance, calculate_cell_centered_branch_reluctance, calculate_cell_centered_mesh_branch_permeances, distance_weighted_harmonic_permeability, harmonic_mean_permeability
from src.solver import build_magnetic_excitation, solve_linear_mrn


def test_equal_length_harmonic_mean() -> None:
    assert harmonic_mean_permeability(1.0, 1000.0) == pytest.approx(2 * 1000 / 1001)


def test_unequal_length_weighted_harmonic_mean_matches_series() -> None:
    mu = distance_weighted_harmonic_permeability(2.0, 10.0, 3.0, 7.0)
    assert mu == pytest.approx((3.0 + 7.0) / (3.0 / 2.0 + 7.0 / 10.0))


def test_arithmetic_mean_is_not_used() -> None:
    assert harmonic_mean_permeability(1.0, 1000.0) != pytest.approx(500.5)


def _mesh(material_ids=None):
    return generate_cell_centered_axisymmetric_mesh([1.0, 2.0, 3.0], [0.0, 1.0, 2.0], material_ids=material_ids)


def test_axial_homogeneous_branch() -> None:
    mesh = _mesh([[0, 0], [0, 0]])
    branch = next(b for b in mesh.branches.values() if b.orientation is BranchOrientation.AXIAL)
    materials = {0: MU0 * 5.0}
    r0, r1 = 1.0, 2.0
    area = pi * (r1 * r1 - r0 * r0)
    expected = branch.length / (materials[0] * area)
    assert calculate_cell_centered_branch_reluctance(mesh, branch, materials) == pytest.approx(expected)


def test_axial_material_interface() -> None:
    mesh = _mesh([[0, 0], [1, 1]])
    branch = next(b for b in mesh.branches.values() if b.orientation is BranchOrientation.AXIAL)
    materials = {0: MU0, 1: MU0 * 1000.0}
    segments = calculate_branch_reluctance_segments(mesh, branch, materials)
    total = sum(s.reluctance for s in segments)
    assert calculate_cell_centered_branch_reluctance(mesh, branch, materials) == pytest.approx(total)
    assert calculate_cell_centered_branch_permeance(mesh, branch, materials) == pytest.approx(1.0 / total)


def test_radial_homogeneous_branch_exact_logarithmic() -> None:
    mesh = _mesh([[0, 0], [0, 0]])
    branch = next(b for b in mesh.branches.values() if b.orientation is BranchOrientation.RADIAL)
    mu = MU0 * 7.0
    expected = log(2.5 / 1.5) / (mu * 2.0 * pi * 1.0)
    assert calculate_cell_centered_branch_reluctance(mesh, branch, {0: mu}) == pytest.approx(expected)


def test_radial_material_interface_split_logarithmic() -> None:
    mesh = _mesh([[0, 1], [0, 1]])
    branch = next(b for b in mesh.branches.values() if b.orientation is BranchOrientation.RADIAL)
    materials = {0: MU0, 1: MU0 * 1000.0}
    expected = log(2.0 / 1.5) / (materials[0] * 2 * pi) + log(2.5 / 2.0) / (materials[1] * 2 * pi)
    assert calculate_cell_centered_branch_reluctance(mesh, branch, materials) == pytest.approx(expected)


def test_cell_centered_topology_node_count() -> None:
    assert _mesh().number_of_nodes == 4


def test_cell_centered_topology_branch_count() -> None:
    mesh = generate_cell_centered_axisymmetric_mesh([1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 2.0])
    nr, nz = 3, 2
    assert mesh.number_of_radial_branches == (nr - 1) * nz
    assert mesh.number_of_axial_branches == nr * (nz - 1)


def test_deterministic_numbering() -> None:
    first = _mesh([[0, 1], [2, 3]])
    second = _mesh([[0, 1], [2, 3]])
    assert first.nodes == second.nodes
    assert first.cells == second.cells
    assert first.branches == second.branches
    assert first.branch_id_to_cell_ids == second.branch_id_to_cell_ids


def test_interface_branch_has_exactly_two_segments() -> None:
    mesh = _mesh([[0, 1], [0, 1]])
    segments = calculate_branch_reluctance_segments(mesh, mesh.get_branch(0), {0: MU0, 1: MU0})
    assert len(segments) == 2


def test_missing_material_rejected() -> None:
    mesh = _mesh([[0, 1], [0, 1]])
    with pytest.raises(ValueError, match="missing material"):
        calculate_branch_reluctance_segments(mesh, mesh.get_branch(0), {0: MU0})


def test_nonmonotonic_radial_edges_rejected() -> None:
    with pytest.raises(ValueError, match="radial_edges must be strictly increasing"):
        generate_cell_centered_axisymmetric_mesh([1.0, 1.0, 2.0], [0.0, 1.0])


def test_nonmonotonic_axial_edges_rejected() -> None:
    with pytest.raises(ValueError, match="axial_edges must be strictly increasing"):
        generate_cell_centered_axisymmetric_mesh([1.0, 2.0], [0.0, -1.0])


def test_full_solver_integration() -> None:
    mesh = _mesh([[0, 1], [0, 1]])
    permeances = calculate_cell_centered_mesh_branch_permeances(mesh, {0: MU0, 1: MU0 * 1000.0})
    assert all(value > 0 and isfinite(value) for value in permeances.values())
    solution = solve_linear_mrn(mesh, permeances, build_magnetic_excitation(mesh, branch_mmf_by_id={0: 1.0}))
    assert np.all(np.isfinite(solution.branch_flux))
    assert solution.max_abs_residual < 1e-9


def test_large_permeability_contrast() -> None:
    mu = harmonic_mean_permeability(1.0, 100000.0)
    assert isfinite(mu)
    assert mu == pytest.approx(2.0, rel=3e-5)
