from math import isclose, isfinite

from src.material import LinearMagneticMaterial, LinearPermanentMagnetMaterial, MagnetizationAxis
from src.mesh import BranchOrientation, generate_cell_centered_axisymmetric_mesh
from src.mrn import build_permanent_magnet_assignments, build_physical_branch_model
from src.solver import solve_permanent_magnet_linear_mrn


def fixture(assignments):
    mats = {0: LinearMagneticMaterial(0, "air", 1.0), 1: LinearPermanentMagnetMaterial(1, "pm", 1.05, 0.45)}
    mesh = generate_cell_centered_axisymmetric_mesh([1.0, 2.0, 3.0], [0.0, 1.0, 2.0], material_ids=[[1, 1], [0, 0]])
    a = build_permanent_magnet_assignments(mesh, assignments, materials=mats)
    return mesh, mats, a


def test_pm_only_solve_nonzero_residual_finite_and_axial_zero_sources():
    mesh, mats, a = fixture({0: MagnetizationAxis.RADIAL_POSITIVE, 1: MagnetizationAxis.RADIAL_POSITIVE})
    sol = solve_permanent_magnet_linear_mrn(mesh, mats, a)
    assert sol.maximum_absolute_flux > 0.0
    assert sol.linear_solution.max_abs_residual < 1e-9
    assert all(isfinite(v) for b in sol.branches.values() for v in (b.flux, b.potential_drop, b.max_abs_B, b.max_abs_H))
    sources = sol.excitation_diagnostics["branch_sources"]
    assert all(abs(sources[bid].total_mmf) < 1e-12 for bid,b in mesh.branches.items() if b.orientation is BranchOrientation.AXIAL)


def test_reversing_magnetizations_reverses_fluxes():
    mesh, mats, a = fixture({0: MagnetizationAxis.RADIAL_POSITIVE, 1: MagnetizationAxis.RADIAL_POSITIVE})
    _, _, ar = fixture({0: MagnetizationAxis.RADIAL_NEGATIVE, 1: MagnetizationAxis.RADIAL_NEGATIVE})
    s1 = solve_permanent_magnet_linear_mrn(mesh, mats, a)
    s2 = solve_permanent_magnet_linear_mrn(mesh, mats, ar)
    for bid in mesh.branches:
        assert isclose(s1.branches[bid].flux, -s2.branches[bid].flux, rel_tol=1e-9, abs_tol=1e-18)


def test_reference_node_and_potential_invariance_and_constitutive_relation():
    mesh, mats, a = fixture({0: MagnetizationAxis.RADIAL_POSITIVE, 1: MagnetizationAxis.RADIAL_POSITIVE})
    s0 = solve_permanent_magnet_linear_mrn(mesh, mats, a, reference_node_id=0, reference_potential=0.0)
    s1 = solve_permanent_magnet_linear_mrn(mesh, mats, a, reference_node_id=1, reference_potential=7.0)
    phys = build_physical_branch_model(mesh, mats)
    for bid in mesh.branches:
        assert isclose(s0.branches[bid].flux, s1.branches[bid].flux, rel_tol=1e-9, abs_tol=1e-18)
        b = s0.branches[bid]
        assert isclose(b.flux, phys[bid].permeance * (b.source_mmf - b.potential_drop), rel_tol=1e-9, abs_tol=1e-18)
