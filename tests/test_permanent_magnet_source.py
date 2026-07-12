from math import isclose
import pytest

from src.material import LinearMagneticMaterial, LinearPermanentMagnetMaterial, MagnetizationAxis
from src.mesh import BranchOrientation, generate_cell_centered_axisymmetric_mesh
from src.mrn import build_permanent_magnet_assignments, build_permanent_magnet_branch_sources, build_permanent_magnet_excitation, build_physical_branch_model


def fixture(material_ids=None):
    mats = {0: LinearMagneticMaterial(0, "air", 1.0), 1: LinearPermanentMagnetMaterial(1, "pm", 1.05, 0.45)}
    mesh = generate_cell_centered_axisymmetric_mesh([1.0, 2.0, 3.0], [0.0, 1.0, 2.0], material_ids=material_ids or [[1, 1], [0, 0]])
    phys = build_physical_branch_model(mesh, mats)
    return mesh, mats, phys


def test_radial_signs_axial_zero_and_lengths():
    mesh, mats, phys = fixture()
    radial = next(bid for bid,b in mesh.branches.items() if b.orientation is BranchOrientation.RADIAL)
    axial = next(bid for bid,b in mesh.branches.items() if b.orientation is BranchOrientation.AXIAL and 0 in mesh.branch_id_to_cell_ids[bid])
    pos = build_permanent_magnet_assignments(mesh, {0: MagnetizationAxis.RADIAL_POSITIVE, 1: MagnetizationAxis.RADIAL_POSITIVE}, materials=mats)
    src = build_permanent_magnet_branch_sources(mesh, phys, mats, pos)
    assert src[radial].total_mmf > 0
    assert isclose(src[radial].segment_sources[0].magnetized_length, 0.5)
    assert isclose(src[axial].total_mmf, 0.0, abs_tol=1e-12)
    neg = build_permanent_magnet_assignments(mesh, {0: MagnetizationAxis.RADIAL_NEGATIVE, 1: MagnetizationAxis.RADIAL_NEGATIVE}, materials=mats)
    assert build_permanent_magnet_branch_sources(mesh, phys, mats, neg)[radial].total_mmf < 0


def test_axial_positive_axial_branch_and_axial_length():
    mesh, mats, phys = fixture([[1,0],[1,0]])
    axial = next(bid for bid,b in mesh.branches.items() if b.orientation is BranchOrientation.AXIAL and set(mesh.branch_id_to_cell_ids[bid]) == {0,2})
    assignments = build_permanent_magnet_assignments(mesh, {0: MagnetizationAxis.AXIAL_POSITIVE, 2: MagnetizationAxis.AXIAL_POSITIVE}, materials=mats)
    src = build_permanent_magnet_branch_sources(mesh, phys, mats, assignments)[axial]
    assert src.total_mmf > 0
    assert isclose(src.segment_sources[0].magnetized_length, phys[axial].segments[0].length)


def test_two_segments_add_and_opposites_cancel_non_pm_zero():
    mesh, mats, phys = fixture()
    radial = next(bid for bid,b in mesh.branches.items() if b.orientation is BranchOrientation.RADIAL and set(mesh.branch_id_to_cell_ids[bid]) == {0,1})
    assignments = build_permanent_magnet_assignments(mesh, {0: MagnetizationAxis.RADIAL_POSITIVE, 1: MagnetizationAxis.RADIAL_NEGATIVE}, materials=mats)
    srcs = build_permanent_magnet_branch_sources(mesh, phys, mats, assignments)
    assert isclose(srcs[radial].total_mmf, 0.0, abs_tol=1e-9)
    non_pm = next(bid for bid,b in mesh.branches.items() if set(mesh.branch_id_to_cell_ids[bid]) == {2,3})
    assert srcs[non_pm].total_mmf == 0.0


def test_assignment_validation_and_strict():
    mesh, mats, _ = fixture()
    with pytest.raises(ValueError):
        build_permanent_magnet_assignments(mesh, {2: MagnetizationAxis.RADIAL_POSITIVE}, materials=mats)
    with pytest.raises(ValueError):
        build_permanent_magnet_assignments(mesh, {99: MagnetizationAxis.RADIAL_POSITIVE}, materials=mats)
    with pytest.raises(ValueError):
        build_permanent_magnet_assignments(mesh, {0: MagnetizationAxis.RADIAL_POSITIVE}, materials=mats, strict=True)


def test_excitation_deterministic_and_additional_superposition():
    mesh, mats, phys = fixture()
    assignments = build_permanent_magnet_assignments(mesh, {0: MagnetizationAxis.RADIAL_POSITIVE, 1: MagnetizationAxis.RADIAL_POSITIVE}, materials=mats)
    e0 = build_permanent_magnet_excitation(mesh, phys, mats, assignments)
    e1 = build_permanent_magnet_excitation(mesh, phys, mats, assignments, additional_branch_mmf_by_id={0: 3.0})
    assert tuple(e0.branch_mmf) == tuple(build_permanent_magnet_excitation(mesh, phys, mats, assignments).branch_mmf)
    assert isclose(e1.branch_mmf[0], e0.branch_mmf[0] + 3.0)
    with pytest.raises(ValueError):
        build_permanent_magnet_excitation(mesh, phys, mats, assignments, additional_branch_mmf_by_id={99: 1.0})
