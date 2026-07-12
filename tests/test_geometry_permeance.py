from math import pi

import pytest

from src.material import MU_0, LinearMagneticMaterial
from src.mesh import BranchOrientation, Cell, MeshGenerator
from src.mrn import calculate_branch_permeance, calculate_mesh_branch_permeances, find_adjacent_cells


def air_library():
    return {0: LinearMagneticMaterial(0, "Air", 1.0)}


def test_radial_boundary_branch() -> None:
    mesh = MeshGenerator.generate_rectangular_mesh(1, 1, 1.0, 2.0)
    branch = next(b for b in mesh.branches.values() if b.orientation is BranchOrientation.RADIAL and b.center_z == 0.0)
    assert len(find_adjacent_cells(mesh, branch)) == 1
    assert calculate_branch_permeance(mesh, branch, air_library()) == pytest.approx(MU_0 * pi)


def test_axial_boundary_branch() -> None:
    mesh = MeshGenerator.generate_rectangular_mesh(1, 1, 1.0, 2.0)
    branch = next(b for b in mesh.branches.values() if b.orientation is BranchOrientation.AXIAL and b.center_r == 1.0)
    expected = MU_0 * pi * (1.0**2 - 0.5**2) / 2.0
    assert calculate_branch_permeance(mesh, branch, air_library()) == pytest.approx(expected)


def test_material_interface_adds_parallel_contributions() -> None:
    mesh = MeshGenerator.generate_rectangular_mesh(2, 1, 2.0, 1.0)
    first = mesh.cells[0]
    second = mesh.cells[1]
    mesh.cells[0] = Cell(first.id, first.center_x, first.center_y, first.dx, first.dy, 0)
    mesh.cells[1] = Cell(second.id, second.center_x, second.center_y, second.dx, second.dy, 1)
    materials = {
        0: LinearMagneticMaterial(0, "Air", 1.0),
        1: LinearMagneticMaterial(1, "Linear steel", 1000.0),
    }
    branch = next(b for b in mesh.branches.values() if b.orientation is BranchOrientation.AXIAL and b.center_r == 1.0)
    expected = MU_0 * pi * (1.0**2 - 0.5**2) + 1000.0 * MU_0 * pi * (1.5**2 - 1.0**2)
    assert calculate_branch_permeance(mesh, branch, materials) == pytest.approx(expected)


def test_all_branches_receive_positive_permeance() -> None:
    mesh = MeshGenerator.generate_rectangular_mesh(2, 2, 2.0, 1.0)
    values = calculate_mesh_branch_permeances(mesh, air_library())
    assert tuple(values) == tuple(sorted(mesh.branches))
    assert all(value > 0.0 for value in values.values())


def test_missing_material_is_rejected() -> None:
    mesh = MeshGenerator.generate_rectangular_mesh(1, 1, 1.0, 1.0, material_id=7)
    with pytest.raises(ValueError, match="missing material definitions"):
        calculate_mesh_branch_permeances(mesh, air_library())
