"""Geometry-derived permeance for structured axisymmetric r-z meshes.

Each graph branch is treated as a magnetic flux tube in an axisymmetric domain.
The half-cells touching a branch form parallel flux tubes, so their permeances
are summed. This supports material interfaces without assigning one material
directly to a branch.
"""

from __future__ import annotations

from math import pi
from typing import Mapping

from src.material import LinearMagneticMaterial, validate_material_library
from src.mesh import Branch, BranchOrientation, Cell, Mesh

_DEFAULT_TOLERANCE = 1.0e-12


def _touches_radial_branch(cell: Cell, branch: Branch, tolerance: float) -> bool:
    radial_min = cell.center_x - 0.5 * cell.dx
    radial_max = cell.center_x + 0.5 * cell.dx
    axial_min = cell.center_y - 0.5 * cell.dy
    axial_max = cell.center_y + 0.5 * cell.dy
    return (
        radial_min - tolerance <= branch.center_r <= radial_max + tolerance
        and (abs(branch.center_z - axial_min) <= tolerance or abs(branch.center_z - axial_max) <= tolerance)
    )


def _touches_axial_branch(cell: Cell, branch: Branch, tolerance: float) -> bool:
    radial_min = cell.center_x - 0.5 * cell.dx
    radial_max = cell.center_x + 0.5 * cell.dx
    axial_min = cell.center_y - 0.5 * cell.dy
    axial_max = cell.center_y + 0.5 * cell.dy
    return (
        (abs(branch.center_r - radial_min) <= tolerance or abs(branch.center_r - radial_max) <= tolerance)
        and axial_min - tolerance <= branch.center_z <= axial_max + tolerance
    )


def find_adjacent_cells(
    mesh: Mesh, branch: Branch, *, tolerance: float = _DEFAULT_TOLERANCE
) -> tuple[Cell, ...]:
    """Return the one or two cells whose boundary contains ``branch``."""

    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative.")
    predicate = (
        _touches_radial_branch
        if branch.orientation is BranchOrientation.RADIAL
        else _touches_axial_branch
    )
    adjacent = tuple(cell for cell in mesh.cells.values() if predicate(cell, branch, tolerance))
    if not adjacent:
        raise ValueError(f"branch {branch.id} has no adjacent cells.")
    if len(adjacent) > 2:
        raise ValueError(
            f"branch {branch.id} has {len(adjacent)} adjacent cells; expected at most two."
        )
    return tuple(sorted(adjacent, key=lambda cell: cell.id))


def _radial_half_cell_area(branch: Branch, cell: Cell) -> float:
    """Cylindrical area normal to a radial branch for one touching half-cell."""

    return 2.0 * pi * branch.center_r * (0.5 * cell.dy)


def _axial_half_cell_area(branch: Branch, cell: Cell) -> float:
    """Exact annular area normal to an axial branch for one half-cell."""

    half_width = 0.5 * cell.dx
    if cell.center_x < branch.center_r:
        inner_radius = branch.center_r - half_width
        outer_radius = branch.center_r
    else:
        inner_radius = branch.center_r
        outer_radius = branch.center_r + half_width
    if inner_radius < 0.0:
        raise ValueError(f"cell {cell.id} produces a negative inner radius.")
    return pi * (outer_radius**2 - inner_radius**2)


def calculate_branch_permeance(
    mesh: Mesh,
    branch: Branch,
    materials: Mapping[int, LinearMagneticMaterial],
    *,
    tolerance: float = _DEFAULT_TOLERANCE,
) -> float:
    """Calculate one branch permeance in henry from geometry and material data."""

    validate_material_library(materials, {cell.material_id for cell in mesh.cells.values()})
    permeance = 0.0
    for cell in find_adjacent_cells(mesh, branch, tolerance=tolerance):
        area = (
            _radial_half_cell_area(branch, cell)
            if branch.orientation is BranchOrientation.RADIAL
            else _axial_half_cell_area(branch, cell)
        )
        permeance += materials[cell.material_id].permeability * area / branch.length
    if permeance <= 0.0:
        raise ValueError(f"branch {branch.id} has non-positive permeance.")
    return permeance


def calculate_mesh_branch_permeances(
    mesh: Mesh,
    materials: Mapping[int, LinearMagneticMaterial],
    *,
    tolerance: float = _DEFAULT_TOLERANCE,
) -> dict[int, float]:
    """Calculate physical permeances for all branches in deterministic ID order."""

    validate_material_library(materials, {cell.material_id for cell in mesh.cells.values()})
    return {
        branch_id: calculate_branch_permeance(mesh, branch, materials, tolerance=tolerance)
        for branch_id, branch in sorted(mesh.branches.items())
    }
