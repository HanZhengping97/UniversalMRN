"""Series material-interface reluctance for cell-centered MRN meshes."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite, log, pi
from typing import Mapping, Any

from src.mesh import Branch, BranchOrientation, Mesh

MU0 = 4.0e-7 * pi


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return value


@dataclass(frozen=True, slots=True)
class ReluctanceSegment:
    branch_id: int
    cell_id: int
    material_id: int
    length: float
    reluctance: float
    area: float | None = None
    inner_radius: float | None = None
    outer_radius: float | None = None
    permeability: float | None = None

    def __post_init__(self) -> None:
        if self.branch_id < 0 or self.cell_id < 0:
            raise ValueError("branch_id and cell_id must be non-negative.")
        _positive(self.length, "length")
        _positive(self.reluctance, "reluctance")
        if self.area is not None:
            _positive(self.area, "area")
        if self.permeability is not None:
            _positive(self.permeability, "permeability")
        if self.inner_radius is not None:
            _positive(self.inner_radius, "inner_radius")
        if self.outer_radius is not None:
            _positive(self.outer_radius, "outer_radius")


def distance_weighted_harmonic_permeability(mu_1: float, mu_2: float, length_1: float, length_2: float) -> float:
    mu_1 = _positive(mu_1, "mu_1")
    mu_2 = _positive(mu_2, "mu_2")
    length_1 = _positive(length_1, "length_1")
    length_2 = _positive(length_2, "length_2")
    return (length_1 + length_2) / (length_1 / mu_1 + length_2 / mu_2)


def harmonic_mean_permeability(mu_1: float, mu_2: float) -> float:
    return distance_weighted_harmonic_permeability(mu_1, mu_2, 1.0, 1.0)


def _material_permeability(materials: Mapping[int, Any], material_id: int) -> float:
    if material_id not in materials:
        raise ValueError(f"missing material for material_id {material_id}.")
    material = materials[material_id]
    if isinstance(material, (int, float)):
        mu = float(material)
    elif hasattr(material, "permeability"):
        mu = float(material.permeability)
    elif hasattr(material, "mu"):
        mu = float(material.mu)
    elif hasattr(material, "relative_permeability"):
        mu = MU0 * float(material.relative_permeability)
    elif hasattr(material, "mu_r"):
        mu = MU0 * float(material.mu_r)
    elif isinstance(material, Mapping):
        if "permeability" in material:
            mu = float(material["permeability"])
        elif "mu" in material:
            mu = float(material["mu"])
        elif "relative_permeability" in material:
            mu = MU0 * float(material["relative_permeability"])
        elif "mu_r" in material:
            mu = MU0 * float(material["mu_r"])
        else:
            raise ValueError(f"material {material_id} lacks permeability.")
    else:
        raise ValueError(f"material {material_id} lacks permeability.")
    return _positive(mu, f"permeability for material_id {material_id}")


def _cells_for_branch(mesh: Mesh, branch: Branch) -> tuple[int, int]:
    if branch.id in mesh.branch_id_to_cell_ids:
        return mesh.branch_id_to_cell_ids[branch.id]
    try:
        return (mesh.node_id_to_cell_id[branch.start_node_id], mesh.node_id_to_cell_id[branch.end_node_id])
    except KeyError as exc:
        raise ValueError("branch does not connect two cell-centered control volumes.") from exc


def calculate_branch_reluctance_segments(mesh: Mesh, branch: Branch, materials: Mapping[int, Any], *, full_annulus: bool = True, angular_span: float | None = None) -> tuple[ReluctanceSegment, ReluctanceSegment]:
    span = 2.0 * pi if full_annulus and angular_span is None else _positive(angular_span if angular_span is not None else 2.0 * pi, "angular_span")
    c0_id, c1_id = _cells_for_branch(mesh, branch)
    c0, c1 = mesh.get_cell(c0_id), mesh.get_cell(c1_id)
    mu0 = _material_permeability(materials, c0.material_id)
    mu1 = _material_permeability(materials, c1.material_id)

    if branch.orientation is BranchOrientation.AXIAL:
        if not isclose(c0.dx, c1.dx) or not isclose(c0.center_x, c1.center_x):
            raise ValueError("axial branch cells must share the same radial interval.")
        r_inner = c0.center_x - 0.5 * c0.dx
        r_outer = c0.center_x + 0.5 * c0.dx
        _positive(r_outer, "outer radius")
        if r_inner < 0.0:
            raise ValueError("inner radius must be non-negative.")
        area = 0.5 * (r_outer * r_outer - r_inner * r_inner) * span
        z_upper = c0.center_y + 0.5 * c0.dy if c0.center_y < c1.center_y else c1.center_y + 0.5 * c1.dy
        z_lower = c1.center_y - 0.5 * c1.dy if c0.center_y < c1.center_y else c0.center_y - 0.5 * c0.dy
        if not isclose(z_upper, z_lower):
            raise ValueError("axial branch cells must share a common axial interface.")
        z_interface = 0.5 * (z_upper + z_lower)
        lengths = (abs(z_interface - c0.center_y), abs(c1.center_y - z_interface))
        reluctances = (lengths[0] / (mu0 * area), lengths[1] / (mu1 * area))
        return (
            ReluctanceSegment(branch.id, c0_id, c0.material_id, lengths[0], reluctances[0], area=area, inner_radius=r_inner, outer_radius=r_outer, permeability=mu0),
            ReluctanceSegment(branch.id, c1_id, c1.material_id, lengths[1], reluctances[1], area=area, inner_radius=r_inner, outer_radius=r_outer, permeability=mu1),
        )

    if branch.orientation is BranchOrientation.RADIAL:
        if not isclose(c0.dy, c1.dy) or not isclose(c0.center_y, c1.center_y):
            raise ValueError("radial branch cells must share the same axial interval.")
        inner, outer, mu_inner, mu_outer = (c0, c1, mu0, mu1) if c0.center_x < c1.center_x else (c1, c0, mu1, mu0)
        inner_id, outer_id = (inner.id, outer.id)
        r_inner_edge = inner.center_x + 0.5 * inner.dx
        r_outer_edge = outer.center_x - 0.5 * outer.dx
        if not isclose(r_inner_edge, r_outer_edge):
            raise ValueError("radial branch cells must share a common radial interface.")
        r_interface = 0.5 * (r_inner_edge + r_outer_edge)
        r1 = _positive(inner.center_x, "inner cell center radius")
        re = _positive(r_interface, "interface radius")
        r2 = _positive(outer.center_x, "outer cell center radius")
        height = _positive(inner.dy, "axial height")
        r_inner = log(re / r1) / (mu_inner * span * height)
        r_outer = log(r2 / re) / (mu_outer * span * height)
        first = ReluctanceSegment(branch.id, inner_id, inner.material_id, re - r1, r_inner, inner_radius=r1, outer_radius=re, permeability=mu_inner)
        second = ReluctanceSegment(branch.id, outer_id, outer.material_id, r2 - re, r_outer, inner_radius=re, outer_radius=r2, permeability=mu_outer)
        return (first, second) if c0.center_x < c1.center_x else (second, first)

    raise ValueError("unsupported branch orientation.")


def calculate_cell_centered_branch_reluctance(mesh: Mesh, branch: Branch, materials: Mapping[int, Any], *, angular_span: float = 2.0 * pi) -> float:
    segments = calculate_branch_reluctance_segments(mesh, branch, materials, full_annulus=False, angular_span=angular_span)
    return sum(segment.reluctance for segment in segments)


def calculate_cell_centered_branch_permeance(mesh: Mesh, branch: Branch, materials: Mapping[int, Any], *, angular_span: float = 2.0 * pi) -> float:
    return 1.0 / calculate_cell_centered_branch_reluctance(mesh, branch, materials, angular_span=angular_span)


def calculate_cell_centered_mesh_branch_permeances(mesh: Mesh, materials: Mapping[int, Any], *, angular_span: float = 2.0 * pi) -> dict[int, float]:
    return {branch_id: calculate_cell_centered_branch_permeance(mesh, mesh.branches[branch_id], materials, angular_span=angular_span) for branch_id in sorted(mesh.branches)}
