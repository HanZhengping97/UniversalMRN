"""Permanent-magnet branch-source generation for physical MRN branches.

All mesh branches are returned deterministically. Branches without PM material, or with
only perpendicular magnetization in the r-z projection, have zero total MMF.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite
from typing import Any, Mapping


from src.material.magnetization import MagnetizationAxis, MagnetizationDirection, coerce_magnetization_direction
from src.mesh import Branch, BranchOrientation, Mesh
from src.solver.excitation import MagneticExcitation, build_magnetic_excitation
from src.topology import build_topology_index
from .physical_branch import PhysicalBranch, SegmentGeometryKind


@dataclass(frozen=True, slots=True)
class PermanentMagnetAssignment:
    cell_id: int
    magnetization: MagnetizationDirection

    def __post_init__(self) -> None:
        if self.cell_id < 0:
            raise ValueError("cell_id must be non-negative.")
        object.__setattr__(self, "magnetization", coerce_magnetization_direction(self.magnetization))


@dataclass(frozen=True, slots=True)
class PermanentMagnetSegmentSource:
    branch_id: int
    segment_index: int
    cell_id: int
    material_id: int
    coercive_field_strength: float
    magnetization_projection: float
    magnetized_length: float
    mmf: float

    def __post_init__(self) -> None:
        for name in ("branch_id", "segment_index", "cell_id", "material_id"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative.")
        if not isfinite(float(self.coercive_field_strength)) or self.coercive_field_strength <= 0.0:
            raise ValueError("coercive_field_strength must be positive and finite.")
        if not isfinite(float(self.magnetization_projection)) or not -1.0 <= self.magnetization_projection <= 1.0:
            raise ValueError("magnetization_projection must be finite and between -1 and 1.")
        if not isfinite(float(self.magnetized_length)) or self.magnetized_length <= 0.0:
            raise ValueError("magnetized_length must be positive and finite.")
        if not isfinite(float(self.mmf)):
            raise ValueError("mmf must be finite.")


@dataclass(frozen=True, slots=True)
class PermanentMagnetBranchSource:
    branch_id: int
    segment_sources: tuple[PermanentMagnetSegmentSource, ...]
    total_mmf: float

    def __post_init__(self) -> None:
        if self.branch_id < 0:
            raise ValueError("branch_id must be non-negative.")
        if tuple(s.segment_index for s in self.segment_sources) != tuple(sorted(s.segment_index for s in self.segment_sources)):
            raise ValueError("segment sources must be sorted deterministically.")
        if any(s.branch_id != self.branch_id for s in self.segment_sources):
            raise ValueError("segment branch IDs must match branch source.")
        if not isfinite(float(self.total_mmf)):
            raise ValueError("total_mmf must be finite.")
        if not isclose(sum(s.mmf for s in self.segment_sources), self.total_mmf, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("total_mmf must equal segment MMF sum.")


@dataclass(frozen=True, slots=True)
class BranchSourceComponents:
    permanent_magnet_mmf: float
    winding_mmf: float = 0.0
    external_mmf: float = 0.0

    def __post_init__(self) -> None:
        for name in ("permanent_magnet_mmf", "winding_mmf", "external_mmf"):
            if not isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite.")

    @property
    def total_mmf(self) -> float:
        return self.permanent_magnet_mmf + self.winding_mmf + self.external_mmf


def _is_pm(material: Any) -> bool:
    return bool(getattr(material, "is_permanent_magnet", False))


def branch_direction_vector(branch: Branch) -> tuple[float, float]:
    """Return the positive branch direction in (radial, axial) components."""
    if branch.length <= 0.0:
        raise ValueError("branch length must be positive.")
    if branch.orientation is BranchOrientation.RADIAL:
        return (1.0, 0.0)
    if branch.orientation is BranchOrientation.AXIAL:
        return (0.0, 1.0)
    raise ValueError("unsupported branch orientation.")


def build_permanent_magnet_assignments(mesh: Mesh, assignments_by_cell_id: Mapping[int, MagnetizationDirection | MagnetizationAxis], *, materials: Mapping[int, Any] | None = None, strict: bool = False) -> dict[int, PermanentMagnetAssignment]:
    out: dict[int, PermanentMagnetAssignment] = {}
    for cell_id, direction in assignments_by_cell_id.items():
        if cell_id not in mesh.cells:
            raise ValueError(f"unknown cell ID {cell_id} in permanent magnet assignments.")
        cell = mesh.cells[cell_id]
        if materials is not None and not _is_pm(materials.get(cell.material_id)):
            raise ValueError(f"cell {cell_id} material is not a permanent magnet.")
        out[cell_id] = PermanentMagnetAssignment(cell_id, coerce_magnetization_direction(direction))
    if strict and materials is not None:
        missing = sorted(c.id for c in mesh.cells.values() if _is_pm(materials.get(c.material_id)) and c.id not in out)
        if missing:
            raise ValueError(f"permanent magnet cells lack assignments: {missing}.")
    return dict(sorted(out.items()))


def _magnetized_length(segment) -> float:
    if segment.geometry_kind is SegmentGeometryKind.AXIAL_PRISMATIC:
        return float(segment.length)
    if segment.geometry_kind is SegmentGeometryKind.RADIAL_CYLINDRICAL:
        return float(segment.outer_radius) - float(segment.inner_radius)
    raise ValueError("unsupported segment geometry kind.")


def build_permanent_magnet_branch_source(physical_branch: PhysicalBranch, materials: Mapping[int, Any], magnet_assignments: Mapping[int, PermanentMagnetAssignment | MagnetizationDirection | MagnetizationAxis]) -> PermanentMagnetBranchSource:
    branch = Branch(physical_branch.branch_id, physical_branch.start_node_id, physical_branch.end_node_id, physical_branch.orientation, physical_branch.total_centerline_length, 0.0, 0.0)
    b_hat = branch_direction_vector(branch)
    sources = []
    for seg in physical_branch.segments:
        mat = materials.get(seg.material_id)
        if not _is_pm(mat):
            continue
        if seg.cell_id not in magnet_assignments:
            raise ValueError(f"missing magnetization assignment for PM cell {seg.cell_id}.")
        assignment = magnet_assignments[seg.cell_id]
        direction = assignment.magnetization if isinstance(assignment, PermanentMagnetAssignment) else coerce_magnetization_direction(assignment)
        m_hat = direction.unit_vector
        projection = max(-1.0, min(1.0, m_hat[0] * b_hat[0] + m_hat[1] * b_hat[1]))
        length = _magnetized_length(seg)
        hc = float(mat.coercive_field_strength)
        sources.append(PermanentMagnetSegmentSource(seg.branch_id, seg.segment_index, seg.cell_id, seg.material_id, hc, projection, length, hc * length * projection))
    sources_t = tuple(sorted(sources, key=lambda s: s.segment_index))
    return PermanentMagnetBranchSource(physical_branch.branch_id, sources_t, sum(s.mmf for s in sources_t))


def build_permanent_magnet_branch_sources(mesh: Mesh, physical_branches: Mapping[int, PhysicalBranch], materials: Mapping[int, Any], magnet_assignments: Mapping[int, PermanentMagnetAssignment | MagnetizationDirection | MagnetizationAxis]) -> dict[int, PermanentMagnetBranchSource]:
    if set(physical_branches) != set(mesh.branches):
        raise ValueError("physical branches must match mesh branches.")
    return {bid: build_permanent_magnet_branch_source(physical_branches[bid], materials, magnet_assignments) for bid in sorted(mesh.branches)}


def build_permanent_magnet_excitation(mesh: Mesh, physical_branches: Mapping[int, PhysicalBranch], materials: Mapping[int, Any], magnet_assignments: Mapping[int, PermanentMagnetAssignment | MagnetizationDirection | MagnetizationAxis], *, additional_branch_mmf_by_id: Mapping[int, float] | None = None, nodal_flux_by_id: Mapping[int, float] | None = None) -> MagneticExcitation:
    sources = build_permanent_magnet_branch_sources(mesh, physical_branches, materials, magnet_assignments)
    index = build_topology_index(mesh)
    branch_mmf_by_id = {bid: sources[bid].total_mmf for bid in sorted(sources)}
    for bid, value in (additional_branch_mmf_by_id or {}).items():
        if bid not in index.branch_id_to_column:
            raise ValueError(f"unknown branch ID {bid} in additional_branch_mmf_by_id.")
        if not isfinite(float(value)):
            raise ValueError("additional branch MMF values must be finite.")
        branch_mmf_by_id[bid] = branch_mmf_by_id.get(bid, 0.0) + float(value)
    return build_magnetic_excitation(mesh, branch_mmf_by_id=branch_mmf_by_id, nodal_flux_by_id=nodal_flux_by_id)


def build_branch_source_components(mesh: Mesh, branch_sources: Mapping[int, PermanentMagnetBranchSource], additional_branch_mmf_by_id: Mapping[int, float] | None = None) -> dict[int, BranchSourceComponents]:
    return {bid: BranchSourceComponents(branch_sources[bid].total_mmf, 0.0, float((additional_branch_mmf_by_id or {}).get(bid, 0.0))) for bid in sorted(mesh.branches)}
