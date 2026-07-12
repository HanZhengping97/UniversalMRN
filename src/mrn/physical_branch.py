"""Immutable physical branch models for cell-centered MRN meshes."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from math import isclose, isfinite, log, pi
from typing import Any, Mapping

from src.mesh import Branch, BranchOrientation, Mesh
from .interface_reluctance import MU0, calculate_branch_reluctance_segments

class SegmentGeometryKind(Enum):
    AXIAL_PRISMATIC = auto()
    RADIAL_CYLINDRICAL = auto()

def _pos(v: float, name: str) -> float:
    v = float(v)
    if not isfinite(v) or v <= 0:
        raise ValueError(f"{name} must be positive and finite.")
    return v

def _nonneg(v: int, name: str) -> int:
    if int(v) < 0:
        raise ValueError(f"{name} must be non-negative.")
    return int(v)

def _rel_mu(material: Any) -> float | None:
    if hasattr(material, "relative_permeability"):
        return float(material.relative_permeability)
    if hasattr(material, "mu_r"):
        return float(material.mu_r)
    if isinstance(material, Mapping):
        if "relative_permeability" in material: return float(material["relative_permeability"])
        if "mu_r" in material: return float(material["mu_r"])
        if "permeability" in material: return float(material["permeability"]) / MU0
        if "mu" in material: return float(material["mu"]) / MU0
    if isinstance(material, (int, float)):
        return float(material) / MU0
    if hasattr(material, "permeability"):
        return float(material.permeability) / MU0
    if hasattr(material, "mu"):
        return float(material.mu) / MU0
    return None

@dataclass(frozen=True, slots=True)
class PhysicalBranchSegment:
    branch_id: int
    segment_index: int
    cell_id: int
    material_id: int
    length: float
    area: float
    permeability: float
    reluctance: float
    geometry_kind: SegmentGeometryKind = SegmentGeometryKind.AXIAL_PRISMATIC
    inner_radius: float | None = None
    outer_radius: float | None = None
    axial_height: float | None = None
    angular_span: float | None = None
    relative_permeability: float | None = None

    def __post_init__(self) -> None:
        _nonneg(self.branch_id, "branch_id"); _nonneg(self.segment_index, "segment_index")
        _nonneg(self.cell_id, "cell_id"); _nonneg(self.material_id, "material_id")
        _pos(self.length, "length"); _pos(self.area, "area"); _pos(self.permeability, "permeability"); _pos(self.reluctance, "reluctance")
        if self.geometry_kind is SegmentGeometryKind.AXIAL_PRISMATIC:
            expected = self.length / (self.permeability * self.area)
        elif self.geometry_kind is SegmentGeometryKind.RADIAL_CYLINDRICAL:
            ri = _pos(self.inner_radius, "inner_radius"); ro = _pos(self.outer_radius, "outer_radius")
            h = _pos(self.axial_height, "axial_height"); span = _pos(self.angular_span, "angular_span")
            if ro <= ri: raise ValueError("outer_radius must exceed inner_radius.")
            expected = log(ro / ri) / (self.permeability * span * h)
        else:
            raise ValueError("unsupported segment geometry kind.")
        if not isclose(self.reluctance, expected, rel_tol=1e-9, abs_tol=1e-18):
            raise ValueError("segment reluctance is inconsistent with geometry and permeability.")
        if self.relative_permeability is not None and (not isfinite(float(self.relative_permeability)) or float(self.relative_permeability) <= 0):
            raise ValueError("relative_permeability must be positive and finite.")

    @property
    def effective_area(self) -> float:
        """Equivalent constant area length/(mu*R); diagnostic for radial segments."""
        return self.length / (self.permeability * self.reluctance)

@dataclass(frozen=True, slots=True)
class PhysicalBranch:
    branch_id: int
    start_node_id: int
    end_node_id: int
    orientation: BranchOrientation
    segments: tuple[PhysicalBranchSegment, ...]
    reluctance: float
    permeance: float

    def __post_init__(self) -> None:
        _nonneg(self.branch_id, "branch_id"); _nonneg(self.start_node_id, "start_node_id"); _nonneg(self.end_node_id, "end_node_id")
        if not self.segments: raise ValueError("physical branch requires at least one segment.")
        if tuple(s.segment_index for s in self.segments) != tuple(range(len(self.segments))):
            raise ValueError("segments must be in deterministic segment_index order.")
        if any(s.branch_id != self.branch_id for s in self.segments): raise ValueError("segment branch IDs must match branch_id.")
        _pos(self.reluctance, "reluctance"); _pos(self.permeance, "permeance")
        if not isclose(self.reluctance, sum(s.reluctance for s in self.segments), rel_tol=1e-9): raise ValueError("branch reluctance must equal sum of segment reluctances.")
        if not isclose(self.permeance, 1.0 / self.reluctance, rel_tol=1e-9): raise ValueError("branch permeance must equal inverse reluctance.")

    @property
    def number_of_segments(self) -> int: return len(self.segments)
    @property
    def material_ids(self) -> tuple[int, ...]: return tuple(s.material_id for s in self.segments)
    @property
    def is_material_interface(self) -> bool: return len(set(self.material_ids)) > 1
    @property
    def total_centerline_length(self) -> float: return sum(s.length for s in self.segments)
    @property
    def equivalent_permeability(self) -> float | None:
        if len({s.geometry_kind for s in self.segments}) != 1: return None
        if self.orientation is BranchOrientation.AXIAL and len({round(s.area, 15) for s in self.segments}) == 1:
            return self.total_centerline_length / (self.reluctance * self.segments[0].area)
        return None
    @property
    def minimum_relative_permeability(self) -> float | None:
        vals = [s.relative_permeability for s in self.segments if s.relative_permeability is not None]
        return min(vals) if vals else None
    @property
    def maximum_relative_permeability(self) -> float | None:
        vals = [s.relative_permeability for s in self.segments if s.relative_permeability is not None]
        return max(vals) if vals else None

def build_physical_branch(mesh: Mesh, branch: Branch, materials: Mapping[int, Any], *, angular_span: float = 2*pi) -> PhysicalBranch:
    if not mesh.cell_centered:
        raise ValueError("physical branch model requires a cell-centered mesh.")
    if branch.id not in mesh.branch_id_to_cell_ids:
        raise ValueError("physical branch model requires branch_id_to_cell_ids mapping.")
    raw = calculate_branch_reluctance_segments(mesh, branch, materials, full_annulus=False, angular_span=angular_span)
    if len(raw) != 2: raise ValueError("cell-centered physical branches require exactly two series segments.")
    segs = []
    for i, r in enumerate(raw):
        kind = SegmentGeometryKind.AXIAL_PRISMATIC if branch.orientation is BranchOrientation.AXIAL else SegmentGeometryKind.RADIAL_CYLINDRICAL
        area = r.area if r.area is not None else r.length / (r.permeability * r.reluctance)
        cell = mesh.get_cell(r.cell_id)
        segs.append(PhysicalBranchSegment(branch.id, i, r.cell_id, r.material_id, r.length, area, r.permeability, r.reluctance, kind, r.inner_radius, r.outer_radius, cell.dy if kind is SegmentGeometryKind.RADIAL_CYLINDRICAL else None, angular_span if kind is SegmentGeometryKind.RADIAL_CYLINDRICAL else None, _rel_mu(materials[r.material_id])))
    reluctance = sum(s.reluctance for s in segs)
    return PhysicalBranch(branch.id, branch.start_node_id, branch.end_node_id, branch.orientation, tuple(segs), reluctance, 1.0/reluctance)

def build_physical_branch_model(mesh: Mesh, materials: Mapping[int, Any], *, angular_span: float = 2*pi) -> dict[int, PhysicalBranch]:
    if not mesh.cell_centered:
        raise ValueError("physical branch model requires a cell-centered mesh.")
    return {bid: build_physical_branch(mesh, mesh.branches[bid], materials, angular_span=angular_span) for bid in sorted(mesh.branches)}
