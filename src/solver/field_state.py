"""Recover immutable branch and segment magnetic field state."""
from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite, sqrt, pi
from types import MappingProxyType
from typing import Mapping


from src.mesh import Mesh
from src.mrn.physical_branch import PhysicalBranch, SegmentGeometryKind, build_physical_branch_model
from src.solver.excitation import MagneticExcitation
from src.solver.linear import solve_linear_mrn
from src.solver.solution import LinearMagneticSolution
from src.topology import build_topology_index

@dataclass(frozen=True, slots=True)
class SegmentFieldState:
    branch_id: int; segment_index: int; cell_id: int; material_id: int
    flux: float; flux_density_average: float; flux_density_min: float; flux_density_max: float
    magnetic_field_strength_average: float; magnetic_field_strength_min: float; magnetic_field_strength_max: float
    mmf_drop: float
    @property
    def flux_density(self) -> float: return self.flux_density_average
    @property
    def magnetic_field_strength(self) -> float: return self.magnetic_field_strength_average
    def __post_init__(self) -> None:
        for name in ("flux","flux_density_average","flux_density_min","flux_density_max","magnetic_field_strength_average","magnetic_field_strength_min","magnetic_field_strength_max","mmf_drop"):
            if not isfinite(float(getattr(self, name))): raise ValueError(f"{name} must be finite.")

@dataclass(frozen=True, slots=True)
class BranchFieldState:
    branch_id: int; source_mmf: float; potential_drop: float; net_mmf: float; flux: float
    equivalent_reluctance: float; equivalent_permeance: float; segment_states: tuple[SegmentFieldState, ...]
    def __post_init__(self) -> None:
        for name in ("source_mmf","potential_drop","net_mmf","flux","equivalent_reluctance","equivalent_permeance"):
            if not isfinite(float(getattr(self, name))): raise ValueError(f"{name} must be finite.")
        if self.equivalent_reluctance <= 0 or self.equivalent_permeance <= 0: raise ValueError("equivalent reluctance/permeance must be positive.")
        if not isclose(self.flux, self.equivalent_permeance*self.net_mmf, rel_tol=1e-8, abs_tol=1e-18): raise ValueError("branch constitutive equation mismatch.")
        if not isclose(self.net_mmf, self.flux*self.equivalent_reluctance, rel_tol=1e-8, abs_tol=1e-12): raise ValueError("branch MMF equation mismatch.")
        if not isclose(sum(s.mmf_drop for s in self.segment_states), self.net_mmf, rel_tol=1e-8, abs_tol=1e-12): raise ValueError("segment MMF drops do not sum to branch net MMF.")
    @property
    def max_abs_B(self) -> float: return max((max(abs(s.flux_density_min), abs(s.flux_density_max), abs(s.flux_density_average)) for s in self.segment_states), default=0.0)
    @property
    def max_abs_H(self) -> float: return max((max(abs(s.magnetic_field_strength_min), abs(s.magnetic_field_strength_max), abs(s.magnetic_field_strength_average)) for s in self.segment_states), default=0.0)
    @property
    def number_of_segments(self) -> int: return len(self.segment_states)
    @property
    def is_material_interface(self) -> bool: return len({s.material_id for s in self.segment_states}) > 1
    @property
    def total_segment_mmf_drop(self) -> float: return sum(s.mmf_drop for s in self.segment_states)

@dataclass(frozen=True, slots=True)
class MagneticFieldSolution:
    linear_solution: LinearMagneticSolution; branches: Mapping[int, BranchFieldState]; excitation_diagnostics: object | None = None
    def __post_init__(self) -> None:
        ordered = dict(sorted(self.branches.items()))
        if len(ordered) != self.linear_solution.number_of_branches: raise ValueError("branch count must match linear solution.")
        object.__setattr__(self, "branches", MappingProxyType(ordered))
    @property
    def maximum_absolute_flux(self) -> float: return max((abs(b.flux) for b in self.branches.values()), default=0.0)
    @property
    def maximum_absolute_flux_density(self) -> float: return max((b.max_abs_B for b in self.branches.values()), default=0.0)
    @property
    def branch_id_of_maximum_absolute_flux_density(self) -> int | None: return max(self.branches, key=lambda bid: self.branches[bid].max_abs_B) if self.branches else None
    @property
    def maximum_absolute_field_strength(self) -> float: return max((b.max_abs_H for b in self.branches.values()), default=0.0)
    @property
    def branch_id_of_maximum_absolute_field_strength(self) -> int | None: return max(self.branches, key=lambda bid: self.branches[bid].max_abs_H) if self.branches else None
    @property
    def material_interface_branch_count(self) -> int: return sum(b.is_material_interface for b in self.branches.values())
    @property
    def maximum_absolute_B_by_material(self) -> dict[int, float]:
        out: dict[int,float] = {}
        for b in self.branches.values():
            for s in b.segment_states: out[s.material_id] = max(out.get(s.material_id, 0.0), abs(s.flux_density_max), abs(s.flux_density_min), abs(s.flux_density_average))
        return out
    @property
    def maximum_absolute_H_by_material(self) -> dict[int, float]:
        out: dict[int,float] = {}
        for b in self.branches.values():
            for s in b.segment_states: out[s.material_id] = max(out.get(s.material_id, 0.0), abs(s.magnetic_field_strength_max), abs(s.magnetic_field_strength_min), abs(s.magnetic_field_strength_average))
        return out

def _segment_state(seg, flux: float) -> SegmentFieldState:
    mmf = flux * seg.reluctance
    if seg.geometry_kind in (SegmentGeometryKind.AXIAL_PRISMATIC, SegmentGeometryKind.CIRCUMFERENTIAL_PRISMATIC):
        b = flux / seg.area; h = b / seg.permeability
        return SegmentFieldState(seg.branch_id, seg.segment_index, seg.cell_id, seg.material_id, flux, b, b, b, h, h, h, mmf)
    ri, ro, hz, span = seg.inner_radius, seg.outer_radius, seg.axial_height, seg.angular_span
    rc = sqrt(ri*ro); bavg = flux/(span*hz*rc); binn = flux/(span*hz*ri); bout = flux/(span*hz*ro)
    bmax = binn if abs(binn) >= abs(bout) else bout; bmin = bout if abs(bout) <= abs(binn) else binn
    return SegmentFieldState(seg.branch_id, seg.segment_index, seg.cell_id, seg.material_id, flux, bavg, bmin, bmax, bavg/seg.permeability, bmin/seg.permeability, bmax/seg.permeability, mmf)

def recover_magnetic_field_solution(mesh: Mesh, physical_branches: Mapping[int, PhysicalBranch], linear_solution: LinearMagneticSolution) -> MagneticFieldSolution:
    index = build_topology_index(mesh)
    if mesh.number_of_branches != linear_solution.number_of_branches: raise ValueError("solution branch vector dimensions do not match mesh.")
    if set(physical_branches) != set(mesh.branches): raise ValueError("physical branch mapping must match mesh branches.")
    states = {}
    for bid in sorted(mesh.branches):
        col = index.branch_id_to_column[bid]; pb = physical_branches[bid]
        source = float(linear_solution.branch_mmf[col]); drop = float(linear_solution.branch_potential_drop[col]); flux = float(linear_solution.branch_flux[col])
        net = source - drop
        seg_states = tuple(_segment_state(s, flux) for s in pb.segments)
        states[bid] = BranchFieldState(bid, source, drop, net, flux, pb.reluctance, pb.permeance, seg_states)
    return MagneticFieldSolution(linear_solution, states)

def solve_permanent_magnet_linear_mrn(mesh, materials, magnet_assignments, *, additional_branch_mmf_by_id=None, angular_span=2*pi, reference_node_id=None, reference_potential=0.0, residual_tolerance=1e-9) -> MagneticFieldSolution:
    from src.mrn.permanent_magnet_source import build_branch_source_components, build_permanent_magnet_branch_sources, build_permanent_magnet_excitation
    physical = build_physical_branch_model(mesh, materials, angular_span=angular_span)
    branch_sources = build_permanent_magnet_branch_sources(mesh, physical, materials, magnet_assignments)
    excitation = build_permanent_magnet_excitation(mesh, physical, materials, magnet_assignments, additional_branch_mmf_by_id=additional_branch_mmf_by_id)
    linear = solve_linear_mrn(mesh, {bid: b.permeance for bid,b in physical.items()}, excitation, reference_node_id=reference_node_id, reference_potential=reference_potential, residual_tolerance=residual_tolerance)
    solution = recover_magnetic_field_solution(mesh, physical, linear)
    source_by_segment = {(src.branch_id, src.segment_index): src for branch_src in branch_sources.values() for src in branch_src.segment_sources}
    adjusted_branches = {}
    for bid, branch_state in solution.branches.items():
        adjusted_segments = []
        for seg_state in branch_state.segment_states:
            src = source_by_segment.get((seg_state.branch_id, seg_state.segment_index))
            if src is None:
                adjusted_segments.append(seg_state)
                continue
            h_offset = src.coercive_field_strength * src.magnetization_projection
            adjusted_segments.append(SegmentFieldState(
                seg_state.branch_id, seg_state.segment_index, seg_state.cell_id, seg_state.material_id,
                seg_state.flux, seg_state.flux_density_average, seg_state.flux_density_min, seg_state.flux_density_max,
                seg_state.magnetic_field_strength_average - h_offset,
                seg_state.magnetic_field_strength_min - h_offset,
                seg_state.magnetic_field_strength_max - h_offset,
                seg_state.mmf_drop,
            ))
        adjusted_branches[bid] = BranchFieldState(
            branch_state.branch_id, branch_state.source_mmf, branch_state.potential_drop, branch_state.net_mmf, branch_state.flux,
            branch_state.equivalent_reluctance, branch_state.equivalent_permeance, tuple(adjusted_segments)
        )
    solution = MagneticFieldSolution(linear, adjusted_branches)
    object.__setattr__(solution, "excitation_diagnostics", {
        "branch_sources": branch_sources,
        "source_components": build_branch_source_components(mesh, branch_sources, additional_branch_mmf_by_id),
    })
    return solution

def solve_physical_linear_mrn(mesh, materials, excitation: MagneticExcitation | None = None, *, angular_span=2*pi, reference_node_id=None, reference_potential=0.0, residual_tolerance=1e-9) -> MagneticFieldSolution:
    physical = build_physical_branch_model(mesh, materials, angular_span=angular_span)
    linear = solve_linear_mrn(mesh, {bid: b.permeance for bid,b in physical.items()}, excitation, reference_node_id=reference_node_id, reference_potential=reference_potential, residual_tolerance=residual_tolerance)
    return recover_magnetic_field_solution(mesh, physical, linear)
