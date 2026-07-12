"""DSSR spoke-type AFPM phi-z magnetic reluctance model."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from math import pi, isfinite, sqrt
from types import MappingProxyType
from typing import Mapping, Any
from collections import Counter
import numpy as np
from src.material import LinearMagneticMaterial, LinearPermanentMagnetMaterial, MagnetizationAxis, MU_0
from src.mesh.phi_z import PhiZMeshModel, generate_phi_z_mesh, periodic_angle, build_phi_edges, build_z_edges
from src.mesh import BranchOrientation
from src.mrn import PhysicalBranch, PhysicalBranchSegment, SegmentGeometryKind, PermanentMagnetAssignment, build_permanent_magnet_assignments, build_permanent_magnet_excitation, build_permanent_magnet_branch_sources, build_branch_source_components
from src.solver import solve_linear_mrn, recover_magnetic_field_solution
from src.solver.field_state import MagneticFieldSolution

AIR, STATOR_STEEL, ROTOR_STEEL, FERRITE = 0, 1, 2, 3

class DssrAfpmRegion(Enum):
    UPPER_STATOR_YOKE = auto(); UPPER_STATOR_TOOTH = auto(); UPPER_SLOT_AIR = auto(); UPPER_AIRGAP = auto()
    ROTOR_IRON = auto(); PERMANENT_MAGNET_POSITIVE = auto(); PERMANENT_MAGNET_NEGATIVE = auto()
    LOWER_AIRGAP = auto(); LOWER_SLOT_AIR = auto(); LOWER_STATOR_TOOTH = auto(); LOWER_STATOR_YOKE = auto()

@dataclass(frozen=True, slots=True)
class DssrAfpmPhiZConfig:
    outer_diameter: float = 0.301; inner_diameter: float = 0.180
    slot_count: int = 36; pole_count: int = 44
    stator_yoke_thickness: float = 0.010; stator_tooth_height: float = 0.015; slot_opening_ratio: float = 0.45
    airgap_length: float = 0.001; rotor_axial_thickness: float = 0.012; magnet_circumferential_width: float = 0.004
    upper_stator_axial_offset: float = 0.0; lower_stator_axial_offset: float = 0.0
    circumferential_cells_per_slot: int = 2; upper_yoke_layers: int = 1; upper_tooth_layers: int = 2; upper_airgap_layers: int = 2
    rotor_layers: int = 2; lower_airgap_layers: int = 2; lower_tooth_layers: int = 2; lower_yoke_layers: int = 1
    angular_span: float = 2*pi
    slot_phase_offset: float = 0.0; rotor_mechanical_angle: float = 0.0; upper_stator_slot_offset: float = 0.0; lower_stator_slot_offset: float = 0.0; magnet_position_offset: float = 0.0
    stator_steel_relative_permeability: float = 1000.0; rotor_steel_relative_permeability: float = 1000.0
    ferrite_remanence: float = 0.45; ferrite_relative_permeability: float = 1.05
    circumferential_cells_per_pole: int = 1
    def __post_init__(self):
        for n in ['outer_diameter','inner_diameter','stator_yoke_thickness','stator_tooth_height','airgap_length','rotor_axial_thickness','magnet_circumferential_width']:
            v=getattr(self,n)
            if not isfinite(v) or v<=0: raise ValueError(f'{n} must be positive and finite')
        if self.outer_diameter <= self.inner_diameter: raise ValueError('outer_diameter must exceed inner_diameter')
        if self.slot_count <= 0: raise ValueError('slot_count must be positive')
        if self.pole_count <= 0 or self.pole_count % 2: raise ValueError('pole_count must be positive and even')
        if not 0 < self.slot_opening_ratio < 1: raise ValueError('slot_opening_ratio must be between 0 and 1')
        for n in ['circumferential_cells_per_slot','upper_yoke_layers','upper_tooth_layers','upper_airgap_layers','rotor_layers','lower_airgap_layers','lower_tooth_layers','lower_yoke_layers','circumferential_cells_per_pole']:
            if getattr(self,n) <= 0: raise ValueError(f'{n} must be positive')
        if self.circumferential_cells_per_slot < 1: raise ValueError('circumferential_cells_per_slot too small')
        if self.upper_airgap_layers < 2 or self.lower_airgap_layers < 2 or self.rotor_layers < 2: raise ValueError('air gaps and rotor require at least two layers')
        if abs(self.angular_span-2*pi)>1e-12: raise ValueError('angular_span must be 2*pi')
        if self.magnet_circumferential_width >= self.mean_radius*self.pole_pitch_angle: raise ValueError('magnet width must be smaller than pole pitch')
    @property
    def outer_radius(self): return self.outer_diameter/2
    @property
    def inner_radius(self): return self.inner_diameter/2
    @property
    def mean_radius(self): return 0.5*(self.outer_radius+self.inner_radius)
    @property
    def radial_span(self): return self.outer_radius-self.inner_radius
    @property
    def slot_pitch_angle(self): return 2*pi/self.slot_count
    @property
    def pole_pitch_angle(self): return 2*pi/self.pole_count
    @property
    def circumferential_length(self): return self.mean_radius*self.angular_span
    @property
    def total_axial_length(self): return 2*self.stator_yoke_thickness+2*self.stator_tooth_height+2*self.airgap_length+self.rotor_axial_thickness
    @property
    def total_axial_layers(self): return self.lower_yoke_layers+self.lower_tooth_layers+self.lower_airgap_layers+self.rotor_layers+self.upper_airgap_layers+self.upper_tooth_layers+self.upper_yoke_layers
    @property
    def circumferential_cell_count(self): return len(build_phi_edges(self))-1

@dataclass(frozen=True, slots=True)
class DssrAfpmPhiZModel:
    config: DssrAfpmPhiZConfig; mesh_model: PhiZMeshModel; materials: Mapping[int, Any]; permanent_magnet_assignments: Mapping[int, PermanentMagnetAssignment]; physical_branches: Mapping[int, PhysicalBranch]
    @property
    def mesh(self): return self.mesh_model.mesh
    @property
    def region_counts(self): return Counter(self.mesh_model.cell_id_to_region.values())
    @property
    def number_of_pm_cells(self): return len(self.permanent_magnet_assignments)
    @property
    def number_of_periodic_branches(self): return len(self.mesh_model.periodic_branch_ids)
    @property
    def upper_airgap_cell_ids(self): return tuple(cid for cid,r in self.mesh_model.cell_id_to_region.items() if r is DssrAfpmRegion.UPPER_AIRGAP)
    @property
    def lower_airgap_cell_ids(self): return tuple(cid for cid,r in self.mesh_model.cell_id_to_region.items() if r is DssrAfpmRegion.LOWER_AIRGAP)
    @property
    def upper_airgap_axial_branch_ids(self): return _airgap_internal_axial_branches(self, DssrAfpmRegion.UPPER_AIRGAP)
    @property
    def lower_airgap_axial_branch_ids(self): return _airgap_internal_axial_branches(self, DssrAfpmRegion.LOWER_AIRGAP)

def _mat_for(r):
    if 'SLOT_AIR' in r.name or 'AIRGAP' in r.name: return AIR
    if 'MAGNET' in r.name: return FERRITE
    if 'ROTOR' in r.name: return ROTOR_STEEL
    return STATOR_STEEL

def default_dssr_afpm_phi_z_config(**kw): return DssrAfpmPhiZConfig(**kw)

def default_materials(c):
    return {AIR:LinearMagneticMaterial(AIR,'air',1.0), STATOR_STEEL:LinearMagneticMaterial(STATOR_STEEL,'linear stator steel demo',c.stator_steel_relative_permeability), ROTOR_STEEL:LinearMagneticMaterial(ROTOR_STEEL,'linear rotor steel demo',c.rotor_steel_relative_permeability), FERRITE:LinearPermanentMagnetMaterial(FERRITE,'ferrite PM',c.ferrite_relative_permeability,c.ferrite_remanence)}

def build_dssr_afpm_phi_z_model(config: DssrAfpmPhiZConfig, *, material_overrides: Mapping[int, Any] | None=None) -> DssrAfpmPhiZModel:
    mats=default_materials(config); mats.update(material_overrides or {})
    phi=build_phi_edges(config); z, iface=build_z_edges(config)
    mesh_model=generate_phi_z_mesh(phi,z,config.mean_radius,config.radial_span, lambda pc,zc: classify_region(config, pc, zc, iface), _mat_for)
    assign={cid:(MagnetizationAxis.CIRCUMFERENTIAL_POSITIVE if r is DssrAfpmRegion.PERMANENT_MAGNET_POSITIVE else MagnetizationAxis.CIRCUMFERENTIAL_NEGATIVE) for cid,r in mesh_model.cell_id_to_region.items() if 'PERMANENT_MAGNET' in r.name}
    pm=build_permanent_magnet_assignments(mesh_model.mesh, assign, materials=mats, strict=True)
    phys=_build_physical(config, mesh_model, mats)
    return DssrAfpmPhiZModel(config, mesh_model, MappingProxyType(dict(sorted(mats.items()))), MappingProxyType(pm), MappingProxyType(phys))

def classify_region(c, phi, z, iface):
    if z < iface['lower_yoke_tooth_interface_z']: return DssrAfpmRegion.LOWER_STATOR_YOKE
    if z < iface['lower_stator_surface_z']: return DssrAfpmRegion.LOWER_SLOT_AIR if _in_slot(c, phi, c.lower_stator_slot_offset) else DssrAfpmRegion.LOWER_STATOR_TOOTH
    if z < iface['lower_rotor_surface_z']: return DssrAfpmRegion.LOWER_AIRGAP
    if z < iface['upper_rotor_surface_z']:
        k=_pm_index(c, phi)
        if k is None: return DssrAfpmRegion.ROTOR_IRON
        return DssrAfpmRegion.PERMANENT_MAGNET_POSITIVE if k%2==0 else DssrAfpmRegion.PERMANENT_MAGNET_NEGATIVE
    if z < iface['upper_stator_surface_z']: return DssrAfpmRegion.UPPER_AIRGAP
    if z < iface['upper_tooth_yoke_interface_z']: return DssrAfpmRegion.UPPER_SLOT_AIR if _in_slot(c, phi, c.upper_stator_slot_offset) else DssrAfpmRegion.UPPER_STATOR_TOOTH
    return DssrAfpmRegion.UPPER_STATOR_YOKE

def _in_slot(c, phi, offset): return abs(periodic_angle(phi-(c.slot_phase_offset+offset), c.slot_pitch_angle)) <= 0.5*c.slot_opening_ratio*c.slot_pitch_angle + 1e-12

def _pm_index(c, phi):
    width=c.magnet_circumferential_width/c.mean_radius
    base=c.magnet_position_offset+c.rotor_mechanical_angle
    raw=(phi-base)/c.pole_pitch_angle; k=round(raw)
    if abs(periodic_angle(phi-(base+k*c.pole_pitch_angle), 2*pi)) <= 0.5*width+1e-12: return k % c.pole_count
    return None

def _build_physical(c, mm, mats):
    out={}; phi=mm.phi_edges; z=mm.z_edges; nphi=len(phi)-1
    for bid,b in sorted(mm.mesh.branches.items()):
        c1,c2=mm.mesh.branch_id_to_cell_ids[bid]; cells=(mm.mesh.cells[c1], mm.mesh.cells[c2]); segs=[]
        if b.orientation is BranchOrientation.CIRCUMFERENTIAL:
            j=c1//nphi; dz=z[j+1]-z[j]; kind=SegmentGeometryKind.CIRCUMFERENTIAL_PRISMATIC
            if c1 % nphi == nphi-1 and c2 % nphi == 0: dphis=(2*pi-cells[0].center_x, cells[1].center_x)
            else: dphis=(phi[c1%nphi+1]-cells[0].center_x, cells[1].center_x-phi[c2%nphi])
            area=c.radial_span*dz; lengths=(c.mean_radius*dphis[0], c.mean_radius*dphis[1])
        else:
            i=c1 % nphi; dphi=phi[i+1]-phi[i]; kind=SegmentGeometryKind.AXIAL_PRISMATIC
            area=0.5*(c.outer_radius**2-c.inner_radius**2)*dphi; lengths=(z[c1//nphi+1]-cells[0].center_y, cells[1].center_y-z[c2//nphi])
        for si,(cell,L) in enumerate(zip(cells,lengths)):
            mu=mats[cell.material_id].permeability; R=L/(mu*area)
            segs.append(PhysicalBranchSegment(bid,si,cell.id,cell.material_id,L,area,mu,R,kind,relative_permeability=mats[cell.material_id].relative_permeability))
        R=sum(s.reluctance for s in segs); out[bid]=PhysicalBranch(bid,b.start_node_id,b.end_node_id,b.orientation,tuple(segs),R,1/R)
    return out

def _airgap_internal_axial_branches(model, region):
    mm=model.mesh_model; out=[]
    for bid,b in mm.mesh.branches.items():
        if b.orientation is BranchOrientation.AXIAL:
            a,bid2=mm.mesh.branch_id_to_cell_ids[bid]
            if mm.cell_id_to_region[a] is region and mm.cell_id_to_region[bid2] is region: out.append(bid)
    return tuple(sorted(out))

@dataclass(frozen=True, slots=True)
class DssrAfpmPhiZNoLoadResult:
    model: DssrAfpmPhiZModel; field_solution: MagneticFieldSolution; upper_airgap: Any; lower_airgap: Any
    @property
    def maximum_upper_airgap_B(self): return float(np.max(np.abs(self.upper_airgap.flux_density_axial)))
    @property
    def maximum_lower_airgap_B(self): return float(np.max(np.abs(self.lower_airgap.flux_density_axial)))
    @property
    def rms_upper_airgap_B(self): return float(sqrt(np.mean(self.upper_airgap.flux_density_axial**2)))
    @property
    def rms_lower_airgap_B(self): return float(sqrt(np.mean(self.lower_airgap.flux_density_axial**2)))
    @property
    def mean_upper_airgap_B(self): return float(np.mean(self.upper_airgap.flux_density_axial))
    @property
    def mean_lower_airgap_B(self): return float(np.mean(self.lower_airgap.flux_density_axial))
    @property
    def upper_lower_symmetry_error(self): return float(np.max(np.abs(self.upper_airgap.flux_density_axial-self.lower_airgap.flux_density_axial)))
    @property
    def maximum_nodal_residual(self): return float(np.max(np.abs(self.field_solution.linear_solution.nodal_flux_residual)))

def solve_dssr_afpm_phi_z_no_load(model, *, reference_node_id=None, residual_tolerance=1e-9):
    from src.post.airgap import extract_upper_airgap_profile, extract_lower_airgap_profile
    exc=build_permanent_magnet_excitation(model.mesh, model.physical_branches, model.materials, model.permanent_magnet_assignments)
    lin=solve_linear_mrn(model.mesh,{bid:b.permeance for bid,b in model.physical_branches.items()},exc,reference_node_id=reference_node_id,residual_tolerance=residual_tolerance)
    sol=recover_magnetic_field_solution(model.mesh, model.physical_branches, lin)
    bs=build_permanent_magnet_branch_sources(model.mesh, model.physical_branches, model.materials, model.permanent_magnet_assignments)
    object.__setattr__(sol,'excitation_diagnostics',{'branch_sources':bs,'source_components':build_branch_source_components(model.mesh,bs)})
    return DssrAfpmPhiZNoLoadResult(model, sol, extract_upper_airgap_profile(model, sol), extract_lower_airgap_profile(model, sol))
