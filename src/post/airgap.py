from __future__ import annotations
from dataclasses import dataclass
from math import pi
import numpy as np
from src.mesh import BranchOrientation

@dataclass(frozen=True, slots=True)
class AirgapFluxDensityProfile:
    phi: np.ndarray
    mechanical_angle: np.ndarray
    branch_ids: tuple[int,...]
    flux_density_axial: np.ndarray
    airgap_name: str
    sample_areas: np.ndarray | None = None

class AirgapExtractionError(ValueError):
    pass

def _region_for_name(model, name):
    from src.machine.dssr_afpm_phi_z import DssrAfpmRegion
    return DssrAfpmRegion.UPPER_AIRGAP if name == 'upper' else DssrAfpmRegion.LOWER_AIRGAP

def _midplane(model, name):
    iface=model.mesh_model.interface_z
    if name == 'upper':
        return 0.5*(iface['upper_rotor_surface_z'] + iface['upper_stator_surface_z'])
    return 0.5*(iface['lower_stator_surface_z'] + iface['lower_rotor_surface_z'])

def _validate_profile_coordinates(rows, name, tol=1e-10):
    phis=np.array([r[0] for r in rows], dtype=float)
    if len(phis) == 0:
        raise AirgapExtractionError(f'{name} air-gap profile is empty')
    if not np.all(np.isfinite(phis)):
        raise AirgapExtractionError(f'{name} air-gap phi coordinates are not finite')
    if np.any(np.diff(phis) <= tol):
        raise AirgapExtractionError(f'{name} air-gap phi coordinates are not strictly increasing')
    if phis[0] < -tol or phis[-1] >= 2*pi + tol:
        raise AirgapExtractionError(f'{name} air-gap phi coordinates outside [0, 2*pi)')
    if abs(phis[0]) < tol and abs(phis[-1] - 2*pi) < tol:
        raise AirgapExtractionError(f'{name} air-gap profile contains duplicated 0/2*pi endpoints')

def _extract(model, solution, branch_ids, name):
    target_region=_region_for_name(model, name); mid=_midplane(model, name); rows=[]
    nphi=len(model.mesh_model.phi_edges)-1
    expected_by_phi={}
    for bid in branch_ids:
        branch=model.mesh.branches[bid]
        if branch.orientation is not BranchOrientation.AXIAL:
            raise AirgapExtractionError(f'{name} air-gap branch {bid} is not axial')
        c1,c2=model.mesh.branch_id_to_cell_ids[bid]
        cells=(model.mesh.cells[c1], model.mesh.cells[c2])
        if any(model.mesh_model.cell_id_to_region[c.id] is not target_region for c in cells):
            raise AirgapExtractionError(f'{name} air-gap branch {bid} is not internal to {target_region.name}')
        if any(c.material_id != 0 for c in cells):
            raise AirgapExtractionError(f'{name} air-gap branch {bid} is not bounded only by air cells')
        if c1 % nphi != c2 % nphi:
            raise AirgapExtractionError(f'{name} air-gap branch {bid} connects mismatched phi columns')
        expected_phi=cells[0].center_x
        if abs(branch.center_r - expected_phi) > 1e-10:
            raise AirgapExtractionError(f'{name} air-gap branch {bid} phi does not match cell center')
        # Ensure selected internal interface is nearest to the air-gap midplane for each phi column.
        z_on_branch=0.5*(cells[0].center_y + cells[1].center_y)
        col=c1 % nphi
        prior=expected_by_phi.get(col)
        dist=abs(z_on_branch-mid)
        if prior is None or dist < prior[0]:
            expected_by_phi[col]=(dist, bid)
        bstate=solution.branches[bid]
        air_seg=[s for s in bstate.segment_states if model.mesh.cells[s.cell_id].material_id==0]
        if len(air_seg) != len(bstate.segment_states):
            raise AirgapExtractionError(f'{name} air-gap branch {bid} includes a non-air segment')
        area=model.physical_branches[bid].segments[0].area
        rows.append((expected_phi,bid,air_seg[0].flux_density_average,area,col,dist))
    for _, rbid, *_rest, col, dist in rows:
        if expected_by_phi[col][1] != rbid:
            raise AirgapExtractionError(f'{name} air-gap branch {rbid} is not closest to the air-gap midplane')
    rows.sort(key=lambda r: r[0]); _validate_profile_coordinates(rows, name)
    return AirgapFluxDensityProfile(np.array([r[0] for r in rows]), np.array([r[0] for r in rows]), tuple(r[1] for r in rows), np.array([r[2] for r in rows]), name, np.array([r[3] for r in rows]))

def validate_matching_airgap_profiles(upper, lower, *, phi_tolerance=1e-10):
    if len(upper.phi) != len(lower.phi):
        raise ValueError('upper and lower air-gap profiles have different lengths')
    _validate_profile_coordinates(list(zip(upper.phi, upper.branch_ids, upper.flux_density_axial)), upper.airgap_name, phi_tolerance)
    _validate_profile_coordinates(list(zip(lower.phi, lower.branch_ids, lower.flux_density_axial)), lower.airgap_name, phi_tolerance)
    if not np.allclose(upper.phi, lower.phi, rtol=0.0, atol=phi_tolerance):
        raise ValueError('upper and lower air-gap phi coordinates do not match')

def extract_upper_airgap_profile(model, solution): return _extract(model, solution, model.upper_airgap_axial_branch_ids, 'upper')
def extract_lower_airgap_profile(model, solution): return _extract(model, solution, model.lower_airgap_axial_branch_ids, 'lower')
