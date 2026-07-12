from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from src.mesh import BranchOrientation

@dataclass(frozen=True, slots=True)
class AirgapFluxDensityProfile:
    phi: np.ndarray
    mechanical_angle: np.ndarray
    branch_ids: tuple[int,...]
    flux_density_axial: np.ndarray
    airgap_name: str

def _extract(model, solution, branch_ids, name):
    rows=[]
    for bid in branch_ids:
        c1,c2=model.mesh.branch_id_to_cell_ids[bid]; cell=model.mesh.cells[c1]
        b=solution.branches[bid]
        air_seg=[s for s in b.segment_states if model.mesh.cells[s.cell_id].material_id==0][0]
        rows.append((cell.center_x,bid,air_seg.flux_density_average))
    rows.sort()
    return AirgapFluxDensityProfile(np.array([r[0] for r in rows]), np.array([r[0] for r in rows]), tuple(r[1] for r in rows), np.array([r[2] for r in rows]), name)

def extract_upper_airgap_profile(model, solution): return _extract(model, solution, model.upper_airgap_axial_branch_ids, 'upper')
def extract_lower_airgap_profile(model, solution): return _extract(model, solution, model.lower_airgap_axial_branch_ids, 'lower')
