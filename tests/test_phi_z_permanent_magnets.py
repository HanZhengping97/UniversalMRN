import math
from src.machine.dssr_afpm_phi_z import *
from src.mrn import build_permanent_magnet_branch_sources
from src.mesh import BranchOrientation

def test_circumferential_pm_sources_and_pole_counts():
    for poles in (44,46):
        m=build_dssr_afpm_phi_z_model(default_dssr_afpm_phi_z_config(pole_count=poles))
        src=build_permanent_magnet_branch_sources(m.mesh,m.physical_branches,m.materials,m.permanent_magnet_assignments)
        vals=[s.total_mmf for bid,s in src.items() if m.mesh.branches[bid].orientation is BranchOrientation.CIRCUMFERENTIAL and abs(s.total_mmf)>0]
        assert vals and max(vals)>0 and min(vals)<0
        ax=[s.total_mmf for bid,s in src.items() if m.mesh.branches[bid].orientation is BranchOrientation.AXIAL]
        assert all(abs(v)<1e-12 for v in ax)
        one=next(s.segment_sources[0] for s in src.values() if s.segment_sources)
        assert math.isclose(abs(one.mmf), one.coercive_field_strength*one.magnetized_length, rel_tol=1e-12)
