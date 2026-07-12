import math
from src.machine.dssr_afpm_phi_z import *
from src.mesh import BranchOrientation

def test_reluctance_positive_and_formulas():
    c=default_dssr_afpm_phi_z_config(); m=build_dssr_afpm_phi_z_model(c)
    for pb in m.physical_branches.values(): assert pb.reluctance>0 and math.isfinite(pb.permeance)
    cb=next(pb for pb in m.physical_branches.values() if pb.orientation is BranchOrientation.CIRCUMFERENTIAL and not pb.is_material_interface)
    s=cb.segments[0]; assert math.isclose(s.reluctance, s.length/(s.permeability*s.area), rel_tol=1e-12)
    ab=next(pb for pb in m.physical_branches.values() if pb.orientation is BranchOrientation.AXIAL)
    s=ab.segments[0]; assert math.isclose(s.area, .5*(c.outer_radius**2-c.inner_radius**2)*m.mesh.cells[s.cell_id].dx, rel_tol=1e-12)
    seam=m.physical_branches[m.mesh_model.periodic_branch_ids[0]]; assert seam.total_centerline_length < c.mean_radius*0.1
