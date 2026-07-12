import numpy as np
from src.machine.dssr_afpm_phi_z import *
from src.material import MagnetizationAxis
from src.mrn import build_permanent_magnet_assignments

def test_no_load_solution_profiles_invariance_and_46_pole():
    m=build_dssr_afpm_phi_z_model(default_dssr_afpm_phi_z_config())
    r0=solve_dssr_afpm_phi_z_no_load(m, reference_node_id=0); r1=solve_dssr_afpm_phi_z_no_load(m, reference_node_id=1)
    assert np.all(np.isfinite(r0.field_solution.linear_solution.node_potential))
    assert np.all(np.isfinite([b.flux for b in r0.field_solution.branches.values()]))
    assert r0.maximum_nodal_residual < 1e-8 and r0.field_solution.maximum_absolute_flux > 0
    assert np.all(np.isfinite(r0.upper_airgap.flux_density_axial)); assert len(r0.upper_airgap.phi)==len(m.mesh_model.phi_edges)-1
    assert np.allclose([b.flux for b in r0.field_solution.branches.values()],[b.flux for b in r1.field_solution.branches.values()])
    rev={cid:(MagnetizationAxis.CIRCUMFERENTIAL_NEGATIVE if a.magnetization.circumferential>0 else MagnetizationAxis.CIRCUMFERENTIAL_POSITIVE) for cid,a in m.permanent_magnet_assignments.items()}
    revm=DssrAfpmPhiZModel(m.config,m.mesh_model,m.materials,build_permanent_magnet_assignments(m.mesh,rev,materials=m.materials),m.physical_branches)
    rr=solve_dssr_afpm_phi_z_no_load(revm)
    assert np.allclose(r0.upper_airgap.flux_density_axial, -rr.upper_airgap.flux_density_axial)
    assert r0.upper_lower_symmetry_error < 1.0
    solve_dssr_afpm_phi_z_no_load(build_dssr_afpm_phi_z_model(default_dssr_afpm_phi_z_config(pole_count=46)))
