import numpy as np
import pytest
from src.mesh import BranchOrientation
from src.post.airgap import AirgapFluxDensityProfile, validate_matching_airgap_profiles
from src.machine.dssr_afpm_phi_z import *
from src.material import MagnetizationAxis
from src.mrn import build_permanent_magnet_assignments

def _profile(vals, phis=(0.1, 0.2, 0.3), name='upper'):
    phis = tuple(phis[:len(vals)]) if len(phis) != len(vals) else phis
    return AirgapFluxDensityProfile(np.array(phis), np.array(phis), tuple(range(len(vals))), np.array(vals, dtype=float), name, np.ones(len(vals)))

def test_opposite_signed_synthetic_symmetry_metrics():
    r=DssrAfpmPhiZNoLoadResult(None, None, _profile([1,2,3]), _profile([-1,-2,-3], name='lower'))
    assert r.signed_upper_lower_symmetry_error == pytest.approx(0)
    assert np.max(np.abs(r.upper_airgap.flux_density_axial-r.lower_airgap.flux_density_axial)) > 0
    assert r.magnitude_upper_lower_symmetry_error == pytest.approx(0)

def test_profile_length_and_phi_mismatch_rejected():
    with pytest.raises(ValueError, match='different lengths'):
        validate_matching_airgap_profiles(_profile([1,2,3]), _profile([-1,-2], name='lower'))
    with pytest.raises(ValueError, match='phi coordinates'):
        validate_matching_airgap_profiles(_profile([1,2,3]), _profile([-1,-2,-3], phis=(0.1,0.21,0.3), name='lower'))

def test_default_airgap_extraction_and_diagnostics():
    m=build_dssr_afpm_phi_z_model(default_dssr_afpm_phi_z_config())
    r=solve_dssr_afpm_phi_z_no_load(m, reference_node_id=0)
    for p in (r.upper_airgap, r.lower_airgap):
        assert all(m.mesh.branches[bid].orientation is BranchOrientation.AXIAL for bid in p.branch_ids)
        assert all(all(m.mesh.cells[cid].material_id == 0 for cid in m.mesh.branch_id_to_cell_ids[bid]) for bid in p.branch_ids)
    assert np.allclose(r.upper_airgap.phi, r.lower_airgap.phi)
    assert r.signed_upper_lower_symmetry_error < 1e-10
    assert r.normalized_signed_symmetry_error < 1e-9
    assert r.total_airgap_flux_balance_error < 1e-12
    assert np.all(np.isfinite(r.upper_airgap_harmonics.amplitudes))
    assert r.upper_airgap_harmonics.amplitude_at(22) > 0

def test_reversed_magnets_reference_invariance_and_46_pole_harmonic():
    m=build_dssr_afpm_phi_z_model(default_dssr_afpm_phi_z_config())
    r0=solve_dssr_afpm_phi_z_no_load(m, reference_node_id=0); r1=solve_dssr_afpm_phi_z_no_load(m, reference_node_id=1)
    assert np.allclose([b.flux for b in r0.field_solution.branches.values()],[b.flux for b in r1.field_solution.branches.values()])
    rev={cid:(MagnetizationAxis.CIRCUMFERENTIAL_NEGATIVE if a.magnetization.circumferential>0 else MagnetizationAxis.CIRCUMFERENTIAL_POSITIVE) for cid,a in m.permanent_magnet_assignments.items()}
    rr=solve_dssr_afpm_phi_z_no_load(DssrAfpmPhiZModel(m.config,m.mesh_model,m.materials,build_permanent_magnet_assignments(m.mesh,rev,materials=m.materials),m.physical_branches))
    assert np.allclose(r0.upper_airgap.flux_density_axial, -rr.upper_airgap.flux_density_axial)
    r46=solve_dssr_afpm_phi_z_no_load(build_dssr_afpm_phi_z_model(default_dssr_afpm_phi_z_config(pole_count=46)))
    assert r46.upper_airgap_harmonics.amplitude_at(23) > 0
