from math import pi
import numpy as np
import pytest

from src.machine.dssr_afpm_phi_z import *
from src.machine.dssr_afpm_phi_z import _pm_index
from src.mesh import BranchOrientation
from src.material import MagnetizationAxis
from src.mrn import build_permanent_magnet_branch_sources
from src.post.plot_phi_z import plot_material_map, plot_spoke_rotor_detail


def _model(**kw):
    return build_dssr_afpm_phi_z_model(default_dssr_afpm_phi_z_config(**kw))


def test_spoke_interval_counts_centers_and_widths():
    c=default_dssr_afpm_phi_z_config()
    assert len(magnet_centers(c)) == c.pole_count
    assert len(rotor_iron_pole_centers(c)) == c.pole_count
    assert np.allclose(np.diff(sorted(magnet_centers(c))), c.pole_pitch_angle)
    for pm,pole in zip(magnet_centers(c), rotor_iron_pole_centers(c)):
        assert (pole-pm)%(2*pi) == pytest.approx(0.5*c.pole_pitch_angle)
    assert c.magnet_angular_width == pytest.approx(c.magnet_arc_ratio*c.pole_pitch_angle)
    assert c.rotor_iron_pole_angular_width == pytest.approx(c.pole_pitch_angle-c.magnet_angular_width)
    assert c.magnet_angular_width < c.rotor_iron_pole_angular_width


def test_width_input_compatibility_and_inconsistency_rejected():
    c=default_dssr_afpm_phi_z_config()
    w=c.magnet_angular_width*c.mean_radius
    assert default_dssr_afpm_phi_z_config(magnet_circumferential_width=w).magnet_angular_width == pytest.approx(c.magnet_angular_width)
    assert default_dssr_afpm_phi_z_config(magnet_circumferential_width=w*1.2).magnet_arc_ratio == pytest.approx(c.magnet_arc_ratio*1.2)
    with pytest.raises(ValueError, match='inconsistent'):
        default_dssr_afpm_phi_z_config(magnet_circumferential_width=w*1.2, magnet_arc_ratio=c.magnet_arc_ratio)


def test_no_phi_cell_crosses_pm_iron_boundary_and_seam_classifies():
    m=_model(rotor_mechanical_angle=-0.01)
    edges=m.mesh_model.phi_edges; c=m.config
    boundaries=[]
    for ctr in magnet_centers(c):
        boundaries += [(ctr-0.5*c.magnet_angular_width)%(2*pi), (ctr+0.5*c.magnet_angular_width)%(2*pi)]
    for b in boundaries:
        assert any(abs(((e-b+pi)%(2*pi))-pi) < 1e-10 for e in edges)
    nphi=len(edges)-1
    rotor_rows=[cid//nphi for cid,r in m.mesh_model.cell_id_to_region.items() if r in (DssrAfpmRegion.ROTOR_IRON_POLE,DssrAfpmRegion.PERMANENT_MAGNET_POSITIVE,DssrAfpmRegion.PERMANENT_MAGNET_NEGATIVE)]
    j=rotor_rows[0]
    regs=[m.mesh_model.cell_id_to_region[j*nphi+i] for i in range(nphi)]
    assert regs[0] in (DssrAfpmRegion.ROTOR_IRON_POLE,DssrAfpmRegion.PERMANENT_MAGNET_POSITIVE,DssrAfpmRegion.PERMANENT_MAGNET_NEGATIVE)


def test_pm_alternates_and_iron_separates_every_pair():
    c=default_dssr_afpm_phi_z_config(); eps=1e-7
    signs=[]
    for k,ctr in enumerate(magnet_centers(c)):
        r=classify_region(c, ctr, c.stator_yoke_thickness+c.stator_tooth_height+c.airgap_length+0.5*c.rotor_axial_thickness, {'lower_yoke_tooth_interface_z':0,'lower_stator_surface_z':0,'lower_rotor_surface_z':0,'upper_rotor_surface_z':999,'upper_stator_surface_z':999,'upper_tooth_yoke_interface_z':999})
        signs.append(r)
    assert all(signs[i] != signs[(i+1)%len(signs)] for i in range(len(signs)))
    for ctr in rotor_iron_pole_centers(c):
        assert _pm_index(c, ctr) is None


def test_pm_assignments_and_sources_are_circumferential_only_and_alternate():
    m=_model(); sources=build_permanent_magnet_branch_sources(m.mesh,m.physical_branches,m.materials,m.permanent_magnet_assignments)
    for cid,a in m.permanent_magnet_assignments.items():
        assert abs(a.magnetization.circumferential) == 1.0 and a.magnetization.radial == 0.0 and a.magnetization.axial == 0.0
    for cid,r in m.mesh_model.cell_id_to_region.items():
        if r is DssrAfpmRegion.ROTOR_IRON_POLE:
            assert cid not in m.permanent_magnet_assignments
    circ=[s for bid,s in sources.items() if m.physical_branches[bid].orientation is BranchOrientation.CIRCUMFERENTIAL and abs(s.total_mmf)>0]
    axial=[s for bid,s in sources.items() if m.physical_branches[bid].orientation is BranchOrientation.AXIAL]
    assert circ and all(abs(s.total_mmf)==0 for s in axial)
    diag=permanent_magnet_source_diagnostics(m)
    nonzero=[d['magnet_source_mmf'] for d in diag]
    assert any(x>0 for x in nonzero) and any(x<0 for x in nonzero)


def test_pole_face_polarity_alternates_and_plots_succeed(tmp_path):
    m=_model(); prof=rotor_pole_face_profile(m)
    assert [s.expected_polarity for s in prof.samples[:4]] == ['N','S','N','S']
    plot_material_map(m, tmp_path/'map.png')
    plot_spoke_rotor_detail(m, tmp_path/'detail.png')
    assert (tmp_path/'map.png').exists() and (tmp_path/'detail.png').exists()


def test_default_and_46_pole_solve_finite():
    for pc in (44,46):
        r=solve_dssr_afpm_phi_z_no_load(_model(pole_count=pc))
        assert np.isfinite(r.maximum_upper_airgap_B)
        assert np.isfinite(r.maximum_nodal_residual)
