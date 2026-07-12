import numpy as np, pytest
from src.mesh import BranchOrientation, generate_cell_centered_axisymmetric_mesh
from src.mrn import MU0, build_physical_branch_model
from src.solver import build_magnetic_excitation, recover_magnetic_field_solution, solve_linear_mrn, solve_physical_linear_mrn

def fixture():
    m=generate_cell_centered_axisymmetric_mesh([1,2,3],[0,1,2], material_ids=[[0,1],[2,0]])
    mats={0:MU0,1:MU0*50,2:MU0*100}
    phys=build_physical_branch_model(m,mats)
    return m,mats,phys

def solve(excite=1.0, **kw):
    m,mats,phys=fixture(); e=build_magnetic_excitation(m, branch_mmf_by_id={0:excite})
    lin=solve_linear_mrn(m,{i:b.permeance for i,b in phys.items()},e,**kw)
    return m,mats,phys,recover_magnetic_field_solution(m,phys,lin)

def test_zero_excitation_all_fields_zero():
    m,mats,phys=fixture(); sol=solve_physical_linear_mrn(m,mats)
    for b in sol.branches.values():
        assert b.flux == pytest.approx(0); assert b.net_mmf == pytest.approx(0)
        for s in b.segment_states:
            assert s.flux_density_average == pytest.approx(0); assert s.magnetic_field_strength_average == pytest.approx(0); assert s.mmf_drop == pytest.approx(0)

def test_nonzero_series_continuity_mmf_sum_and_finite():
    *_, sol=solve(2.0)
    assert sol.maximum_absolute_flux > 0 and sol.maximum_absolute_flux_density > 0
    for b in sol.branches.values():
        assert all(np.isfinite([b.flux,b.net_mmf,b.max_abs_B,b.max_abs_H]))
        assert b.total_segment_mmf_drop == pytest.approx(b.net_mmf)
        for s in b.segment_states: assert s.flux == pytest.approx(b.flux)

def test_axial_field_relations_and_radial_diagnostics():
    m,mats,phys,sol=solve(3.0)
    for bid,b in sol.branches.items():
        pb=phys[bid]
        for seg,state in zip(pb.segments,b.segment_states):
            assert state.mmf_drop == pytest.approx(state.flux*seg.reluctance)
            if pb.orientation is BranchOrientation.AXIAL:
                assert state.flux_density_average == pytest.approx(state.flux/seg.area)
                assert state.magnetic_field_strength_average == pytest.approx(state.flux_density_average/seg.permeability)
                assert state.mmf_drop == pytest.approx(state.magnetic_field_strength_average*seg.length)
            else:
                binn=state.flux/(seg.angular_span*seg.axial_height*seg.inner_radius)
                bout=state.flux/(seg.angular_span*seg.axial_height*seg.outer_radius)
                assert abs(state.flux_density_max) == pytest.approx(max(abs(binn),abs(bout)))

def test_sign_preservation_negative_flux():
    *_, sol=solve(-2.0)
    b=sol.branches[0]
    assert b.flux < 0
    assert all(s.flux_density_average < 0 and s.magnetic_field_strength_average < 0 and s.mmf_drop < 0 for s in b.segment_states)

def test_reference_and_high_level_invariance_and_aggregates():
    m,mats,phys=fixture(); e=build_magnetic_excitation(m, branch_mmf_by_id={0:1.0})
    a=solve_physical_linear_mrn(m,mats,e,reference_node_id=0)
    c=solve_physical_linear_mrn(m,mats,e,reference_node_id=1, reference_potential=5.0)
    for bid in a.branches: assert a.branches[bid].flux == pytest.approx(c.branches[bid].flux)
    manual=recover_magnetic_field_solution(m,phys,solve_linear_mrn(m,{i:b.permeance for i,b in phys.items()},e))
    hi=solve_physical_linear_mrn(m,mats,e)
    assert [manual.branches[i].flux for i in manual.branches] == pytest.approx([hi.branches[i].flux for i in hi.branches])
    assert set(hi.maximum_absolute_B_by_material) == {0,1,2}
    assert hi.branch_id_of_maximum_absolute_flux_density in hi.branches

def test_missing_physical_branch_and_bad_dimensions_rejected():
    m,mats,phys=fixture(); lin=solve_linear_mrn(m,{i:b.permeance for i,b in phys.items()})
    bad=dict(phys); bad.pop(0)
    with pytest.raises(ValueError, match='physical branch mapping'):
        recover_magnetic_field_solution(m,bad,lin)
    from src.solver.solution import LinearMagneticSolution
    bogus=LinearMagneticSolution(lin.node_potential, lin.branch_mmf[:-1], lin.branch_potential_drop[:-1], lin.branch_flux[:-1], lin.nodal_flux_residual, lin.reference_node_id, lin.reference_row, lin.reference_potential)
    with pytest.raises(ValueError, match='dimensions'):
        recover_magnetic_field_solution(m,phys,bogus)
