from math import log, pi
import pytest
from src.mesh import BranchOrientation, MeshGenerator, generate_cell_centered_axisymmetric_mesh
from src.mrn import MU0, SegmentGeometryKind, build_physical_branch_model, calculate_cell_centered_mesh_branch_permeances

def mesh(mats=None): return generate_cell_centered_axisymmetric_mesh([1,2,3],[0,1,2], material_ids=mats)

def test_physical_axial_branch_series_reluctance():
    m=mesh([[0,0],[0,0]]); pb=build_physical_branch_model(m,{0:MU0*10})
    b=next(x for x in pb.values() if x.orientation is BranchOrientation.AXIAL)
    assert b.number_of_segments == 2
    assert b.reluctance == pytest.approx(sum(s.reluctance for s in b.segments))
    assert b.permeance == pytest.approx(1/b.reluctance)
    assert all(s.geometry_kind is SegmentGeometryKind.AXIAL_PRISMATIC for s in b.segments)

def test_physical_radial_branch_exact_log_formula():
    m=mesh([[0,0],[0,0]]); mu=MU0*7; b=build_physical_branch_model(m,{0:mu})[0]
    assert all(s.geometry_kind is SegmentGeometryKind.RADIAL_CYLINDRICAL for s in b.segments)
    expected=log(2.5/1.5)/(mu*2*pi*1.0)
    assert b.reluctance == pytest.approx(expected)

def test_material_interface_has_no_single_material_and_ordering():
    m=mesh([[0,1],[0,1]]); model=build_physical_branch_model(m,{0:MU0,1:MU0*1000})
    assert list(model) == sorted(model)
    b=model[0]
    assert b.is_material_interface
    assert b.material_ids == (0,1)
    assert not hasattr(b, 'material_id')

def test_existing_permeance_helper_matches_physical_model():
    m=mesh([[0,1],[2,0]]); mats={0:MU0,1:MU0*10,2:MU0*100}
    model=build_physical_branch_model(m,mats)
    helper=calculate_cell_centered_mesh_branch_permeances(m,mats)
    assert helper == pytest.approx({bid: b.permeance for bid,b in model.items()})

def test_vertex_centered_mesh_rejected():
    m=MeshGenerator.generate_rectangular_mesh(nx=1, ny=1, width=1, height=1)
    with pytest.raises(ValueError, match='cell-centered'):
        build_physical_branch_model(m,{0:MU0})
