import math
from collections import deque
from src.machine.dssr_afpm_phi_z import default_dssr_afpm_phi_z_config, build_dssr_afpm_phi_z_model
from src.mesh import BranchOrientation

def test_phi_z_grid_and_topology():
    c=default_dssr_afpm_phi_z_config(); m=build_dssr_afpm_phi_z_model(c); phi=m.mesh_model.phi_edges; z=m.mesh_model.z_edges
    assert all(b>a for a,b in zip(phi,phi[1:])); assert all(b>a for a,b in zip(z,z[1:]))
    assert phi[0]==0.0 and math.isclose(phi[-1],2*math.pi)
    for k in range(c.slot_count):
        center=c.slot_phase_offset+k*c.slot_pitch_angle; half=.5*c.slot_opening_ratio*c.slot_pitch_angle
        assert any(math.isclose(e%(2*math.pi),(center-half)%(2*math.pi),abs_tol=1e-11) for e in phi)
        assert any(math.isclose(e%(2*math.pi),(center+half)%(2*math.pi),abs_tol=1e-11) for e in phi)
    half=.5*c.magnet_angular_width
    for k in range(c.pole_count):
        center=c.magnet_position_offset+c.rotor_mechanical_angle+k*c.pole_pitch_angle
        assert any(math.isclose(e%(2*math.pi),(center-half)%(2*math.pi),abs_tol=1e-11) for e in phi)
        assert any(math.isclose(e%(2*math.pi),(center+half)%(2*math.pi),abs_tol=1e-11) for e in phi)
    nphi=len(phi)-1; nz=len(z)-1
    assert all(cid == j*nphi+i for j in range(nz) for i,cid in enumerate(range(j*nphi,(j+1)*nphi)))
    assert len(m.mesh_model.periodic_branch_ids)==nz
    assert m.mesh.number_of_circumferential_branches == nphi*nz
    assert m.mesh.number_of_axial_branches == nphi*(nz-1)
    adj={n:set() for n in m.mesh.nodes}
    for b in m.mesh.branches.values(): adj[b.start_node_id].add(b.end_node_id); adj[b.end_node_id].add(b.start_node_id)
    seen={0}; q=deque([0])
    while q:
        n=q.popleft()
        for nb in adj[n]-seen: seen.add(nb); q.append(nb)
    assert len(seen)==m.mesh.number_of_nodes
