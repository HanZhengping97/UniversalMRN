from src.machine.dssr_afpm_phi_z import *

def test_regions_materials_offsets_and_pm():
    m=build_dssr_afpm_phi_z_model(default_dssr_afpm_phi_z_config(upper_stator_slot_offset=0.01, lower_stator_slot_offset=0.02))
    counts=m.region_counts
    for r in DssrAfpmRegion: assert counts[r] > 0
    assert counts[DssrAfpmRegion.PERMANENT_MAGNET_POSITIVE] == counts[DssrAfpmRegion.PERMANENT_MAGNET_NEGATIVE]
    for cid,r in m.mesh_model.cell_id_to_region.items():
        mat=m.mesh.cells[cid].material_id
        assert (r.name.endswith('AIR') or 'AIRGAP' in r.name) == (mat==0) or mat in (1,2,3)
    moved=build_dssr_afpm_phi_z_model(default_dssr_afpm_phi_z_config(rotor_mechanical_angle=0.01))
    assert [cid for cid,r in m.mesh_model.cell_id_to_region.items() if 'MAGNET' in r.name] != [cid for cid,r in moved.mesh_model.cell_id_to_region.items() if 'MAGNET' in r.name]
