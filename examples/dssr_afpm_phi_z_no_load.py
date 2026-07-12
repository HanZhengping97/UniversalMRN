from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import csv
from pathlib import Path
from collections import Counter
from src.machine.dssr_afpm_phi_z import default_dssr_afpm_phi_z_config, build_dssr_afpm_phi_z_model, solve_dssr_afpm_phi_z_no_load, DssrAfpmRegion
from src.post.plot_phi_z import generate_all_phi_z_figures
from src.mesh import BranchOrientation

def export_csv(result):
    model=result.model; Path('output').mkdir(exist_ok=True)
    nphi=len(model.mesh_model.phi_edges)-1
    with open('output/dssr_afpm_cells.csv','w',newline='') as f:
        w=csv.writer(f); w.writerow(['cell_id','phi_index','z_index','phi_center_rad','phi_center_deg','z_center_m','region','material_id','magnetization'])
        for cid,c in sorted(model.mesh.cells.items()):
            mag=model.permanent_magnet_assignments.get(cid); w.writerow([cid,cid%nphi,cid//nphi,c.center_x,c.center_x*180/3.141592653589793,c.center_y,model.mesh_model.cell_id_to_region[cid].name,c.material_id,mag.magnetization if mag else ''])
    sources=result.field_solution.excitation_diagnostics['branch_sources']
    with open('output/dssr_afpm_branches.csv','w',newline='') as f:
        w=csv.writer(f); w.writerow(['branch_id','orientation','start_node_id','end_node_id','first_cell_id','second_cell_id','periodic','reluctance','permeance','pm_mmf','flux','max_abs_B','max_abs_H'])
        for bid,b in sorted(model.mesh.branches.items()):
            st=result.field_solution.branches[bid]; pb=model.physical_branches[bid]
            w.writerow([bid,b.orientation.name,b.start_node_id,b.end_node_id,*model.mesh.branch_id_to_cell_ids[bid],bid in model.mesh_model.periodic_branch_ids,pb.reluctance,pb.permeance,sources[bid].total_mmf,st.flux,st.max_abs_B,st.max_abs_H])
    with open('output/dssr_afpm_airgap_B.csv','w',newline='') as f:
        w=csv.writer(f); w.writerow(['phi_rad','phi_deg','upper_Bz_T','lower_Bz_T'])
        for p,u,l in zip(result.upper_airgap.phi,result.upper_airgap.flux_density_axial,result.lower_airgap.flux_density_axial): w.writerow([p,p*180/3.141592653589793,u,l])

def main():
    cfg=default_dssr_afpm_phi_z_config()
    model=build_dssr_afpm_phi_z_model(cfg)
    print('DSSR AFPM phi-z no-load demonstration (linear demo dimensions, not optimized)')
    print(f'nodes={model.mesh.number_of_nodes} cells={model.mesh.number_of_cells} branches={model.mesh.number_of_branches}')
    print(f'circumferential={model.mesh.number_of_circumferential_branches} axial={model.mesh.number_of_axial_branches} periodic={model.number_of_periodic_branches}')
    print('regions:', {k.name:v for k,v in model.region_counts.items()})
    print('materials:', Counter(c.material_id for c in model.mesh.cells.values()))
    print('PM+ PM-:', model.region_counts[DssrAfpmRegion.PERMANENT_MAGNET_POSITIVE], model.region_counts[DssrAfpmRegion.PERMANENT_MAGNET_NEGATIVE])
    result=solve_dssr_afpm_phi_z_no_load(model)
    print(f'maximum residual={result.maximum_nodal_residual:.3e}')
    print(f'upper airgap max/RMS/mean B={result.maximum_upper_airgap_B:.6g}/{result.rms_upper_airgap_B:.6g}/{result.mean_upper_airgap_B:.6g} T')
    print(f'lower airgap max/RMS/mean B={result.maximum_lower_airgap_B:.6g}/{result.rms_lower_airgap_B:.6g}/{result.mean_lower_airgap_B:.6g} T')
    print(f'upper/lower symmetry error={result.upper_lower_symmetry_error:.3e} T')
    generate_all_phi_z_figures(result); export_csv(result)
    print('wrote figures/ and output/ runtime artifacts')
if __name__=='__main__': main()
