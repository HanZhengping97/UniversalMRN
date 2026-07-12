from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import csv
import numpy as np
from src.machine.dssr_afpm_phi_z import default_dssr_afpm_phi_z_config, build_dssr_afpm_phi_z_model, solve_dssr_afpm_phi_z_no_load, DssrAfpmRegion

def export(model,result):
    Path('output').mkdir(exist_ok=True)
    with open('output/dssr_afpm_cells.csv','w',newline='') as f:
        w=csv.writer(f); w.writerow(['cell_id','phi','z','region','material_id'])
        for cid,c in model.mesh.cells.items(): w.writerow([cid,c.center_x,c.center_y,model.mesh_model.cell_id_to_region[cid].name,c.material_id])
    with open('output/dssr_afpm_branches.csv','w',newline='') as f:
        w=csv.writer(f); w.writerow(['branch_id','orientation','c1','c2','reluctance','permeance'])
        for bid,b in model.physical_branches.items():
            c1,c2=model.mesh.branch_id_to_cell_ids[bid]; w.writerow([bid,b.orientation.name,c1,c2,b.reluctance,b.permeance])
    with open('output/dssr_afpm_airgap_B.csv','w',newline='') as f:
        w=csv.writer(f); w.writerow(['phi_rad','phi_deg','upper_Bz_T','lower_Bz_T','lower_Bz_sign_normalized_T','signed_symmetry_residual_T','magnitude_symmetry_residual_T'])
        for p,u,l in zip(result.upper_airgap.phi,result.upper_airgap.flux_density_axial,result.lower_airgap.flux_density_axial):
            w.writerow([p,np.degrees(p),u,l,-l,u+l,abs(u)-abs(l)])
    with open('output/dssr_afpm_airgap_harmonics.csv','w',newline='') as f:
        up=result.upper_airgap_harmonics; lo=result.lower_airgap_harmonics; sn=result.sign_normalized_lower_airgap_harmonics
        w=csv.writer(f); w.writerow(['harmonic_order','upper_amplitude_T','lower_amplitude_T','sign_normalized_lower_amplitude_T'])
        for k,u,l,s in zip(up.harmonic_orders,up.amplitudes,lo.amplitudes,sn.amplitudes): w.writerow([k,u,l,s])

def main():
    cfg=default_dssr_afpm_phi_z_config(); model=build_dssr_afpm_phi_z_model(cfg); result=solve_dssr_afpm_phi_z_no_load(model)
    export(model,result)
    p=cfg.pole_count//2; h=result.upper_airgap_harmonics
    print(f'cells={model.mesh.number_of_cells} branches={model.mesh.number_of_branches} pm_cells={model.number_of_pm_cells}')
    print(f'upper airgap max/RMS/mean B={result.maximum_upper_airgap_B:.6g}/{result.rms_upper_airgap_B:.6g}/{result.mean_upper_airgap_B:.6g} T')
    print(f'lower airgap max/RMS/mean B={result.maximum_lower_airgap_B:.6g}/{result.rms_lower_airgap_B:.6g}/{result.mean_lower_airgap_B:.6g} T')
    print(f'signed upper/lower symmetry error={result.signed_upper_lower_symmetry_error:.3e} T')
    print(f'normalized signed symmetry error={result.normalized_signed_symmetry_error:.3e}')
    print(f'magnitude symmetry error={result.magnitude_upper_lower_symmetry_error:.3e} T')
    print(f'upper total signed air-gap flux={result.upper_airgap_flux_integral:.6e} Wb')
    print(f'lower total signed air-gap flux={result.lower_airgap_flux_integral:.6e} Wb')
    print(f'total flux-balance error={result.total_airgap_flux_balance_error:.3e} Wb')
    print(f'dominant spatial harmonic={h.dominant_mechanical_order}')
    print(f'pole-pair harmonic order {p} amplitude={h.amplitude_at(p):.6g} T')
if __name__=='__main__': main()
