from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.machine.dssr_afpm_phi_z import default_dssr_afpm_phi_z_config, build_dssr_afpm_phi_z_model, solve_dssr_afpm_phi_z_no_load
if __name__ == '__main__':
    model=build_dssr_afpm_phi_z_model(default_dssr_afpm_phi_z_config(pole_count=46))
    result=solve_dssr_afpm_phi_z_no_load(model)
    print(model.mesh.number_of_cells, result.maximum_upper_airgap_B, result.maximum_nodal_residual)
