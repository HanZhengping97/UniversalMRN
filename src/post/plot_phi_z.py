from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def plot_material_map(model, path='figures/dssr_afpm_phi_z_material_map.png'):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    nphi=len(model.mesh_model.phi_edges)-1; nz=len(model.mesh_model.z_edges)-1
    arr=np.array([[model.mesh_model.cell_id_to_region[j*nphi+i].value for i in range(nphi)] for j in range(nz)])
    plt.figure(figsize=(12,4)); plt.pcolormesh(np.degrees(model.mesh_model.phi_edges), np.array(model.mesh_model.z_edges)*1000, arr, shading='flat'); plt.xlabel('mechanical angle [deg]'); plt.ylabel('z [mm]'); plt.title('DSSR AFPM phi-z regions'); plt.colorbar(label='region enum'); plt.tight_layout(); plt.savefig(path); plt.close()

def plot_mesh(model, path='figures/dssr_afpm_phi_z_mesh.png'):
    Path(path).parent.mkdir(parents=True, exist_ok=True); plt.figure(figsize=(12,4))
    for x in np.degrees(model.mesh_model.phi_edges): plt.axvline(x,color='0.85',lw=0.3)
    for y in np.array(model.mesh_model.z_edges)*1000: plt.axhline(y,color='0.6',lw=0.5)
    plt.xlabel('mechanical angle [deg]'); plt.ylabel('z [mm]'); plt.tight_layout(); plt.savefig(path); plt.close()

def plot_branches(model, path='figures/dssr_afpm_phi_z_branches.png', decimate=20):
    Path(path).parent.mkdir(parents=True, exist_ok=True); plt.figure(figsize=(12,4)); k=0
    for b in model.mesh.branches.values():
        if k%decimate==0:
            n1=model.mesh.nodes[b.start_node_id]; n2=model.mesh.nodes[b.end_node_id]; x=[np.degrees(n1.x),np.degrees(n2.x)]; y=[n1.y*1000,n2.y*1000]; plt.plot(x,y,'r-' if b.orientation.name.startswith('CIRC') else 'b-',lw=.4)
        k+=1
    plt.xlabel('mechanical angle [deg]'); plt.ylabel('z [mm]'); plt.tight_layout(); plt.savefig(path); plt.close()

def plot_airgap(result, path='figures/dssr_afpm_airgap_B.png'):
    Path(path).parent.mkdir(parents=True, exist_ok=True); plt.figure(figsize=(10,4)); plt.plot(np.degrees(result.upper_airgap.phi), result.upper_airgap.flux_density_axial, label='upper'); plt.plot(np.degrees(result.lower_airgap.phi), result.lower_airgap.flux_density_axial, label='lower'); plt.xlabel('mechanical angle [deg]'); plt.ylabel('axial B [T]'); plt.legend(); plt.tight_layout(); plt.savefig(path); plt.close()

def generate_all_phi_z_figures(result):
    plot_material_map(result.model); plot_mesh(result.model); plot_branches(result.model); plot_airgap(result)
