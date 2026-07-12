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
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    phi=np.degrees(result.upper_airgap.phi); upper=result.upper_airgap.flux_density_axial; lower=result.lower_airgap.flux_density_axial
    fig,(ax0,ax1)=plt.subplots(2,1,figsize=(10,7),sharex=True)
    ax0.plot(phi, upper, label='signed upper $B_z$')
    ax0.plot(phi, lower, label='signed lower $B_z$')
    ax0.plot(phi, -lower, '--', label='sign-normalized lower $-B_z$')
    ax0.set_ylabel('axial B [T]'); ax0.legend()
    ax1.plot(phi, upper+lower, label='$B_{upper}+B_{lower}$')
    ax1.axhline(0,color='0.5',lw=.8); ax1.set_xlabel('mechanical angle [deg]'); ax1.set_ylabel('residual [T]'); ax1.legend()
    fig.tight_layout(); fig.savefig(path); plt.close(fig)

def plot_airgap_harmonics(result, path='figures/dssr_afpm_airgap_harmonics.png'):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    up=result.upper_airgap_harmonics; low=result.sign_normalized_lower_airgap_harmonics
    plt.figure(figsize=(10,4)); plt.stem(up.harmonic_orders, up.amplitudes, linefmt='C0-', markerfmt='C0o', basefmt=' ', label='upper')
    plt.stem(low.harmonic_orders, low.amplitudes, linefmt='C1-', markerfmt='C1s', basefmt=' ', label='sign-normalized lower')
    plt.xlabel('mechanical spatial harmonic order'); plt.ylabel('amplitude [T]'); plt.legend(); plt.tight_layout(); plt.savefig(path); plt.close()

def generate_all_phi_z_figures(result):
    plot_material_map(result.model); plot_mesh(result.model); plot_branches(result.model); plot_airgap(result)
