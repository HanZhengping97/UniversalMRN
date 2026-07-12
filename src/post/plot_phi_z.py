from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

REGION_COLORS = {
    'UPPER_STATOR_YOKE':'#4c72b0','UPPER_STATOR_TOOTH':'#5b8fd1','UPPER_SLOT_AIR':'#d9eef7','UPPER_AIRGAP':'#ffffff',
    'ROTOR_IRON_POLE':'#888888','PERMANENT_MAGNET_POSITIVE':'#d62728','PERMANENT_MAGNET_NEGATIVE':'#1f77b4',
    'LOWER_AIRGAP':'#ffffff','LOWER_SLOT_AIR':'#d9eef7','LOWER_STATOR_TOOTH':'#5b8fd1','LOWER_STATOR_YOKE':'#4c72b0',
}

def _region_array(model):
    regs=list(type(next(iter(model.mesh_model.cell_id_to_region.values()))))
    order={r:i for i,r in enumerate(regs)}
    nphi=len(model.mesh_model.phi_edges)-1; nz=len(model.mesh_model.z_edges)-1
    arr=np.array([[order[model.mesh_model.cell_id_to_region[j*nphi+i]] for i in range(nphi)] for j in range(nz)])
    return regs, arr

def plot_material_map(model, path='figures/dssr_afpm_phi_z_material_map.png'):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    regs, arr = _region_array(model)
    colors=[REGION_COLORS.get(r.name,'#cccccc') for r in regs]
    fig, ax = plt.subplots(figsize=(12,4))
    ax.pcolormesh(np.degrees(model.mesh_model.phi_edges), np.array(model.mesh_model.z_edges)*1000, arr, shading='flat', cmap=ListedColormap(colors), norm=BoundaryNorm(np.arange(len(regs)+1)-0.5, len(regs)))
    ax.set_xlabel('mechanical angle [deg]'); ax.set_ylabel('z [mm]'); ax.set_title('DSSR spoke-type AFPM phi-z material regions')
    labels={'ROTOR_IRON_POLE':'rotor iron pole piece','PERMANENT_MAGNET_POSITIVE':'PM +phi','PERMANENT_MAGNET_NEGATIVE':'PM -phi','UPPER_AIRGAP':'air gap','LOWER_AIRGAP':'air gap','UPPER_STATOR_TOOTH':'stator steel','LOWER_STATOR_TOOTH':'stator steel'}
    handles=[]; seen=set()
    for r in regs:
        lab=labels.get(r.name, r.name.lower().replace('_',' '))
        if lab not in seen:
            handles.append(Patch(color=REGION_COLORS.get(r.name,'#ccc'), label=lab)); seen.add(lab)
    ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.01,0.5), fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)

def plot_mesh(model, path='figures/dssr_afpm_phi_z_mesh.png'):
    Path(path).parent.mkdir(parents=True, exist_ok=True); plt.figure(figsize=(12,4))
    for x in np.degrees(model.mesh_model.phi_edges): plt.axvline(x,color='0.85',lw=0.3)
    for y in np.array(model.mesh_model.z_edges)*1000: plt.axhline(y,color='0.6',lw=0.5)
    plt.xlabel('mechanical angle [deg]'); plt.ylabel('z [mm]'); plt.tight_layout(); plt.savefig(path); plt.close()

def plot_branches(model, path='figures/dssr_afpm_phi_z_branches.png', decimate=20):
    Path(path).parent.mkdir(parents=True, exist_ok=True); plt.figure(figsize=(12,4)); k=0
    for b in model.mesh.branches.values():
        if k%decimate==0:
            n1=model.mesh.nodes[b.start_node_id]; n2=model.mesh.nodes[b.end_node_id]; plt.plot([np.degrees(n1.x),np.degrees(n2.x)],[n1.y*1000,n2.y*1000],'r-' if b.orientation.name.startswith('CIRC') else 'b-',lw=.4)
        k+=1
    plt.xlabel('mechanical angle [deg]'); plt.ylabel('z [mm]'); plt.tight_layout(); plt.savefig(path); plt.close()

def plot_airgap(result, path='figures/dssr_afpm_airgap_B.png'):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    phi=np.degrees(result.upper_airgap.phi); upper=result.upper_airgap.flux_density_axial; lower=result.lower_airgap.flux_density_axial
    fig,(ax0,ax1)=plt.subplots(2,1,figsize=(10,7),sharex=True)
    ax0.plot(phi, upper, 'o-', ms=2, label='signed upper $B_z$'); ax0.plot(phi, lower, 's-', ms=2, label='signed lower $B_z$'); ax0.plot(phi, -lower, '--', label='sign-normalized lower $-B_z$')
    ax0.set_ylabel('axial B [T]'); ax0.legend(); ax1.plot(phi, upper+lower, 'o-', ms=2, label='$B_{upper}+B_{lower}$')
    ax1.axhline(0,color='0.5',lw=.8); ax1.set_xlabel('mechanical angle [deg]'); ax1.set_ylabel('residual [T]'); ax1.legend(); fig.tight_layout(); fig.savefig(path); plt.close(fig)

def plot_airgap_harmonics(result, path='figures/dssr_afpm_airgap_harmonics.png'):
    Path(path).parent.mkdir(parents=True, exist_ok=True); up=result.upper_airgap_harmonics; low=result.sign_normalized_lower_airgap_harmonics
    plt.figure(figsize=(10,4)); plt.stem(up.harmonic_orders, up.amplitudes, linefmt='C0-', markerfmt='C0o', basefmt=' ', label='upper'); plt.stem(low.harmonic_orders, low.amplitudes, linefmt='C1-', markerfmt='C1s', basefmt=' ', label='sign-normalized lower')
    plt.xlabel('mechanical spatial harmonic order'); plt.ylabel('amplitude [T]'); plt.legend(); plt.tight_layout(); plt.savefig(path); plt.close()

def plot_spoke_rotor_detail(model, path='figures/dssr_afpm_spoke_rotor_detail.png', pole_pitches=4):
    from src.machine.dssr_afpm_phi_z import rotor_pole_face_profile
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    regs, arr=_region_array(model); n=int(np.searchsorted(model.mesh_model.phi_edges, pole_pitches*model.config.pole_pitch_angle, side='right'))
    fig, ax=plt.subplots(figsize=(10,4))
    ax.pcolormesh(np.degrees(model.mesh_model.phi_edges[:n+1]), np.array(model.mesh_model.z_edges)*1000, arr[:,:n], shading='flat', cmap=ListedColormap([REGION_COLORS.get(r.name,'#ccc') for r in regs]), norm=BoundaryNorm(np.arange(len(regs)+1)-0.5, len(regs)))
    iface=model.mesh_model.interface_z or {}; ax.text(0,(iface['upper_rotor_surface_z']+model.config.airgap_length/2)*1000,'upper air gap',fontsize=8); ax.text(0,(iface['lower_rotor_surface_z']-model.config.airgap_length/2)*1000,'lower air gap',fontsize=8)
    for p in rotor_pole_face_profile(model).samples[:pole_pitches]: ax.text(np.degrees(p.phi), (iface['upper_rotor_surface_z']-0.001)*1000, p.expected_polarity, ha='center', va='top', fontweight='bold')
    zc=0.5*(iface['lower_rotor_surface_z']+iface['upper_rotor_surface_z'])*1000
    for k in range(pole_pitches):
        phi=model.config.magnet_position_offset+k*model.config.pole_pitch_angle
        ax.annotate('', xy=(np.degrees(phi)+(4 if k%2==0 else -4), zc), xytext=(np.degrees(phi), zc), arrowprops=dict(arrowstyle='->', color='k'))
        ax.text(np.degrees(phi), zc+1, 'PM +phi' if k%2==0 else 'PM -phi', ha='center', fontsize=8)
    ax.set_xlabel('mechanical angle [deg]'); ax.set_ylabel('z [mm]'); ax.set_title('DSSR spoke rotor detail: PM | iron pole | PM')
    fig.tight_layout(); fig.savefig(path); plt.close(fig)

def generate_all_phi_z_figures(result):
    plot_material_map(result.model); plot_mesh(result.model); plot_branches(result.model); plot_airgap(result); plot_spoke_rotor_detail(result.model)
