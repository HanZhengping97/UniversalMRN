"""Cell-centered unwrapped phi-z mesh generator."""
from __future__ import annotations
from dataclasses import dataclass
from math import pi, floor, ceil
from typing import Callable, Mapping, Any
from types import MappingProxyType
from src.mesh import Mesh, Node, Cell, Branch, BranchOrientation

def periodic_angle(x: float, period: float=2*pi) -> float:
    return (x + 0.5*period) % period - 0.5*period

def _add_edge(edges, x, tol=1e-12):
    x=x%(2*pi)
    if abs(x-2*pi)<tol or x<tol: x=0.0
    edges.append(x)

def build_phi_edges(config) -> tuple[float,...]:
    edges=[0.0,2*pi]
    # uniform refinement
    n=max(config.slot_count*config.circumferential_cells_per_slot, config.pole_count*config.circumferential_cells_per_pole)
    for i in range(n): edges.append(2*pi*i/n)
    sop=0.5*config.slot_opening_ratio*config.slot_pitch_angle
    for off in (config.upper_stator_slot_offset, config.lower_stator_slot_offset):
        base=config.slot_phase_offset+off
        for k in range(config.slot_count):
            c=base+k*config.slot_pitch_angle
            _add_edge(edges,c-sop); _add_edge(edges,c+sop)
    mw=0.5*config.magnet_circumferential_width/config.mean_radius
    base=config.magnet_position_offset+config.rotor_mechanical_angle
    for k in range(config.pole_count):
        c=base+k*config.pole_pitch_angle
        _add_edge(edges,c-mw); _add_edge(edges,c+mw)
    vals=sorted(edges)
    out=[]
    for v in vals:
        if not out or abs(v-out[-1])>1e-11: out.append(0.0 if abs(v)<1e-12 else (2*pi if abs(v-2*pi)<1e-12 else v))
    if out[0]!=0.0: out=[0.0]+out
    if abs(out[-1]-2*pi)>1e-12: out.append(2*pi)
    return tuple(out)

def _subdivide(start, end, layers):
    return [start+(end-start)*i/layers for i in range(layers)]

def build_z_edges(c):
    z=0.0; edges=[]; iface={}
    edges += _subdivide(z, z+c.stator_yoke_thickness, c.lower_yoke_layers); z += c.stator_yoke_thickness; iface['lower_yoke_tooth_interface_z']=z
    edges += _subdivide(z, z+c.stator_tooth_height, c.lower_tooth_layers); z += c.stator_tooth_height; iface['lower_stator_surface_z']=z
    edges += _subdivide(z, z+c.airgap_length, c.lower_airgap_layers); z += c.airgap_length; iface['lower_rotor_surface_z']=z
    edges += _subdivide(z, z+c.rotor_axial_thickness, c.rotor_layers); z += c.rotor_axial_thickness; iface['upper_rotor_surface_z']=z
    edges += _subdivide(z, z+c.airgap_length, c.upper_airgap_layers); z += c.airgap_length; iface['upper_stator_surface_z']=z
    edges += _subdivide(z, z+c.stator_tooth_height, c.upper_tooth_layers); z += c.stator_tooth_height; iface['upper_tooth_yoke_interface_z']=z
    edges += _subdivide(z, z+c.stator_yoke_thickness, c.upper_yoke_layers); z += c.stator_yoke_thickness
    edges.append(z)
    return tuple(edges), MappingProxyType(iface)

@dataclass(frozen=True, slots=True)
class PhiZMeshModel:
    mesh: Mesh
    phi_edges: tuple[float,...]
    z_edges: tuple[float,...]
    mean_radius: float
    radial_span: float
    cell_id_to_region: Mapping[int, Any]
    periodic_branch_ids: tuple[int,...]
    interface_z: Mapping[str,float] | None = None

def generate_phi_z_mesh(phi_edges, z_edges, mean_radius, radial_span, classify_region: Callable[[float,float],Any], material_for_region: Callable[[Any],int]) -> PhiZMeshModel:
    mesh=Mesh(cell_centered=True); nphi=len(phi_edges)-1; nz=len(z_edges)-1; regions={}
    for j in range(nz):
        for i in range(nphi):
            cid=j*nphi+i; pc=0.5*(phi_edges[i]+phi_edges[i+1]); zc=0.5*(z_edges[j]+z_edges[j+1]); reg=classify_region(pc,zc); regions[cid]=reg
            mesh.add_node(Node(cid, pc, zc)); mesh.add_cell(Cell(cid, pc, zc, phi_edges[i+1]-phi_edges[i], z_edges[j+1]-z_edges[j], material_for_region(reg)))
            mesh.cell_id_to_node_id[cid]=cid; mesh.node_id_to_cell_id[cid]=cid
    bid=0; periodic=[]
    for j in range(nz):
        for i in range(nphi):
            c1=j*nphi+i; c2=j*nphi+((i+1)%nphi)
            length=mean_radius*((phi_edges[i+1]-phi_edges[i]) if i==nphi-1 else (mesh.cells[c2].center_x-mesh.cells[c1].center_x))
            if i==nphi-1: length=mean_radius*((2*pi-mesh.cells[c1].center_x)+mesh.cells[c2].center_x); periodic.append(bid)
            mesh.add_branch(Branch(bid,c1,c2,BranchOrientation.CIRCUMFERENTIAL,length,0.0,mesh.cells[c1].center_y)); mesh.branch_id_to_cell_ids[bid]=(c1,c2); bid+=1
    for j in range(nz-1):
        for i in range(nphi):
            c1=j*nphi+i; c2=(j+1)*nphi+i; length=mesh.cells[c2].center_y-mesh.cells[c1].center_y
            mesh.add_branch(Branch(bid,c1,c2,BranchOrientation.AXIAL,length,mesh.cells[c1].center_x,0.0)); mesh.branch_id_to_cell_ids[bid]=(c1,c2); bid+=1
    return PhiZMeshModel(mesh, tuple(phi_edges), tuple(z_edges), mean_radius, radial_span, MappingProxyType(dict(sorted(regions.items()))), tuple(periodic))
