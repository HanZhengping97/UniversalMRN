"""Cell-centered axisymmetric mesh generation utilities."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite

from .branch import Branch, BranchOrientation
from .cell import Cell
from .mesh import Mesh
from .node import Node


def _validate_edges(edges: Sequence[float], name: str) -> list[float]:
    values = [float(v) for v in edges]
    if len(values) < 2:
        raise ValueError(f"{name} must contain at least two entries.")
    if any(not isfinite(v) for v in values):
        raise ValueError(f"{name} must be finite.")
    if any(b <= a for a, b in zip(values, values[1:])):
        raise ValueError(f"{name} must be strictly increasing.")
    return values


def generate_cell_centered_axisymmetric_mesh(
    radial_edges: Sequence[float],
    axial_edges: Sequence[float],
    *,
    material_ids: Sequence[Sequence[int]] | None = None,
) -> Mesh:
    """Generate a control-volume-centered r-z reluctance-network mesh.

    One node is placed at each cell center and branches connect centers of
    adjacent control volumes.  Cell and node identifiers follow
    ``cell_id = node_id = j * nr + i``; explicit maps are also stored on the
    mesh so callers do not need to rely on that equality.
    """

    r_edges = _validate_edges(radial_edges, "radial_edges")
    z_edges = _validate_edges(axial_edges, "axial_edges")
    nr = len(r_edges) - 1
    nz = len(z_edges) - 1

    if material_ids is not None:
        if len(material_ids) != nz or any(len(row) != nr for row in material_ids):
            raise ValueError("material_ids must have shape [nz][nr].")

    mesh = Mesh()
    mesh.cell_id_to_node_id = {}
    mesh.node_id_to_cell_id = {}
    mesh.branch_id_to_cell_ids = {}
    mesh.cell_centered = True

    for j in range(nz):
        for i in range(nr):
            cell_id = j * nr + i
            r0, r1 = r_edges[i], r_edges[i + 1]
            z0, z1 = z_edges[j], z_edges[j + 1]
            material_id = 0 if material_ids is None else int(material_ids[j][i])
            mesh.add_cell(Cell(id=cell_id, center_x=0.5 * (r0 + r1), center_y=0.5 * (z0 + z1), dx=r1 - r0, dy=z1 - z0, material_id=material_id))
            mesh.add_node(Node(id=cell_id, x=0.5 * (r0 + r1), y=0.5 * (z0 + z1)))
            mesh.cell_id_to_node_id[cell_id] = cell_id
            mesh.node_id_to_cell_id[cell_id] = cell_id

    branch_id = 0
    for j in range(nz):
        for i in range(nr - 1):
            c0 = j * nr + i
            c1 = j * nr + i + 1
            n0 = mesh.get_node(c0)
            n1 = mesh.get_node(c1)
            mesh.add_branch(Branch(id=branch_id, start_node_id=n0.id, end_node_id=n1.id, orientation=BranchOrientation.RADIAL, length=n1.r - n0.r, center_r=0.5 * (n0.r + n1.r), center_z=n0.z))
            mesh.branch_id_to_cell_ids[branch_id] = (c0, c1)
            branch_id += 1

    for j in range(nz - 1):
        for i in range(nr):
            c0 = j * nr + i
            c1 = (j + 1) * nr + i
            n0 = mesh.get_node(c0)
            n1 = mesh.get_node(c1)
            mesh.add_branch(Branch(id=branch_id, start_node_id=n0.id, end_node_id=n1.id, orientation=BranchOrientation.AXIAL, length=n1.z - n0.z, center_r=n0.r, center_z=0.5 * (n0.z + n1.z)))
            mesh.branch_id_to_cell_ids[branch_id] = (c0, c1)
            branch_id += 1

    return mesh
