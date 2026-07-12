"""Topology validation utilities for structured r-z meshes."""

from __future__ import annotations

from math import isclose

from .branch import BranchOrientation
from .mesh import Mesh


def validate_branch_topology(mesh: Mesh) -> None:
    """Validate structured adjacent-node branch topology.

    The structured dimensions are inferred from the number of unique r and z
    coordinates in the mesh.  The validator checks branch references,
    duplicate undirected node pairs, orientation consistency, positive lengths,
    and expected structured-mesh branch counts.
    """

    unique_r = sorted({node.r for node in mesh.nodes.values()})
    unique_z = sorted({node.z for node in mesh.nodes.values()})
    nr = len(unique_r) - 1
    nz = len(unique_z) - 1
    if nr <= 0 or nz <= 0:
        raise ValueError("structured mesh must contain at least one cell in r and z.")

    expected_nodes = (nr + 1) * (nz + 1)
    expected_cells = nr * nz
    if mesh.number_of_nodes != expected_nodes:
        raise ValueError(
            f"node count {mesh.number_of_nodes} does not match structured dimensions: {expected_nodes}."
        )
    if mesh.number_of_cells != expected_cells:
        raise ValueError(
            f"cell count {mesh.number_of_cells} does not match structured dimensions: {expected_cells}."
        )

    seen_pairs: set[frozenset[int]] = set()
    for branch in mesh.branches.values():
        if branch.start_node_id not in mesh.nodes:
            raise ValueError(f"branch {branch.id} references missing start node {branch.start_node_id}.")
        if branch.end_node_id not in mesh.nodes:
            raise ValueError(f"branch {branch.id} references missing end node {branch.end_node_id}.")
        if branch.start_node_id == branch.end_node_id:
            raise ValueError(f"branch {branch.id} connects a node to itself.")
        if branch.length <= 0.0:
            raise ValueError(f"branch {branch.id} has non-positive length.")

        pair = frozenset((branch.start_node_id, branch.end_node_id))
        if pair in seen_pairs:
            raise ValueError(f"duplicate undirected node pair for branch {branch.id}.")
        seen_pairs.add(pair)

        start = mesh.get_node(branch.start_node_id)
        end = mesh.get_node(branch.end_node_id)
        if branch.orientation is BranchOrientation.RADIAL:
            if not isclose(start.z, end.z):
                raise ValueError(f"radial branch {branch.id} must connect nodes with equal z.")
            if not end.r > start.r:
                raise ValueError(f"radial branch {branch.id} must point from smaller r to larger r.")
        elif branch.orientation is BranchOrientation.AXIAL:
            if not isclose(start.r, end.r):
                raise ValueError(f"axial branch {branch.id} must connect nodes with equal r.")
            if not end.z > start.z:
                raise ValueError(f"axial branch {branch.id} must point from smaller z to larger z.")
        elif branch.orientation is BranchOrientation.CIRCUMFERENTIAL:
            if not isclose(start.z, end.z):
                raise ValueError(f"circumferential branch {branch.id} must connect nodes with equal z.")
            # Periodic phi-z seam branches may wrap from high phi to low phi;
            # equal phi would not connect adjacent circumferential cells.
            if isclose(start.r, end.r):
                raise ValueError(f"circumferential branch {branch.id} must connect distinct phi positions.")
        else:
            raise ValueError(f"branch {branch.id} has unsupported orientation {branch.orientation}.")

    if mesh.number_of_circumferential_branches:
        return
    expected_radial = nr * (nz + 1)
    expected_axial = (nr + 1) * nz
    if mesh.number_of_radial_branches != expected_radial:
        raise ValueError(
            f"radial branch count {mesh.number_of_radial_branches} does not match expected {expected_radial}."
        )
    if mesh.number_of_axial_branches != expected_axial:
        raise ValueError(
            f"axial branch count {mesh.number_of_axial_branches} does not match expected {expected_axial}."
        )
