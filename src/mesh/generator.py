"""Mesh generation utilities for UniversalMRN."""

from __future__ import annotations

from .branch import Branch, BranchOrientation
from .cell import Cell
from .mesh import Mesh
from .node import Node


class MeshGenerator:
    """Factory for generating common mesh topologies."""

    @staticmethod
    def generate_rectangular_mesh(
        nx: int,
        ny: int,
        width: float,
        height: float,
        *,
        material_id: int = 0,
    ) -> Mesh:
        """Generate a uniform rectangular mesh.

        The generated mesh spans ``[0, width]`` in x and ``[0, height]`` in y.
        It contains ``(nx + 1) * (ny + 1)`` nodes and ``nx * ny`` rectangular
        cells.  Cell identifiers increase first along x, then along y.

        Args:
            nx: Number of cells in the x-direction.
            ny: Number of cells in the y-direction.
            width: Total mesh width.
            height: Total mesh height.
            material_id: Material identifier assigned to every generated cell.

        Returns:
            A Mesh populated with uniformly spaced nodes and cells.

        Raises:
            ValueError: If ``nx`` or ``ny`` is not positive, or if ``width`` or
                ``height`` is not positive.
        """

        if nx <= 0:
            raise ValueError("nx must be positive.")
        if ny <= 0:
            raise ValueError("ny must be positive.")
        if width <= 0.0:
            raise ValueError("width must be positive.")
        if height <= 0.0:
            raise ValueError("height must be positive.")

        mesh = Mesh()
        dx = width / nx
        dy = height / ny

        node_id = 0
        for j in range(ny + 1):
            y = j * dy
            for i in range(nx + 1):
                x = i * dx
                mesh.add_node(Node(id=node_id, x=x, y=y))
                node_id += 1

        cell_id = 0
        for j in range(ny):
            center_y = (j + 0.5) * dy
            for i in range(nx):
                center_x = (i + 0.5) * dx
                mesh.add_cell(
                    Cell(
                        id=cell_id,
                        center_x=center_x,
                        center_y=center_y,
                        dx=dx,
                        dy=dy,
                        material_id=material_id,
                    )
                )
                cell_id += 1

        MeshGenerator.generate_structured_branches(mesh, nx=nx, ny=ny)
        return mesh

    @staticmethod
    def generate_structured_branches(mesh: Mesh, nx: int, ny: int) -> None:
        """Generate directed adjacent-node branches for a structured mesh.

        Numbering convention: all radial branches are generated first, then all
        axial branches.  Radial branch ids increase with radial index ``i``
        inside each axial row ``j``; axial branch ids increase with radial index
        ``i`` inside each axial interval ``j``.  Node ids are assumed to follow
        the structured convention ``node_id = j * (nx + 1) + i``.
        """

        if nx <= 0:
            raise ValueError("nx must be positive.")
        if ny <= 0:
            raise ValueError("ny must be positive.")
        if mesh.branches:
            raise ValueError("mesh already contains branches.")

        branch_id = 0
        for j in range(ny + 1):
            for i in range(nx):
                start = mesh.get_node(j * (nx + 1) + i)
                end = mesh.get_node(j * (nx + 1) + i + 1)
                mesh.add_branch(
                    Branch(
                        id=branch_id,
                        start_node_id=start.id,
                        end_node_id=end.id,
                        orientation=BranchOrientation.RADIAL,
                        length=end.r - start.r,
                        center_r=0.5 * (start.r + end.r),
                        center_z=start.z,
                    )
                )
                branch_id += 1

        for j in range(ny):
            for i in range(nx + 1):
                start = mesh.get_node(j * (nx + 1) + i)
                end = mesh.get_node((j + 1) * (nx + 1) + i)
                mesh.add_branch(
                    Branch(
                        id=branch_id,
                        start_node_id=start.id,
                        end_node_id=end.id,
                        orientation=BranchOrientation.AXIAL,
                        length=end.z - start.z,
                        center_r=start.r,
                        center_z=0.5 * (start.z + end.z),
                    )
                )
                branch_id += 1
