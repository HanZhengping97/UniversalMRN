"""Core mesh primitives for UniversalMRN.

This module defines lightweight, typed data containers for describing a
structured two-dimensional mesh.  It intentionally contains no magnetic
calculations; numerical physics routines will be introduced in later
milestones.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Node:
    """A mesh node in two-dimensional Cartesian space.

    Attributes:
        id: Unique integer identifier for the node.
        x: Node x-coordinate.
        y: Node y-coordinate.
    """

    id: int
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class Cell:
    """A rectangular mesh cell.

    Attributes:
        id: Unique integer identifier for the cell.
        center_x: Cell-center x-coordinate.
        center_y: Cell-center y-coordinate.
        dx: Cell width in the x-direction.
        dy: Cell height in the y-direction.
        material_id: Identifier of the material assigned to the cell.
    """

    id: int
    center_x: float
    center_y: float
    dx: float
    dy: float
    material_id: int


@dataclass(slots=True)
class Mesh:
    """Container for mesh nodes and cells.

    Nodes and cells are stored by their integer identifiers for efficient
    lookup and to keep the mesh representation independent of insertion order.
    """

    nodes: dict[int, Node] = field(default_factory=dict)
    cells: dict[int, Cell] = field(default_factory=dict)

    def add_node(self, node: Node) -> None:
        """Add a node to the mesh.

        Args:
            node: Node instance to insert.

        Raises:
            ValueError: If a node with the same identifier already exists.
        """

        if node.id in self.nodes:
            msg = f"Node with id {node.id} already exists."
            raise ValueError(msg)
        self.nodes[node.id] = node

    def add_cell(self, cell: Cell) -> None:
        """Add a cell to the mesh.

        Args:
            cell: Cell instance to insert.

        Raises:
            ValueError: If a cell with the same identifier already exists.
        """

        if cell.id in self.cells:
            msg = f"Cell with id {cell.id} already exists."
            raise ValueError(msg)
        self.cells[cell.id] = cell

    def get_node(self, node_id: int) -> Node:
        """Return a node by identifier.

        Args:
            node_id: Identifier of the requested node.

        Returns:
            The requested Node instance.

        Raises:
            KeyError: If no node exists for ``node_id``.
        """

        return self.nodes[node_id]

    def get_cell(self, cell_id: int) -> Cell:
        """Return a cell by identifier.

        Args:
            cell_id: Identifier of the requested cell.

        Returns:
            The requested Cell instance.

        Raises:
            KeyError: If no cell exists for ``cell_id``.
        """

        return self.cells[cell_id]


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

        return mesh
