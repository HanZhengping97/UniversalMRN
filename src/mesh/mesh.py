"""Mesh container for UniversalMRN."""

from __future__ import annotations

from dataclasses import dataclass, field

from .cell import Cell
from .node import Node


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
