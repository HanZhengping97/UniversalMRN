"""Mesh container for UniversalMRN."""

from __future__ import annotations

from dataclasses import dataclass, field

from .branch import Branch, BranchOrientation
from .cell import Cell
from .node import Node


@dataclass(slots=True)
class Mesh:
    """Container for mesh nodes, cells, and magnetic-network branches.

    Nodes, cells, and branches are stored by their integer identifiers for
    efficient lookup and to keep the mesh representation independent of
    insertion order.
    """

    nodes: dict[int, Node] = field(default_factory=dict)
    cells: dict[int, Cell] = field(default_factory=dict)
    branches: dict[int, Branch] = field(default_factory=dict)

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

    def add_branch(self, branch: Branch) -> None:
        """Add a branch to the mesh.

        Raises:
            ValueError: If a branch with the same identifier already exists.
        """

        if branch.id in self.branches:
            msg = f"Branch with id {branch.id} already exists."
            raise ValueError(msg)
        self.branches[branch.id] = branch

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

    def get_branch(self, branch_id: int) -> Branch:
        """Return a branch by identifier."""

        return self.branches[branch_id]

    @property
    def number_of_nodes(self) -> int:
        """Number of nodes in the mesh."""

        return len(self.nodes)

    @property
    def number_of_cells(self) -> int:
        """Number of cells in the mesh."""

        return len(self.cells)

    @property
    def number_of_branches(self) -> int:
        """Number of branches in the mesh."""

        return len(self.branches)

    @property
    def number_of_radial_branches(self) -> int:
        """Number of radial branches in the mesh."""

        return sum(1 for branch in self.branches.values() if branch.orientation is BranchOrientation.RADIAL)

    @property
    def number_of_axial_branches(self) -> int:
        """Number of axial branches in the mesh."""

        return sum(1 for branch in self.branches.values() if branch.orientation is BranchOrientation.AXIAL)
