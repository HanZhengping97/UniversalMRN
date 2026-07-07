"""Mesh node primitives for UniversalMRN."""

from __future__ import annotations

from dataclasses import dataclass


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
