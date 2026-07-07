"""Mesh cell primitives for UniversalMRN."""

from __future__ import annotations

from dataclasses import dataclass


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
