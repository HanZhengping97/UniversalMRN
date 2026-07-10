"""Mesh node primitives for UniversalMRN."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Node:
    """A mesh node in two-dimensional space.

    The DSSR AFPM mesh uses an r-z coordinate system while preserving the
    original Cartesian-style public fields: ``x`` corresponds to ``r`` and
    ``y`` corresponds to ``z``.

    Attributes:
        id: Unique integer identifier for the node.
        x: Node x-coordinate, equivalent to radial coordinate r.
        y: Node y-coordinate, equivalent to axial coordinate z.
    """

    id: int
    x: float
    y: float

    @property
    def r(self) -> float:
        """Radial coordinate alias for ``x``."""

        return self.x

    @property
    def z(self) -> float:
        """Axial coordinate alias for ``y``."""

        return self.y
