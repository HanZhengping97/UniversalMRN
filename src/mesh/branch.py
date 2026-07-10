"""Magnetic-network branch primitives for structured r-z meshes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class BranchOrientation(Enum):
    """Allowed orientations for adjacent-node magnetic-network branches."""

    RADIAL = auto()
    AXIAL = auto()


@dataclass(frozen=True, slots=True)
class Branch:
    """Directed connection between two adjacent structured-mesh nodes.

    In the DSSR AFPM r-z model, radial branches point from smaller r to larger
    r, and axial branches point from smaller z to larger z.  The branch stores
    topology and geometry only; reluctance and permeability are intentionally
    deferred to a later development phase.
    """

    id: int
    start_node_id: int
    end_node_id: int
    orientation: BranchOrientation
    length: float
    center_r: float
    center_z: float

    def __post_init__(self) -> None:
        if self.id < 0:
            raise ValueError("branch id must be non-negative.")
        if self.start_node_id == self.end_node_id:
            raise ValueError("branch start and end nodes must be different.")
        if self.length <= 0.0:
            raise ValueError("branch length must be positive.")
