"""Common magnetic-material typing helpers."""
from __future__ import annotations

from typing import Protocol

from src.mrn.interface_reluctance import MU0 as MU_0


class MagneticMaterial(Protocol):
    """Protocol for linear magnetic materials used by physical MRN builders."""

    id: int
    name: str
    relative_permeability: float

    @property
    def permeability(self) -> float: ...

    @property
    def is_permanent_magnet(self) -> bool: ...
