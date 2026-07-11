"""Magnetic reluctance network matrix assembly helpers."""

from .geometry_permeance import (
    calculate_branch_permeance,
    calculate_mesh_branch_permeances,
    find_adjacent_cells,
)
from .nodal_matrix import apply_reference_node, build_nodal_permeance_matrix, remove_reference_entry
from .permeance import BranchPermeance, build_branch_permeance_matrix

__all__ = [
    "BranchPermeance",
    "apply_reference_node",
    "build_branch_permeance_matrix",
    "build_nodal_permeance_matrix",
    "calculate_branch_permeance",
    "calculate_mesh_branch_permeances",
    "find_adjacent_cells",
    "remove_reference_entry",
]
