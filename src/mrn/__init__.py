"""Magnetic reluctance network matrix assembly helpers."""

from .nodal_matrix import apply_reference_node, build_nodal_permeance_matrix, remove_reference_entry
from .permeance import BranchPermeance, build_branch_permeance_matrix
from .interface_reluctance import (
    MU0,
    ReluctanceSegment,
    calculate_branch_reluctance_segments,
    calculate_cell_centered_branch_permeance,
    calculate_cell_centered_branch_reluctance,
    calculate_cell_centered_mesh_branch_permeances,
    distance_weighted_harmonic_permeability,
    harmonic_mean_permeability,
)

__all__ = [
    "BranchPermeance",
    "MU0",
    "ReluctanceSegment",
    "calculate_branch_reluctance_segments",
    "calculate_cell_centered_branch_permeance",
    "calculate_cell_centered_branch_reluctance",
    "calculate_cell_centered_mesh_branch_permeances",
    "distance_weighted_harmonic_permeability",
    "harmonic_mean_permeability",
    "apply_reference_node",
    "build_branch_permeance_matrix",
    "build_nodal_permeance_matrix",
    "remove_reference_entry",
    "SegmentGeometryKind",
    "PhysicalBranchSegment",
    "PhysicalBranch",
    "build_physical_branch",
    "build_physical_branch_model",
]

from .physical_branch import (
    SegmentGeometryKind,
    PhysicalBranchSegment,
    PhysicalBranch,
    build_physical_branch,
    build_physical_branch_model,
)
