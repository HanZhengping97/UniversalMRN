"""Topology utilities for MRN graph assembly."""

from .connectivity import find_connected_components, validate_connected_mesh
from .incidence import TopologyIndex, build_incidence_matrix, build_topology_index, validate_incidence_matrix

__all__ = [
    "TopologyIndex",
    "build_incidence_matrix",
    "build_topology_index",
    "find_connected_components",
    "validate_connected_mesh",
    "validate_incidence_matrix",
]
