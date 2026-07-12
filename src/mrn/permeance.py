"""Branch permeance matrix framework.

The solver-facing permeance matrix builder accepts already assembled branch
permeances.  Older structured meshes are vertex-centered; any geometry helper
that sums cell contributions on those edge branches should be interpreted only
as a vertex-centered edge dual-area approximation.  Such parallel contribution
summation is valid only for geometrically parallel paths under the same MMF
drop, and is not the material-interface model for two materials encountered
sequentially between neighboring control-volume centers.  Use
``src.mrn.interface_reluctance`` with the cell-centered mesh generator for the
preferred control-volume material-interface formulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping

import numpy as np
from scipy import sparse

from src.mesh import Mesh
from src.topology import TopologyIndex


@dataclass(frozen=True, slots=True)
class BranchPermeance:
    """Validated placeholder permeance assigned to one branch."""

    branch_id: int
    permeance: float

    def __post_init__(self) -> None:
        if self.branch_id < 0:
            raise ValueError("branch_id must be non-negative.")
        if not isfinite(self.permeance) or self.permeance <= 0.0:
            raise ValueError("permeance must be positive and finite.")


def build_branch_permeance_matrix(
    mesh: Mesh, permeances: Mapping[int, float], index: TopologyIndex
) -> sparse.csr_matrix:
    """Build diagonal sparse ``Gb`` ordered by ``index.column_to_branch_id``."""

    mesh_branch_ids = set(mesh.branches)
    provided_branch_ids = set(permeances)
    missing = mesh_branch_ids - provided_branch_ids
    unknown = provided_branch_ids - mesh_branch_ids
    if missing:
        raise ValueError(f"missing permeance values for branch IDs: {tuple(sorted(missing))}.")
    if unknown:
        raise ValueError(f"unknown permeance branch IDs: {tuple(sorted(unknown))}.")
    if set(index.column_to_branch_id) != mesh_branch_ids:
        raise ValueError("topology index branch IDs do not match mesh branches.")

    diagonal = np.array([BranchPermeance(branch_id, float(permeances[branch_id])).permeance for branch_id in index.column_to_branch_id])
    return sparse.diags(diagonal, offsets=0, shape=(mesh.number_of_branches, mesh.number_of_branches), format="csr")
