"""Demonstrate series material-interface reluctance on a cell-centered mesh."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mesh import BranchOrientation, generate_cell_centered_axisymmetric_mesh
from src.mrn import MU0, calculate_branch_reluctance_segments, calculate_cell_centered_branch_permeance, distance_weighted_harmonic_permeability, harmonic_mean_permeability


def main() -> None:
    mesh = generate_cell_centered_axisymmetric_mesh([1.0, 2.0, 3.0], [0.0, 1.0, 2.0], material_ids=[[0, 1], [0, 1]])
    materials = {0: {"relative_permeability": 1.0}, 1: {"relative_permeability": 1000.0}}
    branch = next(b for b in mesh.branches.values() if b.orientation is BranchOrientation.RADIAL)
    segments = calculate_branch_reluctance_segments(mesh, branch, materials)
    permeance = calculate_cell_centered_branch_permeance(mesh, branch, materials)
    total = sum(s.reluctance for s in segments)
    mu_r_weighted = distance_weighted_harmonic_permeability(segments[0].permeability / MU0, segments[1].permeability / MU0, segments[0].length, segments[1].length)
    print(f"branch ID: {branch.id}")
    print(f"orientation: {branch.orientation.name.lower()}")
    print(f"material IDs: {[s.material_id for s in segments]}")
    print(f"segment lengths: {[s.length for s in segments]}")
    print(f"segment reluctances: {[s.reluctance for s in segments]}")
    print(f"total reluctance: {total}")
    print(f"equivalent permeance: {permeance}")
    print(f"distance-weighted equivalent relative permeability: {mu_r_weighted:.6f}")
    print(f"arithmetic mean relative permeability for comparison: {(1.0 + 1000.0) / 2.0:.6f}")
    print(f"harmonic mean relative permeability: {harmonic_mean_permeability(1.0, 1000.0):.6f}")


if __name__ == "__main__":
    main()
