"""Solve a 2 x 2 cell-centered MRN with corrected branch permeances."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mesh import generate_cell_centered_axisymmetric_mesh
from src.mrn import calculate_cell_centered_mesh_branch_permeances
from src.solver import build_magnetic_excitation, solve_linear_mrn


def main() -> None:
    mesh = generate_cell_centered_axisymmetric_mesh([1.0, 2.0, 3.0], [0.0, 1.0, 2.0], material_ids=[[0, 1], [0, 1]])
    materials = {0: {"relative_permeability": 1.0}, 1: {"relative_permeability": 1000.0}}
    permeances = calculate_cell_centered_mesh_branch_permeances(mesh, materials)
    excitation = build_magnetic_excitation(mesh, branch_mmf_by_id={0: 10.0})
    solution = solve_linear_mrn(mesh, permeances, excitation)
    print(f"node potentials: {solution.node_potential.tolist()}")
    print(f"branch fluxes: {solution.branch_flux.tolist()}")
    print(f"max residual: {solution.max_abs_residual}")


if __name__ == "__main__":
    main()
