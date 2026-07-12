"""Demonstrate a linear magnetic scalar-potential solve on a 1 x 1 mesh."""

from __future__ import annotations

from math import pi
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mesh import MeshGenerator  # noqa: E402
from src.solver import build_magnetic_excitation, solve_linear_mrn  # noqa: E402


def main() -> None:
    mesh = MeshGenerator.generate_rectangular_mesh(nx=1, ny=1, width=0.02, height=0.01, material_id=1)
    mu0 = 4.0e-7 * pi
    branch_permeances = {
        branch_id: mu0 * max(branch.center_r, 0.01) / branch.length
        for branch_id, branch in mesh.branches.items()
    }
    excitation = build_magnetic_excitation(mesh, branch_mmf_by_id={0: 10.0})
    solution = solve_linear_mrn(mesh, branch_permeances, excitation)

    print("Linear magnetic solver demo")
    print(f"node count: {solution.number_of_nodes}")
    print(f"branch count: {solution.number_of_branches}")
    print(f"reference node: {solution.reference_node_id}")
    print("nodal potentials [A-turn]:", npformat(solution.node_potential))
    print("branch MMF [A-turn]:", npformat(solution.branch_mmf))
    print("branch potential drops [A-turn]:", npformat(solution.branch_potential_drop))
    print("branch fluxes [Wb]:", npformat(solution.branch_flux))
    print(f"maximum nodal residual [Wb]: {solution.max_abs_residual:.6e}")


def npformat(values) -> str:
    return "[" + ", ".join(f"{float(value): .6e}" for value in values) + "]"


if __name__ == "__main__":
    main()
