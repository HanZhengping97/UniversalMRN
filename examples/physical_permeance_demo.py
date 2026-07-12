"""Assemble physical branch and nodal permeance matrices for the DSSR mesh."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.generate_dssr_afpm_mesh import DssrAfpmConfig, generate_dssr_afpm_mesh
from src.material import LinearMagneticMaterial
from src.mrn import (
    build_branch_permeance_matrix,
    build_nodal_permeance_matrix,
    calculate_mesh_branch_permeances,
)
from src.topology import build_incidence_matrix, build_topology_index


def main() -> None:
    mesh = generate_dssr_afpm_mesh(DssrAfpmConfig())
    materials = {
        0: LinearMagneticMaterial(0, "Air-like region", 1.0),
        1: LinearMagneticMaterial(1, "Ferrite-linear", 1.05),
        2: LinearMagneticMaterial(2, "Electrical steel-linear", 1000.0),
    }

    topology = build_topology_index(mesh)
    incidence = build_incidence_matrix(mesh)
    permeances = calculate_mesh_branch_permeances(mesh, materials)
    branch_matrix = build_branch_permeance_matrix(mesh, permeances, topology)
    nodal_matrix = build_nodal_permeance_matrix(incidence, branch_matrix)

    print("Physical DSSR AFPM MRN assembly")
    print(f"branches: {mesh.number_of_branches}")
    print(f"minimum permeance: {min(permeances.values()):.6e} H")
    print(f"maximum permeance: {max(permeances.values()):.6e} H")
    print(f"Gb shape: {branch_matrix.shape}")
    print(f"Gn shape: {nodal_matrix.shape}")
    print("Note: permanent-magnet MMF and nonlinear steel are later phases.")


if __name__ == "__main__":
    main()
