"""Demonstrate automatic permanent-magnet branch excitation."""
from __future__ import annotations

from math import pi
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.material import LinearMagneticMaterial, LinearPermanentMagnetMaterial, MagnetizationAxis
from src.mesh import BranchOrientation, generate_cell_centered_axisymmetric_mesh
from src.mrn import build_permanent_magnet_assignments, build_permanent_magnet_branch_sources, build_physical_branch_model
from src.solver import solve_permanent_magnet_linear_mrn


def main() -> None:
    ferrite = LinearPermanentMagnetMaterial(2, "Ferrite PM", 1.05, 0.45)
    materials = {
        0: LinearMagneticMaterial(0, "Air", 1.0),
        1: LinearMagneticMaterial(1, "Steel", 1000.0),
        2: ferrite,
    }
    mesh = generate_cell_centered_axisymmetric_mesh([0.02, 0.04, 0.06], [0.0, 0.01, 0.02], material_ids=[[2, 0], [2, 1]])
    assignments = build_permanent_magnet_assignments(
        mesh,
        {0: MagnetizationAxis.RADIAL_POSITIVE, 2: MagnetizationAxis.RADIAL_NEGATIVE},
        materials=materials,
        strict=True,
    )
    physical = build_physical_branch_model(mesh, materials, angular_span=2 * pi)
    sources = build_permanent_magnet_branch_sources(mesh, physical, materials, assignments)
    solution = solve_permanent_magnet_linear_mrn(mesh, materials, assignments)

    print("Material:")
    print(f"  Br = {ferrite.remanence:.6g} T")
    print(f"  mu_r = {ferrite.relative_permeability:.6g}")
    print(f"  calculated H_c = {ferrite.coercive_field_strength:.6g} A/m")
    print("Branches:")
    for bid, branch in sorted(mesh.branches.items()):
        state = solution.branches[bid]
        src = sources[bid]
        print(
            f"  branch {bid} {branch.orientation.name}: PM segments={len(src.segment_sources)}, "
            f"PM MMF={src.total_mmf:.6g} A, drop={state.potential_drop:.6g} A, "
            f"flux={state.flux:.6g} Wb, max|B|={state.max_abs_B:.6g} T, max|H|={state.max_abs_H:.6g} A/m"
        )
        for seg in src.segment_sources:
            print(
                f"    PM cell {seg.cell_id}: projection={seg.magnetization_projection:.6g}, "
                f"length={seg.magnetized_length:.6g} m, source={seg.mmf:.6g} A"
            )
    print(f"maximum nodal residual = {solution.linear_solution.max_abs_residual:.6g}")
    print("net PM source distribution =", {bid: s.total_mmf for bid, s in sources.items()})
    axial_zero = all(abs(sources[bid].total_mmf) < 1e-12 for bid, b in mesh.branches.items() if b.orientation is BranchOrientation.AXIAL)
    print(f"radial magnets give zero MMF on axial branches: {axial_zero}")


if __name__ == "__main__":
    main()
