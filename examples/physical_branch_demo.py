"""Demonstrate physical branch assembly and magnetic field recovery."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mesh import generate_cell_centered_axisymmetric_mesh
from src.mrn import MU0, build_physical_branch_model
from src.solver import build_magnetic_excitation, solve_physical_linear_mrn


def main() -> None:
    mesh = generate_cell_centered_axisymmetric_mesh([1.0, 2.0, 3.0], [0.0, 0.5, 1.0], material_ids=[[0, 1], [2, 0]])
    materials = {0: MU0, 1: {"relative_permeability": 2000.0}, 2: {"relative_permeability": 5000.0}}
    excitation = build_magnetic_excitation(mesh, branch_mmf_by_id={0: 25.0, 3: 10.0})
    solution = solve_physical_linear_mrn(mesh, materials, excitation)
    physical = build_physical_branch_model(mesh, materials)

    print("branch_id orientation cells materials reluctance permeance source_mmf potential_drop net_mmf flux max_abs_B max_abs_H")
    max_constitutive_error = 0.0
    max_segment_balance_error = 0.0
    for branch_id, state in solution.branches.items():
        branch = mesh.get_branch(branch_id)
        phys = physical[branch_id]
        cells = mesh.branch_id_to_cell_ids[branch_id]
        max_constitutive_error = max(max_constitutive_error, abs(state.flux - state.equivalent_permeance * state.net_mmf))
        max_segment_balance_error = max(max_segment_balance_error, abs(state.total_segment_mmf_drop - state.net_mmf))
        print(
            branch_id,
            branch.orientation.name,
            cells,
            phys.material_ids,
            f"{phys.reluctance:.6e}",
            f"{phys.permeance:.6e}",
            f"{state.source_mmf:.6e}",
            f"{state.potential_drop:.6e}",
            f"{state.net_mmf:.6e}",
            f"{state.flux:.6e}",
            f"{state.max_abs_B:.6e}",
            f"{state.max_abs_H:.6e}",
        )

    interface_id = next(branch_id for branch_id, branch in physical.items() if branch.is_material_interface)
    print(f"\nsegments for material-interface branch {interface_id}")
    print("segment_index cell_id material_id geometry_kind length area_or_effective_area reluctance flux B_diagnostic H_diagnostic mmf_drop")
    for segment, state in zip(physical[interface_id].segments, solution.branches[interface_id].segment_states):
        print(
            segment.segment_index,
            segment.cell_id,
            segment.material_id,
            segment.geometry_kind.name,
            f"{segment.length:.6e}",
            f"{segment.effective_area:.6e}",
            f"{segment.reluctance:.6e}",
            f"{state.flux:.6e}",
            f"{state.flux_density_average:.6e}",
            f"{state.magnetic_field_strength_average:.6e}",
            f"{state.mmf_drop:.6e}",
        )

    print(f"\nmaximum nodal residual {solution.linear_solution.max_abs_residual:.6e}")
    print(f"maximum branch constitutive error {max_constitutive_error:.6e}")
    print(f"maximum segment MMF-balance error {max_segment_balance_error:.6e}")


if __name__ == "__main__":
    main()
