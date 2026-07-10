"""Generate DSSR AFPM mesh diagnostics, topology CSV, and figures.

The script writes runtime-only PNG files under ``figures/`` and a branch CSV
under ``output/``.  Those files are ignored by Git so the repository keeps
source code without generated outputs.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

from src.export import export_branches_csv
from src.mesh import Branch, Cell, Mesh, MeshGenerator, validate_branch_topology


@dataclass(frozen=True, slots=True)
class DssrAfpmConfig:
    """Simple double-sided slotted-rotor AFPM plotting and mesh configuration."""

    inner_radius: float = 0.35
    outer_radius: float = 1.0
    axial_length: float = 0.20
    slot_count: int = 12
    radial_layers: int = 4
    axial_layers: int = 3


def _material_for_sector(sector: int) -> int:
    """Return a repeating material id for a circumferential sector."""

    return sector % 3


def _classify_mesh_materials(mesh: Mesh, config: DssrAfpmConfig) -> None:
    """Assign deterministic placeholder material ids to structured cells."""

    classified_cells: dict[int, Cell] = {}
    for cell_id, cell in mesh.cells.items():
        radial_fraction = cell.center_x / config.outer_radius
        material_id = min(2, int(radial_fraction * 3.0))
        classified_cells[cell_id] = Cell(
            id=cell.id,
            center_x=cell.center_x,
            center_y=cell.center_y,
            dx=cell.dx,
            dy=cell.dy,
            material_id=material_id,
        )
    mesh.cells.update(classified_cells)


def generate_dssr_afpm_mesh(config: DssrAfpmConfig = DssrAfpmConfig()) -> Mesh:
    """Generate a structured r-z DSSR AFPM diagnostic mesh with branches."""

    mesh = MeshGenerator.generate_rectangular_mesh(
        nx=config.radial_layers,
        ny=config.axial_layers,
        width=config.outer_radius - config.inner_radius,
        height=config.axial_length,
    )
    shifted_nodes = {
        node_id: type(node)(id=node.id, x=node.x + config.inner_radius, y=node.y)
        for node_id, node in mesh.nodes.items()
    }
    shifted_cells = {
        cell_id: Cell(
            id=cell.id,
            center_x=cell.center_x + config.inner_radius,
            center_y=cell.center_y,
            dx=cell.dx,
            dy=cell.dy,
            material_id=cell.material_id,
        )
        for cell_id, cell in mesh.cells.items()
    }
    shifted_branches = {
        branch_id: Branch(
            id=branch.id,
            start_node_id=branch.start_node_id,
            end_node_id=branch.end_node_id,
            orientation=branch.orientation,
            length=branch.length,
            center_r=branch.center_r + config.inner_radius,
            center_z=branch.center_z,
        )
        for branch_id, branch in mesh.branches.items()
    }
    mesh.nodes.update(shifted_nodes)
    mesh.cells.update(shifted_cells)
    mesh.branches.update(shifted_branches)
    _classify_mesh_materials(mesh, config)
    validate_branch_topology(mesh)
    return mesh


def _draw_annulus(axis, config: DssrAfpmConfig, *, show_mesh: bool, color_by_material: bool) -> None:
    """Draw an annular DSSR AFPM sector layout."""

    colors = ("#729fcf", "#fcaf3e", "#8ae234")
    sector_angle = 360.0 / config.slot_count
    radial_step = (config.outer_radius - config.inner_radius) / config.radial_layers

    for sector in range(config.slot_count):
        theta1 = sector * sector_angle
        theta2 = theta1 + sector_angle
        material_id = _material_for_sector(sector)
        facecolor = colors[material_id] if color_by_material else "#eeeeec"
        for layer in range(config.radial_layers):
            radius = config.inner_radius + (layer + 1) * radial_step
            width = radial_step
            patch = Wedge(
                (0.0, 0.0),
                radius,
                theta1,
                theta2,
                width=width,
                facecolor=facecolor,
                edgecolor="#2e3436" if show_mesh else "white",
                linewidth=0.8 if show_mesh else 0.3,
                alpha=0.9,
            )
            axis.add_patch(patch)

    axis.add_patch(plt.Circle((0.0, 0.0), config.inner_radius, color="white", zorder=3))
    axis.set_aspect("equal")
    axis.set_xlim(-1.08 * config.outer_radius, 1.08 * config.outer_radius)
    axis.set_ylim(-1.08 * config.outer_radius, 1.08 * config.outer_radius)
    axis.axis("off")


def save_dssr_afpm_figures(figures_dir: Path = Path("figures")) -> tuple[Path, Path, Path]:
    """Save DSSR AFPM geometry, mesh, and material-map figures locally."""

    figures_dir.mkdir(parents=True, exist_ok=True)
    config = DssrAfpmConfig()
    outputs = (
        figures_dir / "dssr_geometry.png",
        figures_dir / "dssr_mesh.png",
        figures_dir / "dssr_material_map.png",
    )

    figure_specs = (
        ("DSSR AFPM geometry", False, False),
        ("DSSR AFPM mesh", True, False),
        ("DSSR AFPM material map", True, True),
    )
    for output, (title, show_mesh, color_by_material) in zip(outputs, figure_specs, strict=True):
        figure, axis = plt.subplots(figsize=(6.0, 6.0), dpi=160)
        _draw_annulus(axis, config, show_mesh=show_mesh, color_by_material=color_by_material)
        axis.set_title(title)
        figure.savefig(output, bbox_inches="tight")
        plt.close(figure)

    return outputs


def print_mesh_summary(mesh: Mesh, config: DssrAfpmConfig) -> None:
    """Print mesh, branch, and material diagnostics."""

    expected_radial = config.radial_layers * (config.axial_layers + 1)
    expected_axial = (config.radial_layers + 1) * config.axial_layers
    print("DSSR AFPM non-parametric mesh")
    print(f"Number of nodes: {mesh.number_of_nodes}")
    print(f"Number of cells: {mesh.number_of_cells}")
    print(f"Number of branches: {mesh.number_of_branches}")
    print(f"Number of radial branches: {mesh.number_of_radial_branches}")
    print(f"Number of axial branches: {mesh.number_of_axial_branches}")
    print(f"Expected radial branches: {expected_radial}")
    print(f"Expected axial branches: {expected_axial}")
    validate_branch_topology(mesh)
    print("Topology validation: passed")
    print("Material statistics:")
    for material_id, count in sorted(Counter(cell.material_id for cell in mesh.cells.values()).items()):
        print(f"  material {material_id}: {count} cells")


if __name__ == "__main__":
    config = DssrAfpmConfig()
    dssr_mesh = generate_dssr_afpm_mesh(config)
    print_mesh_summary(dssr_mesh, config)
    branches_csv = export_branches_csv(dssr_mesh, Path("output/branches.csv"))
    print(f"Branch CSV: {branches_csv}")
    for saved_file in save_dssr_afpm_figures():
        print(saved_file)
