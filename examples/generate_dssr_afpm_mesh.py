"""Generate DSSR AFPM mesh diagnostic figures.

The script writes runtime-only PNG files under ``figures/``.  Those files are
ignored by Git so the repository keeps source code without generated outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Wedge


@dataclass(frozen=True, slots=True)
class DssrAfpmConfig:
    """Simple double-sided slotted-rotor AFPM plotting configuration."""

    inner_radius: float = 0.35
    outer_radius: float = 1.0
    slot_count: int = 12
    radial_layers: int = 4


def _material_for_sector(sector: int) -> int:
    """Return a repeating material id for a circumferential sector."""

    return sector % 3


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


if __name__ == "__main__":
    saved_files = save_dssr_afpm_figures()
    for saved_file in saved_files:
        print(saved_file)
