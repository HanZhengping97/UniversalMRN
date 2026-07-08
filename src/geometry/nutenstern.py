"""Matplotlib renderer for German-engineering-style Nutenstern figures.

The renderer intentionally contains no coordinate, angle, or phase-selection
mathematics. It only draws the precomputed data supplied by ``geometry.py`` and
``phase_selection.py``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyArrowPatch

from .geometry import NutensternFigureData, build_nutenstern_figure_data
from .phase_selection import phase_styles


_OUTPUT_FORMATS = ("svg", "pdf", "png")


def _apply_german_engineering_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def _draw_reference_circle(axis, figure_data: NutensternFigureData) -> None:
    circle = figure_data.reference_circle
    axis.add_patch(
        Circle(
            (circle.center.x, circle.center.y),
            circle.radius,
            fill=False,
            linestyle="--",
            linewidth=0.9,
            edgecolor=circle.color,
        )
    )
    axis.text(
        circle.label_position.x,
        circle.label_position.y,
        circle.label,
        color=circle.color,
        ha="center",
        va="center",
    )


def _draw_angle_guides(axis, figure_data: NutensternFigureData) -> None:
    for guide in figure_data.angle_guides:
        axis.add_patch(
            Arc(
                (guide.center.x, guide.center.y),
                guide.width,
                guide.height,
                theta1=guide.theta1,
                theta2=guide.theta2,
                linewidth=0.85,
                color=guide.color,
            )
        )
        axis.text(
            guide.label_position.x,
            guide.label_position.y,
            guide.label,
            color=guide.color,
            ha="center",
            va="center",
            fontsize=9,
        )
    label = figure_data.alpha_label
    axis.text(
        label.position.x,
        label.position.y,
        label.text,
        color=label.color,
        ha=label.horizontal_alignment,
        va=label.vertical_alignment,
        fontsize=label.size,
        fontweight=label.weight,
    )


def _draw_vectors(axis, figure_data: NutensternFigureData) -> None:
    for vector in figure_data.vectors:
        axis.add_patch(
            FancyArrowPatch(
                (vector.start.x, vector.start.y),
                (vector.end.x, vector.end.y),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.45,
                color=vector.color,
            )
        )


def _draw_labels(axis, figure_data: NutensternFigureData) -> None:
    for label_group in (figure_data.slot_labels, figure_data.vector_labels, figure_data.phase_labels):
        for label in label_group:
            axis.text(
                label.position.x,
                label.position.y,
                label.text,
                color=label.color,
                ha=label.horizontal_alignment,
                va=label.vertical_alignment,
                fontsize=label.size,
                fontweight=label.weight,
            )


def _draw_phase_legend(axis) -> None:
    handles = [plt.Line2D([], [], color=phase.color, linewidth=2.2, label=phase.label) for phase in phase_styles()]
    axis.legend(handles=handles, loc="upper right", frameon=True, title="Colored phases")


def create_figure(figure_data: NutensternFigureData | None = None):
    """Create a matplotlib figure from precomputed Nutenstern figure data."""

    _apply_german_engineering_style()
    data = figure_data or build_nutenstern_figure_data()
    figure, axis = plt.subplots(figsize=data.figure_size, dpi=data.dpi)
    axis.set_title(data.title, fontweight="bold")
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(data.limits.x_min, data.limits.x_max)
    axis.set_ylim(data.limits.y_min, data.limits.y_max)
    axis.set_xlabel("Re(Ψ)")
    axis.set_ylabel("Im(Ψ)")
    axis.grid(True, linestyle=":", linewidth=0.55, color="#d0d0d0")
    axis.axhline(0, color="#303030", linewidth=0.75)
    axis.axvline(0, color="#303030", linewidth=0.75)

    _draw_reference_circle(axis, data)
    _draw_angle_guides(axis, data)
    _draw_vectors(axis, data)
    _draw_labels(axis, data)
    _draw_phase_legend(axis)

    return figure


def save_nutenstern(output_directory: str | Path = "figures", basename: str = "nutenstern") -> tuple[Path, ...]:
    """Save the Nutenstern figure as SVG, PDF, and PNG."""

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    figure = create_figure()
    files = tuple(output_path / f"{basename}.{file_format}" for file_format in _OUTPUT_FORMATS)
    for file in files:
        figure.savefig(file)
    plt.close(figure)
    return files


if __name__ == "__main__":
    save_nutenstern()
