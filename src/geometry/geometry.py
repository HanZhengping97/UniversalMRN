"""Geometry generation for publication-quality Nutenstern figures.

All numerical construction is centralized here so plotting modules can remain
free of mathematical logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin

from .phase_selection import PhaseStyle, phase_for_slot, phase_styles


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class Segment:
    start: Point
    end: Point
    color: str
    label: str
    slot_number: int
    phase: str


@dataclass(frozen=True)
class TextLabel:
    position: Point
    text: str
    color: str
    size: int
    weight: str = "normal"
    horizontal_alignment: str = "center"
    vertical_alignment: str = "center"


@dataclass(frozen=True)
class ArcGuide:
    center: Point
    width: float
    height: float
    theta1: float
    theta2: float
    color: str
    label: str
    label_position: Point


@dataclass(frozen=True)
class CircleGuide:
    center: Point
    radius: float
    color: str
    label: str
    label_position: Point


@dataclass(frozen=True)
class PlotLimits:
    x_min: float
    x_max: float
    y_min: float
    y_max: float


@dataclass(frozen=True)
class NutensternFigureData:
    title: str
    figure_size: tuple[float, float]
    dpi: int
    limits: PlotLimits
    vectors: tuple[Segment, ...]
    slot_labels: tuple[TextLabel, ...]
    vector_labels: tuple[TextLabel, ...]
    phase_labels: tuple[TextLabel, ...]
    alpha_label: TextLabel
    reference_circle: CircleGuide
    angle_guides: tuple[ArcGuide, ...]
    phases: tuple[PhaseStyle, ...]


def _point(radius: float, angle_degrees: float) -> Point:
    angle = radians(angle_degrees)
    return Point(radius * cos(angle), radius * sin(angle))


def _mid_angle(start: float, end: float) -> float:
    return start + ((end - start) / 2.0)


def build_nutenstern_figure_data(slot_count: int = 12) -> NutensternFigureData:
    """Build all coordinates, labels, colors, and guides for a Nutenstern plot."""

    if slot_count < 3:
        raise ValueError("slot_count must be at least 3")

    center = Point(0.0, 0.0)
    vector_radius = 1.0
    slot_label_radius = 1.14
    vector_label_radius = 0.63
    phase_label_radius = 1.36
    reference_radius = 1.0
    guide_radius_alpha = 0.28
    guide_radius_120 = 0.43
    guide_radius_180 = 0.53
    step = 360.0 / slot_count

    vectors: list[Segment] = []
    slot_labels: list[TextLabel] = []
    vector_labels: list[TextLabel] = []
    for index in range(slot_count):
        slot_number = index + 1
        angle = index * step
        phase = phase_for_slot(slot_number)
        end = _point(vector_radius, angle)
        vectors.append(Segment(center, end, phase.color, f"q{slot_number}", slot_number, phase.name))
        slot_labels.append(TextLabel(_point(slot_label_radius, angle), str(slot_number), "#111111", 9, "bold"))
        vector_labels.append(TextLabel(_point(vector_label_radius, angle), f"q{slot_number}", phase.color, 8))

    phase_labels = tuple(
        TextLabel(_point(phase_label_radius, offset), style.label, style.color, 11, "bold")
        for style, offset in zip(phase_styles(), (0.0, 120.0, 240.0), strict=True)
    )

    alpha_start = 0.0
    alpha_end = step
    guides = (
        ArcGuide(center, guide_radius_alpha * 2.0, guide_radius_alpha * 2.0, alpha_start, alpha_end,
                 "#222222", "α", _point(guide_radius_alpha + 0.08, _mid_angle(alpha_start, alpha_end))),
        ArcGuide(center, guide_radius_120 * 2.0, guide_radius_120 * 2.0, 0.0, 120.0,
                 "#666666", "120°", _point(guide_radius_120 + 0.09, 60.0)),
        ArcGuide(center, guide_radius_180 * 2.0, guide_radius_180 * 2.0, 0.0, 180.0,
                 "#888888", "180°", _point(guide_radius_180 + 0.09, 90.0)),
    )

    reference = CircleGuide(center, reference_radius, "#9a9a9a", "Reference circle", Point(0.0, -1.28))

    return NutensternFigureData(
        title="Nutenstern – electrical slot vectors",
        figure_size=(7.0, 7.0),
        dpi=300,
        limits=PlotLimits(-1.55, 1.55, -1.55, 1.55),
        vectors=tuple(vectors),
        slot_labels=tuple(slot_labels),
        vector_labels=tuple(vector_labels),
        phase_labels=phase_labels,
        alpha_label=TextLabel(guides[0].label_position, "electrical angle α", "#222222", 9),
        reference_circle=reference,
        angle_guides=guides,
        phases=phase_styles(),
    )
