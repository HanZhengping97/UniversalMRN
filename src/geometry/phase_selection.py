"""Phase selection helpers for slot-star (Nutenstern) figures."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PhaseStyle:
    """Visual metadata for one electrical phase."""

    name: str
    label: str
    color: str


_PHASE_STYLES: tuple[PhaseStyle, ...] = (
    PhaseStyle("U", "Phase U", "#d62728"),
    PhaseStyle("V", "Phase V", "#1f77b4"),
    PhaseStyle("W", "Phase W", "#2ca02c"),
)


def phase_styles() -> tuple[PhaseStyle, ...]:
    """Return the default three-phase color palette."""

    return _PHASE_STYLES


def phase_for_slot(slot_number: int) -> PhaseStyle:
    """Return the phase style for a one-based electrical slot number."""

    if slot_number < 1:
        raise ValueError("slot_number must be one-based")
    return _PHASE_STYLES[(slot_number - 1) % len(_PHASE_STYLES)]
