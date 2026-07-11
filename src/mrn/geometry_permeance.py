"""Geometry-derived branch permeance for axisymmetric r-z meshes.

The structured mesh is interpreted as a meridional section of an axisymmetric
magnetic device.  A node-to-node branch owns a dual cross-section assembled
from the rectangular cells touching the branch.  Contributions of touching
cells are parallel magnetic paths and are therefore summed as permeances.
"""

from __future__ import annotations

from dataclasses