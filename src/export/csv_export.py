"""CSV export helpers for debugging mesh topology."""

from __future__ import annotations

import csv
from pathlib import Path

from src.mesh import Mesh


def export_branches_csv(mesh: Mesh, output_path: Path) -> Path:
    """Export mesh branches to a CSV file and return the written path."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=(
                "branch_id",
                "start_node_id",
                "end_node_id",
                "orientation",
                "length",
                "center_r",
                "center_z",
            ),
        )
        writer.writeheader()
        for branch in sorted(mesh.branches.values(), key=lambda item: item.id):
            writer.writerow(
                {
                    "branch_id": branch.id,
                    "start_node_id": branch.start_node_id,
                    "end_node_id": branch.end_node_id,
                    "orientation": branch.orientation.name,
                    "length": branch.length,
                    "center_r": branch.center_r,
                    "center_z": branch.center_z,
                }
            )
    return output_path
