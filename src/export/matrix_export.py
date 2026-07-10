"""Runtime-only sparse matrix export helpers for debugging."""

from __future__ import annotations

import csv
from pathlib import Path

from scipy import sparse


def export_sparse_matrix_npz(matrix: sparse.spmatrix, output_path: Path) -> Path:
    """Write a sparse matrix using SciPy NPZ format and return the path."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(output_path, matrix.tocsr())
    return output_path


def export_sparse_matrix_csv(matrix: sparse.spmatrix, output_path: Path) -> Path:
    """Write nonzero sparse matrix entries as row,column,value CSV records."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    coo = matrix.tocoo()
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(("row", "column", "value"))
        for row, column, value in zip(coo.row, coo.col, coo.data, strict=True):
            writer.writerow((int(row), int(column), value))
    return output_path
