"""Small sparse linear algebra fallback used when real SciPy is unavailable."""

from __future__ import annotations

import warnings

import numpy as np


class MatrixRankWarning(UserWarning):
    """Warning raised when a sparse solve detects a rank-deficient matrix."""


def spsolve(matrix, rhs):
    """Solve a sparse linear system using NumPy in the fallback implementation."""

    array = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
    vector = np.asarray(rhs, dtype=float)
    try:
        return np.linalg.solve(array, vector)
    except np.linalg.LinAlgError as exc:
        warnings.warn(str(exc), MatrixRankWarning, stacklevel=2)
        return np.full_like(vector, np.nan, dtype=float)
