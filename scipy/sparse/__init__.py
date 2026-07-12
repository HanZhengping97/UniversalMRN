"""Small sparse-matrix compatibility layer used when real SciPy is unavailable."""

from __future__ import annotations

import numpy as np


class spmatrix:
    ndim = 2


class csr_matrix(spmatrix):
    def __init__(self, arg, shape=None, dtype=None):
        if isinstance(arg, csr_matrix):
            self._array = arg._array.copy()
        elif isinstance(arg, tuple) and len(arg) == 2:
            data, (rows, cols) = arg
            if shape is None:
                shape = (max(rows) + 1 if rows else 0, max(cols) + 1 if cols else 0)
            self._array = np.zeros(shape, dtype=dtype or np.asarray(data).dtype)
            for value, row, col in zip(data, rows, cols, strict=True):
                self._array[row, col] += value
        else:
            self._array = np.array(arg, dtype=dtype, copy=True)
        self.shape = self._array.shape

    @property
    def nnz(self):
        return int(np.count_nonzero(self._array))

    @property
    def data(self):
        return self.tocoo().data

    def tocsr(self):
        return csr_matrix(self)

    def tocsc(self):
        return csc_matrix(self._array)

    def tocoo(self):
        rows, cols = np.nonzero(self._array)
        return coo_array(self._array[rows, cols], rows, cols, self.shape)

    def toarray(self):
        return self._array.copy()

    def transpose(self):
        return csr_matrix(self._array.T)

    def diagonal(self):
        return np.diag(self._array)

    def sum(self, axis=None):
        return self._array.sum(axis=axis)

    def __matmul__(self, other):
        if isinstance(other, csr_matrix):
            return csr_matrix(self._array @ other._array)
        result = self._array @ other
        if isinstance(result, np.ndarray) and result.ndim == 2:
            return csr_matrix(result)
        return result

    def __sub__(self, other):
        return csr_matrix(self._array - (other._array if isinstance(other, csr_matrix) else other))

    def __getitem__(self, key):
        result = self._array[key]
        if isinstance(result, np.ndarray) and result.ndim == 2:
            return csr_matrix(result)
        return result


class csc_matrix(csr_matrix):
    @property
    def indptr(self):
        counts = np.count_nonzero(self._array, axis=0)
        return np.concatenate(([0], np.cumsum(counts)))

    @property
    def indices(self):
        indices = []
        for col in range(self._array.shape[1]):
            indices.extend(np.nonzero(self._array[:, col])[0].tolist())
        return np.asarray(indices, dtype=int)

    @property
    def data(self):
        data = []
        for col in range(self._array.shape[1]):
            rows = np.nonzero(self._array[:, col])[0]
            data.extend(self._array[rows, col].tolist())
        return np.asarray(data)

    def tocsc(self):
        return csc_matrix(self._array)


class coo_matrix(csr_matrix):
    pass


class coo_array:
    def __init__(self, data, row, col, shape):
        self.data = np.asarray(data)
        self.row = np.asarray(row, dtype=int)
        self.col = np.asarray(col, dtype=int)
        self.shape = shape


def diags(diagonal, offsets=0, shape=None, format=None):
    if offsets != 0:
        raise NotImplementedError("fallback diags only supports main diagonal")
    diagonal = np.asarray(diagonal)
    if shape is None:
        shape = (len(diagonal), len(diagonal))
    array = np.zeros(shape, dtype=diagonal.dtype)
    np.fill_diagonal(array, diagonal)
    return csr_matrix(array)


def save_npz(file, matrix):
    np.savez(file, data=matrix.tocoo().data, row=matrix.tocoo().row, col=matrix.tocoo().col, shape=matrix.shape)
