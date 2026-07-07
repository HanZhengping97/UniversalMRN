# UniversalMRN Architecture

UniversalMRN is organized into focused packages under `src/` so each scientific
concern can evolve independently:

- `mesh`: mesh primitives, mesh containers, and mesh generation utilities.
- `geometry`: future geometry construction and preprocessing utilities.
- `material`: future material definitions and material-property handling.
- `solver`: future MRN assembly and solution algorithms.
- `utils`: future shared helper utilities.

The current milestone implements only the `mesh` package. It provides immutable
`Node` and `Cell` dataclasses, a mutable `Mesh` container, and a
`MeshGenerator` for uniform rectangular meshes.
