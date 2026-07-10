# UniversalMRN

UniversalMRN is a scientific Python project intended to become a universal
non-parametric Magnetic Reluctance Network (MRN) solver.

The first milestone establishes the project layout and foundational mesh data
structures only. Magnetic calculations are intentionally not implemented yet.


## Structured DSSR AFPM r-z mesh topology

The UniversalMRN structured DSSR AFPM mesh uses an r-z coordinate system.
For compatibility with the original mesh primitives, node ``x`` corresponds to
the radial coordinate ``r`` and node ``y`` corresponds to the axial coordinate
``z``.  Each pair of adjacent structured nodes creates one directed magnetic-
network branch: radial branches point from smaller to larger ``r``, and axial
branches point from smaller to larger ``z``.

Branch generation is topology-only in this development phase. Reluctance, permeability, excitation, and field-solution quantities will
be added in later phases.

## MRN matrix convention

The Phase 3 matrix framework assembles the linear graph matrices for a nodal
magnetic reluctance network, but it does not solve magnetic fields.  The
node-branch incidence matrix ``A`` has shape ``N_nodes x N_branches``.  Each
directed branch follows the stored ``start_node_id -> end_node_id`` convention:
the start-node row receives ``-1`` and the end-node row receives ``+1``.

The branch permeance matrix ``Gb`` is diagonal and ordered by the deterministic
branch-column mapping.  The nodal permeance matrix is assembled as
``Gn = A Gb A^T``.  For a connected network, the complete ``Gn`` is singular
because magnetic scalar potential has an arbitrary reference value; one
reference-node row and column must be removed before a future solve.  Current
permeance values used by the DSSR AFPM example are uniform placeholders only
and are not physical geometry-derived permeances.

