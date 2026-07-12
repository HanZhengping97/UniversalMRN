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


## Linear magnetic solver

Phase 5 adds a linear magnetic scalar-potential solve for the existing
non-parametric MRN graph.  The node-branch incidence matrix `A` has shape
`N_nodes x N_branches`.  For each directed branch `start_node -> end_node`, the
stored sign convention is `A[start_node, branch] = -1` and
`A[end_node, branch] = +1`.

The branch constitutive relation is

```text
Phi_b = Gb @ (F_b - A.T @ psi)
```

where `psi` is nodal magnetic scalar potential in A-turn, `F_b` is a generic
branch MMF source vector in A-turn, `Gb` is the diagonal branch permeance matrix
in H, and `Phi_b` is directed branch flux in Wb.  Positive branch flux follows
the stored branch direction.  Nodal flux conservation is

```text
A @ Phi_b = q_n
```

with optional nodal flux injection `q_n` in Wb.  Substitution gives the solved
linear system

```text
Gn @ psi = rhs
Gn = A @ Gb @ A.T
rhs = A @ Gb @ F_b - q_n
```

The full nodal matrix is singular because magnetic scalar potential is defined
only up to an additive constant.  The solver removes one reference-node row and
column, solves the reduced sparse system, and restores the full potential vector
with the requested reference potential.  Nonzero reference potentials are
included in the reduced right-hand side so branch fluxes are invariant to the
chosen gauge.

Branch MMF sources and nodal flux injections represent different source types:
branch MMF is a directed series source on a branch, while nodal flux injection is
a conserved source/sink pair applied at nodes.  A connected closed magnetic
network cannot accept nonzero net nodal flux injection, so the excitation builder
rejects unbalanced `q_n` values.

Run the demonstration and tests with:

```bash
python examples/linear_solver_demo.py
python -m pytest -v
```

Current limits:

- linear isotropic materials only
- no permanent-magnet source generation yet
- no winding excitation generation yet
- no nonlinear steel yet
- no leakage/fringing corrections yet
