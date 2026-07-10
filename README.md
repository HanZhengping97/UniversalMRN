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

Branch generation is topology-only in this development phase. Reluctance,
permeability, excitation, incidence matrices, and field-solution quantities will
be added in later phases.
