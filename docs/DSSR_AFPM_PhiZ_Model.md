# DSSR AFPM phi-z no-load MRN model

This document describes the first complete two-dimensional unwrapped `phi-z` non-parametric magnetic reluctance network (MRN) model in UniversalMRN for a slotted double-stator single-rotor (DSSR) spoke-type axial-flux permanent-magnet machine.

The model uses one representative mean radius and maps circumferential distance with

```text
s = r_m * phi
```

The complete mechanical circumference `0 <= phi < 2*pi` is meshed, with periodic circumferential connectivity across the seam.

## Axial stack

The bottom-to-top stack is lower stator yoke, lower teeth/slots, lower air gap, spoke rotor, upper air gap, upper teeth/slots, and upper stator yoke. Upper and lower stators are classified independently and may have independent slot offsets.

## Tooth/slot and spoke-PM convention

Slot openings are periodic around the circumference. A tooth/slot cell is slot air when its center lies within the configured slot-opening angular half-width from a slot center; otherwise it is stator steel tooth. Yokes are continuous linear steel.

The rotor uses one narrow embedded spoke magnet centered at each pole pitch location:

```text
phi_m,k = magnet_position_offset + rotor_mechanical_angle + k * pole_pitch
```

Each magnet extends through the rotor axial thickness in this 2-D model. Magnetization alternates with pole index and is circumferential (`+phi`, `-phi`). Radial magnetization is deliberately not used in this phi-z model because the radial direction is normal to the model plane.

## Reluctance formulas

For circumferential branches, the segment length is `r_m * delta_phi_segment` and the area normal to circumferential flux is

```text
A_phi = radial_span * delta_z
R_phi = r_m * delta_phi_segment / (mu * radial_span * delta_z)
```

For axial branches, each phi cell uses the exact annular-sector area

```text
A_z = 0.5 * (r_o^2 - r_i^2) * delta_phi
R_z = delta_z_segment / [mu * A_z]
```

Material interfaces are represented as two series half-segment reluctances. The periodic seam branch uses the short positive circumferential distance from the final cell center to `2*pi` plus the first cell center from `0`.

## PM branch source and no-load solve

Permanent-magnet cells are assigned `CIRCUMFERENTIAL_POSITIVE` or `CIRCUMFERENTIAL_NEGATIVE` magnetization. PM branch MMF is `Hc` times the magnetized branch length times the projection of magnetization onto branch direction. Thus circumferential branches through PM material are excited, while axial branches through circumferential PM material receive zero PM MMF.

The no-load solve reuses the existing linear scalar-potential MRN solver, then recovers branch flux and segment `B/H` states. Air-gap profiles extract signed axial flux density from internal air-gap axial branches near the middle of each air gap; the profile samples are cell-centered and cover the full periodic circumference without duplicating the endpoint.

## Current limitations

- one representative radial span
- no radial discretization
- linear steel
- no winding current
- no rotor-motion sweep
- no torque
- no back EMF
- no nonlinear saturation
- no fringing correction
- no end effects
