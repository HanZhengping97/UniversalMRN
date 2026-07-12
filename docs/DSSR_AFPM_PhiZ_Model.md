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

## Upper/lower air-gap sign convention and symmetry diagnostics

All axial branches in the phi-z MRN are directed toward increasing `z`. With this single branch convention, a symmetric DSSR machine does **not** produce equal signed upper and lower axial air-gap fields. Flux crossing the upper air gap from the rotor toward the upper stator is positive `B_z`, while flux crossing the lower air gap from the rotor toward the lower stator is negative `B_z` under the same `+z` branch direction.

The expected signed relation is therefore

```text
B_upper(phi) ~= -B_lower(phi)
```

The signed upper/lower symmetry error is computed as

```text
epsilon_sym = max |B_upper(phi) + B_lower(phi)|
```

This differs from the old visual comparison `max |B_upper - B_lower|`, which incorrectly treats the two axial branches as if they had opposite reference directions. The backwards-compatible `upper_lower_symmetry_error` property now aliases the signed `B_upper + B_lower` definition.

A separate magnitude-only metric is also reported:

```text
epsilon_mag = max ||B_upper(phi)| - |B_lower(phi)||
```

The signed metric validates cancellation under the global axial sign convention. The magnitude metric validates that the absolute field strengths match even if a caller wants a sign-normalized plot such as `B_upper` compared directly with `-B_lower`.

Air-gap extraction validates that selected branches are axial, internal to the intended air-gap region, bounded only by air material, located at matching phi centers, closest to the corresponding air-gap midplane, sorted deterministically, and free of duplicated `0`/`2*pi` endpoint samples. Integrated signed flux diagnostics use each profile sample's annular-sector axial area rather than a raw angular integral.

A lightweight full-circumference Fourier diagnostic reports the DC component, dominant mechanical spatial harmonic, and selected amplitudes. For the default 44-pole example the pole-pair order is `p = pole_count / 2 = 22`; the model reports the amplitude at order 22 without requiring it to be the largest harmonic for all discretizations.

## Physical interpretation of the spoke rotor

The DSSR AFPM phi-z model represents a mean-radius unwrap of the real three-dimensional spoke rotor. In the physical annulus, the permanent magnets are radial spoke bars. At the representative mean radius used by the two-dimensional phi-z mesh, each radial bar appears as a narrow circumferential interval extending through the rotor axial thickness.

The rotor angular convention is based on `pole_count = Np` and `pole_pitch = 2*pi/Np`. Spoke-magnet centers are located at

```text
phi_pm,k = rotor_mechanical_angle + magnet_position_offset + k * pole_pitch
```

and rotor-iron pole-piece centers are halfway between adjacent magnets:

```text
phi_pole,k = rotor_mechanical_angle + magnet_position_offset + (k + 0.5) * pole_pitch
```

The resulting unwrapped sequence is therefore:

```text
PM+ | iron pole | PM- | iron pole | PM+
```

The preferred geometry parameter is `magnet_arc_ratio = magnet_angular_width / pole_pitch`. The default 44-pole example uses `magnet_arc_ratio = 0.20`, so the demonstration rotor has a narrow 20% pole-pitch magnet interval and a wide 80% pole-pitch rotor-iron pole-piece interval. This is a replaceable demonstration value, not an optimized design result. A physical thesis-style magnet width can still be supplied as `magnet_circumferential_width`; internally it is converted to angular width by dividing by mean radius and must agree with the ratio if both inputs are provided.

The magnetization in the PM intervals is circumferential and alternates `+phi`, `-phi`, `+phi`, ... around the circumference. Circumferential branches through PM cells therefore receive permanent-magnet MMF, while axial PM branches receive zero PM source. The rotor iron between neighboring magnets is modeled as `ROTOR_IRON_POLE`; it carries no permanent-magnet source and forms the geometric axial pole face toward the upper and lower air gaps. The pole-face diagnostic infers alternating `N, S, N, S, ...` source polarity from the neighboring magnet directions; it is a geometric/source-polarity diagnostic, not a nonlinear surface-flux solution.

This phi-z model captures the circumferential source action of the spoke magnets and the axial air-gap flux leaving the rotor-iron pole pieces toward both stators while retaining the existing linear magnetic-reluctance-network solver. Radial bridges, inner/outer retaining structures, radial leakage paths, nonlinear saturation, winding excitation, torque, back-EMF, and rotor-motion sweeping are not represented in this refinement.
