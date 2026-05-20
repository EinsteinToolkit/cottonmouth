# Cottonmouth Examples

This folder contains example files to help you get started with Cottonmouth simulations using the [Einstein Toolkit](https://einsteintoolkit.org/) and [CarpetX](https://github.com/EinsteinToolkit/CarpetX).

## Files

### `cottonmouth.th` — Thornlist

A [GetComponents](https://github.com/gridaphobe/CRL) thornlist that checks out all the code needed to build and run these example simulations. It includes:

- **Cactus flesh**: the core Cactus framework.
- **Simulation Factory**: tools for managing and submitting simulations.
- **CarpetX**: the AMReX-based mesh refinement driver.
- **SpacetimeX**: spacetime evolution thorns, including `Z4c`, `AHFinderDirect`, `TwoPuncturesX`, and wave extraction utilities.
- **AsterX**: general-relativistic magnetohydrodynamics (GRMHD) thorns.
- **CanudaX**: gravitational wave extraction via Newman–Penrose scalars.
- **Cottonmouth** — this repository's own thorns (`CottonmouthBSSNOK4m`, `CottonmouthZ4c4m`, gauge wave / linear wave ID thorns).
- Various external libraries (HDF5, MPI, FFTW3, AMReX, openPMD, etc.).

**Usage:**

```bash
./GetComponents cottonmouth.th
```

### `qc0-bssnok.par` and `qc0-z4c.par`: Quasi-circular binary black hole mergers
Quasi-circular binary black hole merger (BSSN-OK)

A Cactus parameter file for evolving an equal-mass, quasi-circular binary black hole (BBH) inspiral through merger using either **`CottonmouthBSSNOK4m`** or **CottonmouthZ4c4m** (4th-order, matter enabled).

**Key physics & numerics:**

| Setting         | Value                                                          |
|-----------------|----------------------------------------------------------------|
| Formulation     | BSSNOK or Z4c                                                  |
| Initial data    | TwoPunctures, equal mass (`m± = 0.453`), quasi-circular orbit  |
| Time integrator | RK4, CFL factor 0.45                                           |
| Domain          | `[-190, +190]³`, 256³ base grid, 7 AMR levels                  |
| Mesh refinement | BoxInBox, regrid every 16 iterations                           |
| Prolongation    | DDF order 5                                                    |
| Wave extraction | Ψ₄ via `CanudaX_NPScalars` with `Multipole` at 8 radii         |
| Horizon finding | `AHFinderDirect` (3 horizons) and `PunctureTracker`            |
| Checkpointing   | openPMD, every 6 hours of wall time                            |

### `mag_TOV.par`: Magnetised Tolman–Oppenheimer–Volkoff neutron star

A Cactus parameter file for evolving a stable, magnetised neutron star in equilibrium using **`CottonmouthBSSNOK4m`** coupled to the **AsterX** GRMHD solver.

**Key physics & numerics:**

| Setting                   | Value                                                        |
|---------------------------|--------------------------------------------------------------|
| Spacetime formulation     | BSSNOK (CottonmouthBSSNOK4m)                                 |
| Hydro solver              | AsterX (HLLE fluxes, PPM reconstruction, 4th-order spatial)  |
| EOS                       | Ideal gas / polytrope (Γ = 2, K = 100)                       |
| Neutron star              | Central density ρ_c = 1.28×10⁻³ → M = 1.4 M☉, M_b = 1.506 M☉ |
| Magnetic field            | Internal dipole, A_b = 100, pressure cutoff 4% of max        |
| Conservative-to-primitive | Noble (primary) + Palenzuela (fallback)                      |
| Domain                    | `[-512, +512]³`, 128³ base grid, 6 AMR levels                |
| Mesh refinement           | BoxInBox (static), finest half-width ≈ 16 M                  |
| Time integrator           | RK4, CFL factor 0.25                                         |
| Final time                | 1600 M (≈ 48.82 M in geometric units as noted in header)     |
| Checkpointing             | openPMD, every 12 hours of wall time                         |

This example is adapted from `AsterX/test/magTOV_Z4c_AMR.par` and is a good starting point for neutron star or GRMHD simulations with Cottonmouth.

