# Benchmarks

This document records the external benchmark datasets we use to validate this repository’s vibrational machinery, and
why/where each dataset is used. The goal is to make the provenance explicit and “paper-ready”.

## ORCA GVPT2 reference dataset (in-repo)

**What:** ORCA harmonic + GVPT2 anharmonic vibrational reference data (fundamentals, overtones/combination bands, IR data).

**Where (authoritative):**

- `reference/all_gvpt2.json`

Derived convenience tables:

- `reference/fundamentals.csv`
- `reference/overtones_combinations.csv`
- `reference/global_band_map.csv`

**Intended use in this repo:**

- Cross-check harmonic frequencies (after removing 6 RT modes for non-linear molecules).
- Cross-check low anharmonic content that overlaps with ORCA (fundamentals, some 2nd-order bands) for the *same species*.
- Provide a consistent, species-keyed reference object for scripts like `scripts/compare_orca_*.py`.

**Notes:**

- ORCA harmonic lists include RT modes; PySCF comparisons typically remove 6 modes (3N–6).
- The ORCA GVPT2 data are treated as read-only benchmark data and should not be regenerated in this repo.

## IUPAC/MARVEL “band origins” for high-overtone truth (external)

ORCA GVPT2 is not designed to be a “truth” source for high overtones (especially beyond the low-order regime). For
validating **high stretch overtones and stretch–stretch combinations** in *real molecules*, we use IUPAC/MARVEL
recommended **J=0 term values** (“vibrational band origins”).

**Important interpretation notes:**

- These are **band origins / J=0 term values**, not necessarily the same as an observed Q-branch maximum.
- Reported uncertainties in the tables are in units of **10⁻⁶ cm⁻¹** and should be converted to **cm⁻¹** before use.
- At high stretch quanta, normal-mode labels can be affected by resonances/mixing; prefer stretch-dominated targets
  (typically v₂=0 for water monomers) when validating a stretch-only 2D model.

**Sources (open-access PDFs):**

- H₂¹⁶O: Tennyson et al., *JQSRT* 117 (2013) 29–58. DOI: 10.1016/j.jqsrt.2012.10.002
- HD¹⁶O: Császár et al., *JQSRT* 111 (2010) 2160–2184. DOI: 10.1016/j.jqsrt.2010.06.012
- D₂¹⁶O: Tennyson et al., *JQSRT* 142 (2014) 93–108. DOI: 10.1016/j.jqsrt.2014.03.019

**Where we record extracted targets for code use:**

- `reference/marvel_iupac/vbo_stretch_targets.json`

This file is intentionally small and contains only the stretch-only / stretch–stretch targets we want to validate
against (for H₂O, HDO, and D₂O), with explicit provenance fields (DOI + table).

## Why the MARVEL deltas are large (current 2D LBS model)

The MARVEL/IUPAC band origins are experiment-anchored **J=0 term values**. In the current implementation we compare
them to a **reduced-dimensional local-bond-stretch (LBS) 2D DVR** model. It is expected that absolute positions can
be offset by **hundreds of cm⁻¹** at high stretch quanta under this approximation.

Primary contributors:

1. **Reduced dimensionality (2D stretches only):** we scan only two bond lengths (R₁, R₂) while **freezing all other
   coordinates** (e.g., bend angle θ and other intermolecular coordinates). At high excitation, stretch–bend coupling
   and relaxation along other coordinates shifts levels.
2. **Simplified kinetic-energy operator (KEO):** the current 2D Hamiltonian uses **diatomic reduced masses** for each
   stretch and a separable product sinc-DVR kinetic energy. This omits the full polyatomic **G-matrix** and
   **kinetic couplings** between the two stretches (e.g., through the shared O atom in monomers).
3. **Electronic-structure/model mismatch:** high overtones probe the PES/DMS far from equilibrium; DFT method, grid,
   and other settings can introduce additional systematic bias in the overtone regime even if fundamentals look
   reasonable.

## ORCA GVPT2 vs this pipeline (what each is good for)

It is useful to distinguish **positions** vs **intensities** and which tool is expected to do better for each:

- **ORCA GVPT2** is a *full-dimensional normal-mode* anharmonic treatment (perturbative). It includes effects that a
  reduced-dimensional LBS model omits (e.g., bend/stretch coupling, kinetic couplings through the polyatomic KEO).
  It is generally most reliable for **fundamentals and low-order anharmonicity** (and is not designed to be a
  variational, high-overtone intensity engine).
- **This repo’s local-mode DVR/VSCF-style machinery** is designed to produce **variational overtone intensities** in
  selected local OH/OD coordinates (Summary2’s ν5–ν7 use case), which ORCA does not provide. However, with a frozen
  coordinate set and simplified KEO, it is not expected to be spectroscopically accurate for absolute band origins
  without further upgrades (KEO and/or additional coordinates).

## Recommended upgrades (accuracy vs runtime)

Below are high-value paths to reduce MARVEL deltas while staying aligned with the Summary2 scope (local coordinates,
variational intensities for ν≥5):

### A) Upgrade the 2D KEO (keep 2D LBS)

**What:** keep coordinates (R₁, R₂) but replace the diatomic-mass kinetic energy with a polyatomic KEO (at least
including bond–bond kinetic coupling through the shared atom).

**Implementation note (current code):** `pyscf_pme_pipeline.py --task 2d` supports `--keo {gmatrix,reduced}`.
`gmatrix` adds a simple constant G-matrix cross term for monomer stretches that share the same O atom; `reduced`
reproduces the historical separable reduced-mass KEO.

**Expected accuracy gain:** often the largest single improvement for triatomic stretch overtones; may reduce
systematic upward shifts substantially (still limited by missing bend coupling).

**Expected runtime increase:** small (PES points unchanged; modest overhead in Hamiltonian assembly).

### B) Add a 3rd coordinate (still local): (R₁, R₂, θ)

**What:** include the bend angle θ along with both stretches. This captures stretch–bend coupling and relaxation.

**Brute-force 3D grid DVR:** scales as npts³ and is usually too expensive beyond monomers.

**Recommended approach:** 3-mode **n-mode/HDMR expansion** (1D and 2D cuts only), which scales like ~3·npts² PES points
instead of npts³.

**Expected accuracy gain:** reduces errors due to missing bend coupling; can materially improve absolute positions
for high stretch quanta compared to 2D-only.

**Expected runtime increase (order-of-magnitude):**

- Monomer: ~1.5–3× vs a single 2D run (HDMR 1D+2D cuts), vs ~npts× for full 3D grids.
- Dimer: full 3D grids are generally impractical; HDMR-style 3-mode models may be feasible but still expensive.

### C) Production strategy if positions remain biased

Summary2 explicitly treats **positions** and **intensities** differently:

- Use VPT2/GVPT2 (full-dimensional) for **positions** where possible (or calibrated shifts),
- Use reduced-dimensional variational DVR/VSCF/VCI for **high-overtone intensities** in local OH/OD coordinates.

This is a common and defensible path when absolute positions from a reduced-dimensional model are not spectroscopically
accurate but the variational intensities and selectivity trends are the main deliverable.

### D) Relaxed local-mode scans (adiabatic PES/DMS cuts)

**What:** instead of a fully frozen scan, displace the target LBS coordinate(s) (e.g., R or (R₁,R₂)) and, at each
grid point, **optimize the remaining geometry** at the same electronic level under constraints.

**Why it helps:** partially recovers relaxation along “missing” coordinates (e.g., bend and intermolecular modes),
often improving absolute **positions** and sometimes intensities compared to a frozen-cut model.

**Expected runtime increase:** large (an optimization per grid point; scales with number of points).

**Failure modes to watch:** discontinuities from switching between local minima / geometry branches; requires strict
constraints and sanity checks for smoothness across the scan.

### E) Adaptive grids / ADGA-style sampling (keep 2D LBS, smarter points)

**What:** replace a uniform (R₁,R₂) grid with an **adaptive sampling** scheme (often called ADGA in the vibrational
literature) that adds PES/DMS points where they are most needed to converge energies/intensities.

**Why it helps:** for a fixed accuracy target, adaptive sampling can reduce the total number of expensive electronic
structure evaluations (runtime win) while also avoiding under-resolution in difficult regions (accuracy win).

**Expected accuracy gain:** primarily improves **numerical convergence** (grid/discretization error), not the
underlying reduced-dimensional model error.

**Expected runtime change:** can be significantly faster than 41×41 uniform grids if the surface is smooth; worst
case approaches the uniform-grid cost (plus modest overhead for the adaptive controller).

**Failure modes to watch:** adaptive criteria that “look converged” for energies but miss dipole/transition-moment
features; needs explicit convergence checks for both **level positions** and **intensities**.
