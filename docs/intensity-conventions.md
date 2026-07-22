# Intensity conventions

For a transition wavenumber `nu_tilde` in cm^-1 and transition dipole `mu` in
Debye, the package converts to angular frequency and SI dipole units and uses

```text
omega = 2 pi c (100 nu_tilde)

integral sigma(omega) d omega
    = pi omega f_orient |mu|^2 / (epsilon_0 hbar c)
```

The result is in `m^2/s`. `f_orient=1/3` is used for the isotropic average of a
transition-dipole vector norm. `f_orient=1` is used for a projection polarized
along a specified axis.

This is an angular-frequency integrated absorption cross section. It is not a
HITRAN temperature-dependent rovibrational line intensity and must not be
labeled or compared as one without the partition-function, lower-state,
degeneracy, and stimulated-emission factors required by HITRAN.

The formula is cross-validated against the independently evaluated
spontaneous-emission Einstein coefficient

```text
A = omega^3 |mu|^2 / (3 pi epsilon_0 hbar c^3)
```

through `integral sigma d omega = pi^2 c^2 A / omega^2`. Tests also compare DVR
transition moments against the analytic harmonic-oscillator result for a linear
dipole surface.

As an independent molecular-scale check, the archived H2O, HDO, and D2O
two-stretch calculations are converted to `km/mol` and compared with six
ORCA 6.1 harmonic IR intensities computed at the wB97X-D4/aug-cc-pVTZ level.
The package values reproduce the intensity scale and mode/isotope ordering with
a maximum relative deviation of 35.4%. The source summary and its SHA-256 hash
are recorded in `validation_data/orca_stretch_intensity_benchmarks.json` and
`validation_data/manifest.json`.

That comparison is intentionally a broad model-level benchmark. The package
calculation is a reduced-dimensional variational treatment on a local surface,
whereas the ORCA reference is a full harmonic normal-mode calculation. Agreement
does not establish experimental accuracy or validate temperature-dependent
rovibrational line strengths.

References:

- R. C. Hilborn, *American Journal of Physics* **50**, 982 (1982),
  DOI: `10.1119/1.12937`.
- [HITRAN definitions and units](https://hitran.org/docs/definitions-and-units/).
