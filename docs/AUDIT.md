# HeatUp audit (v0.2 -> v0.3)

What was reviewed: 3 notebooks + 16 modules of v0.2. What was done: every module rewritten or replaced,
a test suite added, notebooks rebuilt. **Verification status is stated per item** - nothing here
claims more than was run.

## 0. How it was verified (and what was *not*)

| layer | evidence |
|---|---|
| analytic pieces (ModeSet thermodynamics, DOS normalisation, cubic moduli, supercell widths, hull lever rule, config forwarding, verdict logic, option scoping, MP-index cache, candidate mirroring) | 36 unit tests, `pytest` |
| VDOS / Gate 0 | synthetic trajectories with known answers (Einstein oscillators; rigid shift vs internal displacement vs cell shear) |
| whole pipeline, all four gates | **EMT** potential (Cu/Ag/Au) end-to-end: Cu3Au vs Ag/Au/Cu/CuAu -> OK on every gate; bcc Cu -> Born-fail, other enabled gates still evaluated; immiscible Ag-Cu L1_0 -> +22 meV/atom above the hull, decomposing to Ag + Cu; miscible Ag-Au L1_0 -> -16 meV/atom (`pytest -m slow`, 11 tests, ~12 min); `screen()` stages, thresholds-as-arguments, stage isolation and read-only candidates tested end-to-end with EMT |
| notebooks | 01, 02, 03 executed top to bottom with `HEATUP_BACKEND=emt` |
| **MACE-MP / CHGNet / M3GNet backends** | **not run** - no `mace-torch`, GPU, model files or network in the sandbox. `calculator.py` is a careful rewrite but its first real run is yours |
| **Materials Project download** | **not run** (no API key / network). The query was rewritten (see 2.7) and is only unit-tested through the directory logic |
| **MatterGen wrapper** | **not run**; it only builds the CLI call and parses `generated_crystals_cif.zip` |
| thresholds | EMT-calibrated *sanity* only; **not** calibrated on BaTiO3 / MACE (see section 5) |

## 1. Bugs that changed numbers (physics)

1. **DOS normalised to 1 instead of 3 modes/atom** (harmonic DOS reader in `thermodynamic.py`, `anharmonic_phonons.py`): every F_vib was wrong by up to x3 (ZPE and the temperature slope alike). Now one `ModeSet` (omega, weight; sum = 3) is the only vibrational data structure; HA, QHA and VDOS all reduce to it.
2. **VDOS construction** (`anharmonic_phonons.py`): not mass-weighted, no window on the finite VACF (leakage shows up as "soft-mode weight"), no Nyquist/alias/resolution check, frame spacing not tied to the physics. Now: momenta-based mass-weighted VACF, Hann lag window, cosine transform, resolution + Nyquist + alias diagnostics, per-species partial spectra. Verified against HA F_vib (agreement ~1 meV/atom on EMT Cu).
3. **Gate 0 threshold inside the noise**: the DACF plateau is referenced to a *single thermal snapshot*, so even a perfect crystal plateaus at the mean-square thermal displacement (0.01-0.1 A^2 at 1000 K) while the threshold was 0.10 A^2 - the verdict depended on the random first frame. Now block-mean positions (first vs last block), centre-of-mass drift removed, deviatoric cell strain and volume change, symmetry-breaking amplitude using the ideal structure's operations. (spglib on noisy average structures was tried and rejected: unreliable in small shear-fluctuating NPT cells.)
4. **Clamped-ion elastic constants** (`structure_pipeline.py`): over-stiffens every crystal with internal degrees of freedom (all ferroelectrics - BaTiO3 included). Now relaxed-ion central differences; Born criterion = positive-definite 6x6 (Mouhat & Coudert), Voigt/Reuss/Hill reported.
5. **NPT barostat mass** inconsistent with the bulk modulus: `pfactor = ptime^2 B` now uses the material's own B (from the elastic step when available).
6. **QHA** did not relax ions per volume and minimised a different quantity than it reported; now E(V)+F_vib(V,T)+PV per temperature -> G(T), V(T), alpha, B_T with a validity flag when V_eq leaves the sampled range.
7. **Hull construction / MP query**:
   * `summary.search(elements=[...])` returns only phases containing *all* those elements, then filters - binaries and elements were missing, so the hull was wrong. Now every chemical subsystem is queried (capped, within an energy window).
   * the target was compared with competitors by a *different* method (anharmonic vs harmonic). The offset between methods then dominates the hull distance. Now every phase uses the same method; an anharmonic overlay for the target is reported separately.
   * no pruning: phonons were computed for every competitor. Now a 0 K hull prunes phases above `hull.prune_window_eV` first.
   * hull distance is signed (negative = stable by that margin) and elemental references are required explicitly.
8. **Magnetic free energy**: the Inden-Hillert-Jarl function was wrong (missing terms). Now Dinsdale/HJ with Tc, beta, p, only the thermal part G(T)-G(0) added (the ordered ground state is already in the spin-polarised energy).
9. **Electronic free energy**: electrons were conserved to the nominal count instead of the count inside the tabulated window, and F(0) was not subtracted.
10. **Finite pressure** was a separate "EOS" term; now the relaxation is done under pressure and the enthalpy is the reference (the only consistent way).
11. **Default model name** `mace-mpa-0-medium` is not a valid `mace_mp` name (`medium-mpa-0` is); the path to a local `.model` file still works.

## 2. Bugs that changed robustness (engineering)

1. `python -m` child entry point re-executed `heatup.runtime` -> `_worker.py`.
2. Retry-on-CPU after **any** non-zero exit re-ran genuine bugs for hours; now only known CUDA failures (exit 75) are retried.
3. Config forwarding to subprocesses by patching module globals; now one JSON env var carrying the whole dataclass tree; per-material model resolved before forwarding.
4. Caches never invalidated: a changed config or input structure silently reused old results. Now `manifest.json` = config fingerprint (only the keys that matter for that step) + input-file hashes.
5. The final `stability_report.json` was cached and returned after the config changed; gate verdicts are now recomputed from cached step outputs every time (cheap), so thresholds can be changed without recomputation.
6. Steps swallowed errors and returned partial output; they now raise, `run_step` reports.
7. Space-group symbols with `/` (`P4/mmm`) used as directory names created nested directories.
8. PyXtal sweep over all 230 groups for every sub-composition was the default (hours of overlapping junk); it is now opt-in, seeded, per stoichiometry, de-duplicated with StructureMatcher, and followed by a 0 K pre-screen.
9. Supercell sizing used lattice-vector *lengths*; for non-orthogonal cells the *perpendicular widths* determine the image distance.
10. Duplicated helpers across five modules (cell building, hashing, JSON IO, formula parsing) -> `structure.py`, `paths.py`.
11. `aimd/` directory name was misleading (it is classical MD with an MLIP) -> `md/`.
12. Diffusion needed `pymatgen-analysis-diffusion` (heavy, fixed-cell assumption invalid in NPT) -> small FFT multi-origin MSD in `md_analysis.py`.

## 3. Bugs found *by the new tests* during the rewrite

* **phonopy >= 4 defaults to `primitive_matrix="auto"`**: for fcc 4-atom cells the band count was 3 x 1 atom while weights assumed 4 atoms - every harmonic F_vib/ZPE came out **4x too small** with no error. Fixed (`primitive_matrix="P"`) and guarded (raises if bands != 3 N). Caught by the end-to-end EMT test (Cu ZPE 8.3 vs 33.1 meV/atom).
* **ASE NPT needs an *exactly* triangular cell** (tested with `==`); float noise of 1e-15 from a relaxed tetragonal cell crashed the MD. `upper_triangular` now rebuilds the matrix unless the entries are exactly zero. Caught by notebook 03.

## 3b. Integration contract (added in 0.3.0 after the first review)

Requirements from projects that embed HeatUp, and what answers them (details: `docs/INTEGRATION.md`):

| requirement | change |
|---|---|
| one stable entry point returning a typed verdict | `heatup.screen(sym_dir, stage="pre_md"\|"post_md", options=ScreenOptions(...))` -> `ScreenResult` (`verdict` accept/reject/inconclusive, `reasons`, `e_hull_eV`, per-gate results). The multi-target funnel was renamed `screen_many` |
| per-stage caching, no early report returned in place of the full one | reports/results written per stage (`report_<stage>.json`, `result_<stage>.json`, with `stage` and `complete`); a report is never read back as a result - verdicts are recomputed from step caches that have their own manifests. Also fixed: `complete` is false when a gate is `missing` (found by the new tests) |
| options as arguments | `ScreenOptions` (mobile species, thresholds, model, roots, any `section.option`); applied for one call and restored; unknown names raise before any work |
| one owner for model name and relax/elastic/MD code | `default_model`, `build_calculator`, `relax_atoms`, `elastic_tensor`, `moduli` exported; legacy `mace-mpa-0-medium` normalised to `medium-mpa-0` in one place (`cfg.calc.model_for`). **The other repositories were not available here: their copies still have to be deleted and replaced by these imports** |
| sizing options default to `None` | `phonon`, `fc3` and `md` take `supercell`, `min_length_A` (`md.min_cell_A`) and `max_atoms`, all `None` = "as needed": explicit `supercell` wins (a `min_length` set next to it is ignored with a warning), else the smallest cell with perpendicular widths >= `min_length` (None -> built-in 12 / 8 / 15 A, `config.AUTO_MIN_LENGTH_A`), `max_atoms=None` = no limit (a number is a guard that also applies to explicit cells). One resolver for all three: `structure.resolve_supercell`. New `md.supercell`. `hull.prune_window_eV`, `phases.mp_max_e_above_hull`, `phases.mp_max_per_chemsys`, `gates.min_production_ps` also accept `None` (= off / no limit) but keep their numeric defaults, since turning them off changes the cost or the verdicts |
| third-order force constants | new step `fc3` (`heatup/fc3.py`, `cfg.phonon.fc_order = 3`, options in `cfg.fc3`): phono3py fc3 from the active calculator, RTA kappa(T), phonon linewidths gamma/nu, flag when gamma/nu at the operating T exceeds `fc3.ratio_warn`. Diagnostic only - never a gate. Validated on EMT Cu only; phono3py was not run with MACE |
| manifest bug (found while adding fc3) | `qha` and `phonons` shared `phonons/manifest.json`, so running both made each recompute the other forever; `qha` and `fc3` now keep `manifest_<step>.json` (regression test) |
| gates independent and config-selected | `cfg.gates.enabled` / `ScreenOptions.gates`: every enabled gate is evaluated whatever the others say (no `stop_on_fail`, no `stopped_at`); disabled gates are `skipped`; steps run only for enabled gates; a gate that raises becomes `missing` instead of aborting the others |
| force rerun | `cfg.run.force` / `ScreenOptions.force` / `force=` on `run_step`, `run_pipeline`, `screen_many`: `True`/`"all"` or a list of steps (`relax elastic phonons qha md analysis`) |
| database layout visible | `heatup.report.layout` / `print_layout` (tree with file sizes); notebooks show the canonical layout and print it after each stage |
| competing phases in their own tree | `phases.competing_root` (default `competing_phases/`): MP downloads (with a query index, no repeated network calls), generated structures and *mirrored copies* of candidates; `candidates_root` is never written |

## 4. Structure (old -> new)

| v0.2 | v0.3 |
|---|---|
| `config.py` (module globals) | `config.py` (typed dataclass tree `cfg`, `override()`, JSON/env forwarding) + `constants.py` |
| `structure_utils.py`, `md_utils.py` | `structure.py`, `paths.py` |
| `structure_pipeline.py` | `relax.py`, `elastic.py`, `runtime.py`, `_worker.py` |
| `md_pipeline.py` | `md.py`, `md_analysis.py` |
| `anharmonic_phonons.py` | `vdos.py` (+ `vibthermo.py`) |
| `phonons.py` | `phonons.py` (HA + QHA) |
| `free_energy.py` | `free_energy.py` (assembler of independent contributions) |
| `structural_transition.py`, `mechanical.py`, `vibrational.py` | same names, new logic (Gates 0, 1, 2) |
| `thermodynamic.py` | `thermodynamic.py` (hull only) + `phases.py` (discovery) + `generation.py` |
| `pipeline.py`, `plotting.py` | `pipeline.py`, `plotting.py`, **new** `builders.py`/`report.py`/`validation.py` (notebook helpers), `api.py` (`screen`), `workflow.py` (`screen_many`: targets vs references) |
| `manifest.py` | `manifest.py` (fingerprint + hashes) |

Removed: the exhaustive PyXtal default, `stoichiometry_hints.json`-style hints, the `pymatgen-analysis-diffusion` dependency, notebook-level helper code that duplicated library functions.

## 5. Known limitations / not done

* **Thresholds are not calibrated for MACE.** `transition_rms_A`, `zero_frac_*`, `imag_frac_*`, `hull_*` are physically motivated defaults. They need calibration on cases with known answers: BaTiO3's cubic-tetragonal-orthorhombic-rhombohedral sequence (displacive, anharmonic - the hard case), a diffusive/superionic system, and a few known-unstable structures.
* **BTO_MACE vs MPA energy scales.** `model_overrides` lets the target use the fine-tuned model while competitors use MPA. Absolute energies of two different potentials are not comparable; `thermodynamic.py` flags the mismatch and downgrades OK to warn, but the right fix is to use one potential for the whole hull (or fine-tune to include Ba, Ti, O and the competitors).
* **No gas-phase correction** for O2/N2/F2 references (the usual +0.7 eV/O2 DFT-error correction does not transfer to MLIPs trained on mixed data) and no chemical-potential (partial-pressure) hull. For oxides at high T this matters: the hull is at fixed composition without an O reservoir.
* F_el and F_mag are only included if you supply `electronic/edos.json` / `magnetic/moments.json` (no DFT in this package).
* F_conf is ideal mixing on declared partial occupancies, no short-range order.
* QHA uses per-volume harmonic phonons, so it fails where HA has imaginary modes (cubic perovskites): use `VDOS` overlay there.
* Gate 0 declares a structure unstable if *non-mobile* atoms move; superionic frameworks need `gates.mobile_species`. There is no diffusion gate yet (the MSD routine exists).
* Cost: MD length is set by the resolution you want from the VDOS (`gates.min_production_ps` warns, default 10 ps) - for 100+ atom oxides this is the dominant cost.
* CHGNet / M3GNet backends: code path rewritten, never run.
