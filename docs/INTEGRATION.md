# Using HeatUp from another project

HeatUp is meant to be **the** owner of: the interatomic-potential choice, relaxation, elastic tensor,
MD and the stability verdict. Other repositories call it; they should not carry copies.

## 1. The entry point

```python
from heatup import screen, ScreenOptions

r = screen(sym_dir, stage="pre_md" | "post_md", options=ScreenOptions(...))
```

`sym_dir` = `<root>/<material>/<label>/` containing a `POSCAR`. Returns a `ScreenResult`
(`r.to_dict()` is JSON-serialisable; the same dict is written to `<sym_dir>/stability/result_<stage>.json`):

| field | meaning |
|---|---|
| `verdict` | `"accept"`, `"reject"` or `"inconclusive"` |
| `reasons` | list of human-readable strings (why rejected / what is missing / warnings) |
| `e_hull_eV` | signed distance to the hull in eV/atom (negative = below the competitors): **0 K** for `pre_md`, at `operating_T` (same method for every phase) for `post_md` |
| `e_hull_0K_eV` | 0 K enthalpy hull distance (pre_md) |
| `gates` | `{name: GateResult(status, message)}` - `mechanical`, `hull_0K` (pre_md) / `mechanical`, `structural_transition`, `vibrational`, `thermodynamic` (post_md) |
| `decomposition` | competing phases the target would decompose into (when above the hull) |
| `complete` | every gate of the stage had its evidence |
| `anharmonic` (post_md report only) | with `phonon.fc_order = 3`: kappa_RTA and phonon linewidth ratio gamma/nu at the operating T from the cached `fc3` step (`None` if absent); a diagnostic, never changes the verdict |
| `flags`, `model`, `operating_T_K`, `report_file`, `result_file`, `phase` | diagnostics / provenance |

### Stages and verdict rules

* **`pre_md`** - relax, then the enabled gates among: relaxed-ion elastic tensor (Born criterion) and 0 K hull
  against the competing phases. They are independent: a Born failure does not skip the hull. No phonons, no MD. `reject`: Born-unstable, or 0 K hull energy > `pre_md_hull_reject_eV`
  (default `gates.hull_fail_eV` = 100 meV/atom). `accept`: worth paying for MD.
* **`post_md`** - the enabled gates (default all four); `reject` if any fails, `accept` if all are ok.
* `inconclusive` = the evidence does not decide: warnings (unless `accept_warn=True`), a missing gate
  (e.g. `compute=False` and no MD yet), hull undefined (no competing phases / no elemental references),
  a step that failed. `reasons` says which.

`screen` does not raise for compute problems; it raises `ValueError` for a bad `stage` or an unknown option name.

## 2. State and caching (why partial reports cannot be mistaken for full ones)

* Reports are per stage: `stability/report_pre_md.json`, `report_post_md.json` (with `stage` and `complete`
  fields) and `result_<stage>.json`. A stage never reads or overwrites another stage's files.
* A report is **never returned from cache**. Every call recomputes the verdict from the cached *step*
  outputs (`relaxation/`, `elastic/`, `phonons/`, `md/<T>K/`), and each step is invalidated by its own
  `manifest.json` (fingerprint of the options that matter for that step + hashes of its input files). Changing
  the structure, the model or a relevant option recomputes exactly that step and what depends on it.
* Materials Project queries are cached in `<competing_root>/_mp_index/` (re-queried when `mp_max_*` change; delete the
  index file to force a re-query).

## 3. Options are arguments

All options go through `ScreenOptions` (shorthand: `screen(d, "pre_md", operating_T=600)`):
`operating_T`, `md_temperatures`, `mobile_species`, `thresholds` (`{"hull_ok_eV": 0.02, ...}` = any
`gates.*` key), `pre_md_hull_reject_eV`, `accept_warn`, `hull`, `backend`, `model`, `model_overrides`, `device`,
`database_root`, `competing_root`, `sources`, `mp_api_key`, `compute`, `gates`, `force`, `save_figure`,
and `config={"md.n_steps": 20000, "phonon.delta_A": 0.02}` for anything else (`section.option`).
**Gates are independent and selected by config:** `gates=["mechanical", "thermodynamic"]` (or `cfg.gates.enabled`)
runs only those; every enabled gate is evaluated whatever the others say, and a disabled one is reported as
`skipped` and never influences the verdict. **Force rerun:** `force=True` (all steps), `force="md"` or
`force=["relax", "elastic"]` (any of `relax elastic phonons qha fc3 md analysis`) recompute those steps even if their
manifest is fresh; default `None` follows `cfg.run.force` (empty = use the cache).
They apply for the duration of the call and are restored afterwards - the caller's process-wide `cfg` is not
modified. Calls hold a lock (the config is process-global); parallelise with processes, not threads.
Step subprocesses receive the effective options as one JSON env var.

## 4. One owner for the model and the compute code

Do not copy these - import them:

```python
from heatup import default_model, build_calculator, relax_atoms, elastic_tensor, moduli, cfg

default_model("BaTiO3")            # the single model name/path (cfg.calc.model, env MACE_MODEL_PATH,
                                   # per-material cfg.calc.model_overrides)
calc = build_calculator(device="cuda", material="BaTiO3")           # ASE calculator, all backends
relax_atoms(atoms, calc, fmax=0.005, max_steps=500, relax_cell=True)
C = elastic_tensor(atoms, calc, delta=0.004, relax_ions=True, fmax=0.002)     # GPa, 6x6
```

The default is the valid `mace_mp` name `medium-mpa-0`. Legacy spellings that circulate in notebooks
(`mace-mpa-0-medium`) are normalised to it by `cfg.calc.model_for`, so two repos can no longer end up with
different models by typo. To migrate a repo that carries its own copy: delete the copy, set
`MACE_MODEL_PATH` (or pass `ScreenOptions(model=...)`) once, and call the functions above.
MD is run through `screen(..., "post_md")` or `heatup.run_step("md", phase_dir, temperature=T)`.

## 5. Competing phases have their own tree

| tree | owner | HeatUp writes? |
|---|---|---|
| `database_root` | your targets and their results | yes (inside each target's folder) |
| `candidates_root` | your input structures | **never** |
| `competing_root` (default `competing_phases/`) | Materials Project downloads, generated structures, mirrored copies of `candidates` - each with its own relaxation/phonon results | yes |

A candidate used as a competitor is *copied* to `competing_root/<formula>/cand_<label>/` and computed there;
the structure being screened is excluded from its own references. `competing_root` is a cache and can be deleted at any time. Use one per model: a step
computed with another model is recomputed in place (its manifest fingerprint contains the model), so projects
using different models should not share one.

## 6. Things the calling project still decides

* which potential is physically appropriate for its materials (`model`, `model_overrides`) - one potential
  should cover target *and* competitors; HeatUp flags a mismatch and downgrades the verdict;
* `mobile_species` for superionic frameworks, `operating_T`, and the thresholds (not yet calibrated on
  your systems, see `docs/AUDIT.md` section 5).
