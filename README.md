# HeatUp 0.3

Automatic, temperature-aware stability assessment of a crystal with a machine-learning potential
(MACE-MP by default), plus an optional generative front end for new candidate structures.

| gate | question | evidence |
|---|---|---|
| **1** mechanical | positive-definite stiffness matrix (Born-Huang)? | relaxed-ion elastic tensor (Voigt/Reuss/Hill moduli) |
| **0** structural transition | did the structure stay in its basin? | block-averaged positions, cell strain, symmetry breaking in the MD |
| **2** vibrational | soft modes? | MD VDOS (momenta VACF) and harmonic phonons |
| **3** thermodynamic | above the T-dependent convex hull? | G(T) = H0 + F_vib + F_el + F_mag + F_conf, the *same* method for every phase |

Compute order is cheapest-first (`relax -> elastic -> [1] -> phonons -> MD -> [0][2] -> hull -> [3]`);
a failed gate stops the funnel. Verdicts are `ok / warn / fail / missing` and are recomputed from cached
step outputs, so thresholds can be changed without recomputation. See `docs/AUDIT.md` for what changed
from v0.2, what was verified and what was not.

## Install

```bash
pip install -e .[mace,mp,generate,test]      # mace-torch, mp-api, pyxtal, pytest
export MP_API_KEY=...                         # only for Materials Project downloads
export MACE_MODEL_PATH=/path/to/model.model   # optional; default "medium-mpa-0" (mace_mp name)
```

## Use

**From other software - one call** (details and guarantees in `docs/INTEGRATION.md`):

```python
from heatup import screen, ScreenOptions

r = screen("work/BaTiO3/P4mm", stage="pre_md",             # "pre_md": relax + elastic + 0 K hull (cheap)
           options=ScreenOptions(operating_T=600, mobile_species=[], thresholds={"hull_fail_eV": 0.08}))
r.verdict      # "accept" | "reject" | "inconclusive"
r.reasons, r.e_hull_eV, r.gates                              # why, hull energy (eV/atom), per-gate status
r = screen("work/BaTiO3/P4mm", stage="post_md", options=...) # all four gates incl. MD and the T-dependent hull
```

**Interactive / notebooks:**

```python
from heatup import cfg, run_pipeline, screen_many, summary_rows

cfg.calc.model_overrides = {"BaTiO3": "BTO_MACE.model"}     # per-material fine-tuned model
cfg.hull.operating_T = 600
cfg.gates.mobile_species = []                               # ["Li"] for a superionic conductor

cfg.gates.enabled = ["mechanical", "vibrational", "thermodynamic"]     # gates are independent: pick any subset
cfg.phonon.fc_order = 3                                    # optional: third-order constants (pip install phono3py) -> kappa(T), linewidths
cfg.run.force = ["md"]                                      # recompute these steps (or "all"); [] = use the cache

report = run_pipeline("database/BaTiO3/P4mm", md_temperatures=[600])   # one phase, the enabled gates
print(report["overall"], report["flags"])                              # + stability/report_post_md.{json,png}

rows = screen_many(["input/candidates/X/a", "input/candidates/X/b"], out_dir="screening")   # many targets
```

`heatup.report.print_layout(root)` prints this tree with file sizes (the notebooks do it after every stage).

Directory layout (`<root>/<material>/<label>/`): `POSCAR` -> `relaxation/` `elastic/` `phonons/` `md/<T>K/`
`stability/` (`report_<stage>.json`, `result_<stage>.json`). Every step directory holds a `manifest.json`.
Trees: `database_root` (your targets, read/write), `candidates_root` (your input structures, **read-only**),
`competing_root` (everything HeatUp fetches, generates or mirrors as competing phases, with their results).

Each expensive step runs in a child process (`python -m heatup._worker <step> <phase_dir>`), which returns
all GPU memory to the OS and receives the live config as one JSON env var (`HEATUP_CONFIG`). Only
documented CUDA runtime failures are retried on CPU; real errors are reported once.

Generate and screen new compositions:

```python
from heatup.workflow import generate_and_screen
rows = generate_and_screen({"Ba": 1, "Ti": 1, "O": 3}, n=50, method="pyxtal", out_dir="screening")
# method="matgen" -> MatterGen CLI (cfg.generation.matgen_*), or register_generator("mine", fn)
```

## Modules

`config` `constants` `paths` `structure` `calculator` `manifest` `runtime` - infrastructure ·
`relax` `elastic` `phonons` `md` `md_analysis` `vdos` `vibthermo` - computations ·
`fc3` (anharmonic diagnostics) · `mechanical` `structural_transition` `vibrational` `thermodynamic` - gates ·
`free_energy` `phases` `generation` - thermodynamic inputs ·
`api` (stable entry point) `pipeline` `workflow` `plotting` - orchestration ·
`builders` `report` `validation` - notebook helpers.

## Tests

```bash
pytest                 # 42 fast unit/analytic tests
pytest -m slow         # 11 end-to-end runs with ASE's EMT potential (Cu/Ag/Au), ~4 min
HEATUP_BACKEND=emt jupyter nbconvert --execute notebooks/03_binary_validation.ipynb
```

Notebooks: `01_structure_and_md` (prepare), `02_stability_analysis` (gates + hull), `03_binary_validation`
(Ag-Au-Cu: miscible vs immiscible, elastic moduli vs experiment). Convention (enforced by a test): notebooks hold
configuration and *visible loops* over steps/gates only - no `def`/`class`/`lambda`. Structure builders live in
`heatup.builders`, tables and figures in `heatup.report`, reference data in `heatup.validation`.

## Status

Verified on the EMT toy potential only; the MACE/CHGNet/M3GNet, Materials Project and MatterGen paths
have not been run (see `docs/AUDIT.md`, section 0). Thresholds still need calibration on BaTiO3 and a
diffusive system.
