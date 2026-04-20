# thermophasepy

**Sequential Stability Evaluation for Solid-State Electrolyte Candidates**

[![CI](https://github.com/your-org/thermophasepy/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/thermophasepy/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

`thermophasepy` implements a three-gate sequential stability pipeline for solid-state
electrolyte (SSE) candidates discovered by machine-learning-guided active learning.
Each gate must pass (or at least not fail) before the next one is evaluated, avoiding
expensive computations on materials that are already ruled out by simpler criteria.

```
Candidate structure
       │
       ▼
┌──────────────────┐
│  Gate 1          │  Born–Huang criterion (elastic tensor eigenvalues)
│  Mechanical      │  Bulk modulus B, Shear modulus G
└────────┬─────────┘
         │ pass / warn
         ▼
┌──────────────────┐
│  Gate 2          │  Anharmonic VDOS from AIMD (VACF method)
│  Vibrational     │  Soft-mode weight at ω ≈ 0
└────────┬─────────┘
         │ pass / warn
         ▼
┌──────────────────┐
│  Gate 3          │  Temperature-dependent convex hull
│  Thermodynamic   │  F(T) = E₀ + F_vib(T),  anharmonic for target
└──────────────────┘
         │
         ▼
    stability_report.json  +  stability_report.pdf
```

## Installation

```bash
pip install thermophasepy                        # core only
pip install "thermophasepy[md]"                  # + MACE / ASE for MD triggering
pip install "thermophasepy[generation]"          # + PyXtal for polymorph generation
pip install "thermophasepy[all]"                 # everything
```

For development:

```bash
git clone https://github.com/your-org/thermophasepy
cd thermophasepy
pip install -e ".[all]"
pytest
```

## Quick start

### Python API

```python
from thermophasepy import run_stability_pipeline

report = run_stability_pipeline(
    sym_dir     = "database/LiZrS2/R3m",
    operating_T = 1200.0,          # K
    device      = "cuda",
)

print(report["overall"])           # "ok" | "warn" | "fail" | "missing"
for flag in report["flags"]:
    print(flag)
```

### Command line

```bash
# Single material
thermophasepy database/LiZrS2/R3m --operating-T 1200 --device cuda

# Entire database
thermophasepy batch --database database --operating-T 1200

# Skip PyXtal generation (faster)
thermophasepy batch --no-generate

# Force recompute
thermophasepy database/LiZrS2/R3m --force
```

### Integration with the active-learning pipeline

In `02_validate.ipynb` (or equivalent), replace the existing stability call with:

```python
from thermophasepy import run_stability_pipeline

for sym_dir in validated_dirs:
    report = run_stability_pipeline(
        sym_dir                 = sym_dir,
        operating_T             = OPERATING_TEMPERATURE,
        device                  = DEVICE,
        generate_missing_phases = True,
    )
    if report["overall"] == "ok":
        retraining_candidates.append(sym_dir)
    elif report["overall"] == "warn":
        # borderline — human review
        flagged.append((sym_dir, report["flags"]))
    # "fail" → excluded automatically
```

## Directory layout

```
database/
  <material>/<symmetry>/
    POSCAR
    relaxation/
      energy.json           ← MACE ground-state energy per atom
    elastic/
      elastic_tensor.json   ← 6×6 stiffness tensor + derived moduli
    phonons/
      dos.json              ← harmonic DOS (fallback for Gate 3)
    aimd/
      <T>K/
        output.traj
        simulation-input.json
        anharmonic_phonons/
          vdos.json         ← cached VACF→VDOS (written by this package)
          thermo.json
          free_energy.json
    stability/
      secondary_phases.json
      hull_vs_T.json
      stability_report.json ← gate results + overall verdict
      stability_report.pdf  ← three-panel dashboard
```

## Configuration

All thresholds live in `thermophasepy/config.py` and can be overridden at runtime:

```python
import thermophasepy.config as cfg

cfg.THERMO_HULL_WARN_EV      = 0.05   # tighter metastability window (50 meV)
cfg.VIB_ZERO_FRAC_FAIL       = 0.05   # stricter soft-mode threshold
cfg.MECH_BULK_WARN_GPa       = 20.0   # demand stiffer electrolytes
cfg.PYXTAL_MAX_ATOMS         = 60     # allow larger unit cells
```

## Gate details

### Gate 1 — Mechanical (Born–Huang)

Reads `elastic/elastic_tensor.json`.  All six eigenvalues of the 6×6 Voigt
stiffness tensor C must be positive (Born–Huang criterion).  The Voigt-averaged
bulk modulus B and shear modulus G are checked against configurable thresholds.

### Gate 2 — Vibrational (anharmonic VDOS)

Reads `aimd/<T>K/anharmonic_phonons/vdos.json`.  The VDOS is extracted from
AIMD trajectories via the velocity autocorrelation function (VACF), averaged
across all available MD temperatures, and the fraction of spectral weight within
a narrow window around ω = 0 is computed.  Significant weight there indicates
soft modes surviving at finite temperature — a signature of structural instability
or pre-transitional dynamics.

### Gate 3 — Thermodynamic (convex hull with T)

Constructs F(T) = E₀ + F_vib(T) for the target material (using the anharmonic
VDOS) and for all secondary phases (harmonic approximation).  Missing polymorphs
are generated automatically with PyXtal.  The pymatgen `PhaseDiagram` engine
builds the convex hull at each temperature and the distance above the hull is
evaluated at `operating_T`.

## Running the tests

```bash
pytest                        # all tests
pytest -k mechanical          # Gate 1 only
pytest -v --tb=long           # verbose
pytest --cov=thermophasepy    # with coverage report
```

## License

MIT — see [LICENSE](LICENSE).

## Citation

If you use this package in your research, please cite:

```bibtex
@article{thermophasepy_2025,
  title   = {Sequential Stability Evaluation for Machine-Learning-Guided
             Discovery of Solid-State Electrolytes},
  author  = {Zeni, Claudio and others},
  journal = {npj Computational Materials},
  year    = {2025},
  note    = {preprint}
}
```
