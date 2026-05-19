"""heatup.config
=================
Central configuration for all stability thresholds, file paths, physical
constants, and calculator settings.

**All** tuneable parameters live here so that nothing is hardcoded deeper in
the library.  Override any value *before* calling the pipeline functions::

    import heatup.config as cfg

    cfg.THERMO_HULL_WARN_EV   = 0.05   # tighter metastability window
    cfg.MD_ENSEMBLE           = "NVT"  # switch from NPT to NVT
    cfg.CALC_BACKEND          = "chgnet"  # swap the interatomic potential

Calculator backends
-------------------
HeatUp supports any ASE-compatible calculator through the ``CALC_BACKEND``
string and the ``build_calculator`` factory in ``heatup.calculator``.
Currently implemented backends:

    "mace-mp"   — MACE-MP universal potential (default, recommended)
    "chgnet"    — CHGNet universal potential (needs ``chgnet`` package)
    "m3gnet"    — M3GNet universal potential (needs ``matgl`` package)
    "custom"    — Any ASE calculator returned by ``CUSTOM_CALC_FACTORY(device)``

To use a custom calculator, set::

    cfg.CALC_BACKEND = "custom"
    cfg.CUSTOM_CALC_FACTORY = lambda device: MyCalculator(...)

MD ensembles
------------
Two ensembles are available.  Defaults reproduce the tested NPT setup.

    "NPT"  — Nose-Hoover / Parrinello-Rahman barostat (Martyna 1994).
              Cell shape and volume fluctuate; use for studying structural
              transitions and thermal expansion.
    "NVT"  — Nose-Hoover thermostat only, fixed cell.
              Faster per step; suitable when the relaxed cell is trusted.
"""

from __future__ import annotations
import os as _os
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Paths (relative to the project working directory)
# ---------------------------------------------------------------------------

#: Root of the simulation database.
DATABASE_ROOT: str = "database"

#: Root of the MP / generated candidate POSCAR tree.
CANDIDATES_ROOT: str = "input/candidates"

# ---------------------------------------------------------------------------
# Compute defaults
# ---------------------------------------------------------------------------

#: Default MACE compute device ('cuda' or 'cpu').
DEFAULT_DEVICE: str = "cuda"

# ---------------------------------------------------------------------------
# Status codes (do not change — used as dict keys throughout)
# ---------------------------------------------------------------------------

STATUS_OK      = "ok"
STATUS_WARN    = "warn"
STATUS_FAIL    = "fail"
STATUS_MISSING = "missing"

# ---------------------------------------------------------------------------
# Calculator backend
# ---------------------------------------------------------------------------

#: Which interatomic-potential backend to use.
#: Supported: "mace-mp", "chgnet", "m3gnet", "custom".
CALC_BACKEND: str = "mace-mp"

#: Path or name of the MACE-MP model file.
#: Used only when CALC_BACKEND == "mace-mp".
#:
#: Supported values:
#:   "mace-mpa-0-medium"      — MACE-MP-0 medium (recommended, ~100 MB)
#:   "mace-mpa-0-large"       — MACE-MP-0 large  (more accurate, ~300 MB)
#:   "/abs/path/to/model.model" — custom fine-tuned MACE model
#:
#: Override via environment variable to avoid editing source:
#:   export MACE_MODEL_PATH=mace-mpa-0-medium
MACE_MODEL: str = _os.environ.get("MACE_MODEL_PATH", "mace-mpa-0-medium")

#: Enable the D3 dispersion correction (MACE only; ignored by other backends).
MACE_DISPERSION: bool = False

#: Data type for MACE forward passes.
#: 'float64' is more accurate; 'float32' is ~2× faster on GPU.
MACE_DEFAULT_DTYPE: str = "float64"

#: Factory callable ``(device: str) -> ASE Calculator`` used when
#: CALC_BACKEND == "custom".  Replace with your own function::
#:
#:     import heatup.config as cfg
#:     from my_package import MyCalc
#:     cfg.CALC_BACKEND = "custom"
#:     cfg.CUSTOM_CALC_FACTORY = lambda device: MyCalc(device=device)
CUSTOM_CALC_FACTORY: Callable[[str], Any] | None = None

# ---------------------------------------------------------------------------
# MD ensemble & parameters
# ---------------------------------------------------------------------------

#: MD ensemble: "NPT" (Martyna-Tobias-Klein) or "NVT" (Nose-Hoover).
#: NPT is recommended for discovering thermal-expansion-driven phase transitions.
#: NVT is faster and sufficient when you trust the relaxed cell volume.
MD_ENSEMBLE: str = "NPT"

#: Integration timestep in femtoseconds.
MD_TIMESTEP_FS: float = 1.0

#: Total number of MD integration steps.
MD_N_STEPS: int = 30_000

#: Nose-Hoover thermostat time constant in femtoseconds.
#: Controls how quickly the thermostat exchanges energy with the system.
#: Larger → slower temperature coupling; 50 fs is appropriate for most solids.
MD_TTIME_FS: float = 50.0

#: Parrinello-Rahman barostat time constant in femtoseconds.
#: Only used when MD_ENSEMBLE == "NPT".  10× the thermostat time constant
#: (500 fs) is the standard recommendation.
MD_PTIME_FS: float = 500.0

#: Target hydrostatic pressure in GPa.  Zero means ambient (isobaric NPT).
#: Only used when MD_ENSEMBLE == "NPT".
MD_PRESSURE_GPA: float = 0.0

#: Trajectory write interval in MD steps.
#: One frame is saved every NBLOCK steps → effective frame spacing is
#: ``MD_TIMESTEP_FS × MD_NBLOCK`` fs.
MD_NBLOCK: int = 20

#: Number of written frames to discard as equilibration before diffusion
#: analysis and VDOS computation.
#: With NBLOCK=20 and TIMESTEP=1 fs, 500 frames = 10 ps of equilibration.
MD_STEP_EQUIV: int = 100

#: Read stride when loading the saved trajectory for analysis.
#: Set to 1 to use every written frame; increase to reduce memory usage.
MD_STEP_SKIP: int = 1

# ---------------------------------------------------------------------------
# Geometry optimisation parameters
# ---------------------------------------------------------------------------

#: Maximum force component on any atom for BFGS convergence (eV/Å).
RELAX_FMAX: float = 0.05

#: Maximum number of BFGS steps before aborting.
RELAX_MAX_STEPS: int = 500

#: Relax the simulation cell shape and volume in addition to atomic positions.
RELAX_CELL: bool = True

#: When RELAX_CELL is True, keep the volume fixed and relax only the shape.
RELAX_CONSTANT_VOLUME: bool = False

# ---------------------------------------------------------------------------
# Phonon parameters (harmonic, finite-displacement)
# ---------------------------------------------------------------------------

#: Phonon supercell repetitions along each lattice vector.
PHONON_SUPERCELL: tuple[int, int, int] = (3, 3, 3)

#: Finite-displacement magnitude for force-constant calculations (Å).
PHONON_DELTA: float = 0.05

#: Number of band-structure k-points along each high-symmetry segment.
PHONON_NPOINTS: int = 100

#: Monkhorst-Pack grid used for the phonon DOS integration.
PHONON_DOS_KPTS: tuple[int, int, int] = (20, 20, 20)

#: Number of energy grid points in the phonon DOS.
PHONON_DOS_NPTS: int = 200

#: Gaussian smearing width for the phonon DOS (eV).
PHONON_DOS_WIDTH: float = 1e-3

# ---------------------------------------------------------------------------
# AIMD supercell parameters
# ---------------------------------------------------------------------------

#: Minimum lattice parameter length in Å for the AIMD supercell.
#: Repetitions along each axis are the smallest integer that brings every
#: parameter to at least this value.  25 Å is the recommended minimum for
#: capturing the relevant phonon wavelengths and avoiding self-interaction
#: of the mobile ion across periodic images.
AIMD_MIN_CELL_ANG: float = 25.0

# ---------------------------------------------------------------------------
# Elastic tensor parameters
# ---------------------------------------------------------------------------

#: Strain magnitude for the central-difference stress–strain calculation.
#: 1e-4 is small enough to be in the linear regime for most materials.
ELASTIC_DELTA: float = 1e-4

# ---------------------------------------------------------------------------
# Gate 1 — Mechanical stability thresholds
# ---------------------------------------------------------------------------

#: All eigenvalues of the 6×6 Voigt stiffness tensor C must exceed this
#: value (GPa) for the Born–Huang criterion to be satisfied.
MECH_BORN_EIGENVALUE_FAIL_GPa: float = 0.0

#: Bulk modulus (GPa) below which a WARNING is issued.
MECH_BULK_WARN_GPa: float = 10.0

#: Bulk modulus (GPa) below which the material FAILS the mechanical gate.
MECH_BULK_FAIL_GPa: float = 0.0

#: Shear modulus (GPa) below which a WARNING is issued.
MECH_SHEAR_WARN_GPa: float = 5.0

# ---------------------------------------------------------------------------
# Gate 2 — Vibrational stability thresholds (anharmonic VDOS from AIMD)
# ---------------------------------------------------------------------------

#: Half-width of the soft-mode inspection window around ω = 0 (meV).
#: VDOS weight within |ω| < VIB_ZERO_WINDOW_MEV flags potential instability.
VIB_ZERO_WINDOW_MEV: float = 1.0

#: Fraction of integrated normalised VDOS within the zero window that
#: triggers a WARNING.  Pure acoustic broadening at Γ is typically < 1 %.
VIB_ZERO_FRAC_WARN: float = 0.02   # 2 %

#: Fraction above which the material FAILS the vibrational gate.
VIB_ZERO_FRAC_FAIL: float = 0.08   # 8 %

#: Minimum number of production trajectory frames for a reliable VDOS.
VIB_MIN_FRAMES: int = 500

# ---------------------------------------------------------------------------
# Gate 3 — Thermodynamic stability thresholds
# ---------------------------------------------------------------------------

#: E_above_hull (eV/atom) ≤ this value → material is ON the hull (stable).
THERMO_HULL_STABLE_EV: float = 0.0

#: E_above_hull (eV/atom) ≤ this → metastable WARNING.
THERMO_HULL_WARN_EV: float = 0.10

#: Fractional spread of F(T) across multiple AIMD temperatures that triggers
#: a free-energy consistency WARNING.
THERMO_FE_CONSISTENCY_THRESHOLD: float = 0.05

#: Temperature grid (K) for convex-hull construction.
HULL_TEMPERATURES: list[int] = list(range(0, 1501, 50))

#: External pressure for PV term (GPa).  Zero = ambient.
HULL_PRESSURE_GPa: float = 0.0

#: AIMD temperature triggered when no MD simulation exists.
AIMD_TRIGGER_TEMPERATURE_K: float = 900.0

# ---------------------------------------------------------------------------
# Competing-phase discovery: Materials Project API
# ---------------------------------------------------------------------------

#: Materials Project API key.  Set via environment variable MP_API_KEY or
#: override here.  Required when COMPETING_PHASE_SOURCE includes "mp-api".
MP_API_KEY: str = _os.environ.get("MP_API_KEY", "")

#: Source(s) for competing-phase structures, in priority order.
#: Options:
#:   "mp-api"      — Download from Materials Project (requires MP_API_KEY)
#:   "database"    — Scan the local simulation database
#:   "candidates"  — Scan the local candidates tree
#:   "pyxtal"      — Generate random structures with PyXtal (systematic fallback)
#:
#: The list is checked left to right; all sources are combined.
#: "pyxtal" as the last entry provides systematic coverage of unexplored
#: stoichiometries but is slow.  Remove it to skip random generation.
COMPETING_PHASE_SOURCES: list[str] = ["mp-api", "database", "candidates", "pyxtal"]

#: Maximum number of competing phases to fetch per sub-composition from MP.
#: Higher values → more thorough hull but longer preparation time.
MP_MAX_PHASES_PER_COMPOSITION: int = 10

#: Maximum atoms per unit cell for PyXtal random structure generation.
PYXTAL_MAX_ATOMS: int = 40

#: Number of random generation attempts per (formula, space-group) pair.
PYXTAL_MAX_ATTEMPTS: int = 3

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

#: Boltzmann constant in eV/K.
KB_EV: float = 8.617333262e-5

#: Boltzmann constant in meV/K.
KB_MEV: float = 8.617333262145e-2

#: Conversion factor: meV → eV.
MEV_TO_EV: float = 1e-3

#: Planck constant conversion for VDOS frequency axis:
#: fftfreq (cycles/fs) × 1e3 × H_FS_MEV → meV.
H_FS_MEV: float = 4.135667696

#: Minimum physical frequency (meV) below which VDOS modes are treated as
#: acoustic/numerical noise and excluded from thermodynamic integrals.
OMEGA_MIN_MEV: float = 0.1

# ---------------------------------------------------------------------------
# Phonon mode — controls how the vibrational free energy F_vib(T) is computed
# ---------------------------------------------------------------------------
#
# Three methods are available, in increasing accuracy and cost:
#
#   "HA"   — Harmonic Approximation.
#            Phonon DOS computed via finite displacements at the relaxed
#            volume V₀.  Fast.  Accurate at low T.  Misses thermal expansion
#            and anharmonic mode softening.  Valid for Gate 1/2 and for
#            competing-phase free energies.
#
#   "QHA"  — Quasi-Harmonic Approximation.
#            Phonon DOS computed at QHA_N_VOLUMES volumes around V₀.
#            Fits E(V) + F_vib(V, T) to extract the Grüneisen parameter,
#            thermal expansion α(T), and F(T, P) at the equilibrium volume.
#            Captures volume-driven anharmonicity.  Requires QHA_N_VOLUMES
#            phonon calculations → scales as HA × QHA_N_VOLUMES.
#            Produces ``phonons/qha/`` output sub-directory.
#
#   "VDOS" — Anharmonic VDOS from AIMD trajectory via VACF.
#            Full anharmonicity at the simulated temperature.  Most expensive
#            (requires an MD run) but most physically complete.  Used for
#            the *target* material's free energy in the hull by default.
#
# The mode set here is used:
#   • In the analysis notebook (02_analysis.ipynb) for the target material.
#   • In `heatup.free_energy.build_default_assembler`.
#   • In `heatup.thermodynamic.assess_thermodynamic_stability` for the target.
#
# Competing phases always use HA (fast) unless they already have VDOS cached.
PHONON_MODE: str = "VDOS"   # "HA" | "QHA" | "VDOS"

# ---------------------------------------------------------------------------
# Force-constant order (HA and QHA only; ignored for VDOS)
# ---------------------------------------------------------------------------
#
# Controls the order of the interatomic force constant (IFC) expansion:
#
#   2 — Second-order IFCs (standard harmonic).  Implemented via the ASE
#       Phonons class (PHONON_BACKEND = "ase") or phonopy (PHONON_BACKEND =
#       "phonopy").  Fast; sufficient for HA and QHA.
#
#   3 — Third-order IFCs (phonon–phonon interactions).  Requires phono3py.
#       Gives anharmonic phonon linewidths and lattice thermal conductivity.
#       Currently only meaningful for "HA" mode (QHA with 3rd-order is not
#       standard practice).  Very slow (O(N_disp) ≫ 2nd order).
#
# Note: PHONON_MODE = "VDOS" does not use IFCs at all (velocities are from MD).
FORCE_CONSTANT_ORDER: int = 2   # 2 | 3

# ---------------------------------------------------------------------------
# Phonon calculation backend (HA and QHA only)
# ---------------------------------------------------------------------------
#
# "ase"     — Use ASE's built-in Phonons class.  No extra dependencies.
#             Supports FORCE_CONSTANT_ORDER = 2 only.
#             Stores force cache as JSON files alongside the POSCAR.
#
# "phonopy" — Use phonopy (must be installed: pip install phonopy).
#             Supports FORCE_CONSTANT_ORDER = 2.
#             Required for QHA (via phono3py or phonopy's built-in QHA).
#             Produces a standard phonopy_disp.yaml dataset for interoperability.
#
PHONON_BACKEND: str = "ase"   # "ase" | "phonopy"

# ---------------------------------------------------------------------------
# QHA-specific parameters
# ---------------------------------------------------------------------------

#: Number of volume points to sample for the E(V) + F_vib(V,T) fit.
#: Must be odd (symmetric around V₀) and ≥ 3.  Recommended range: 5–11.
#: More points → better Grüneisen / α(T) fit but N times more phonon jobs.
QHA_N_VOLUMES: int = 7

#: Fractional volume range around V₀ for QHA sampling.
#: Volumes span V₀ × (1 − QHA_VOLUME_RANGE) … V₀ × (1 + QHA_VOLUME_RANGE).
#: 0.06 covers ±6 % — appropriate for most ionic / covalent solids at 0–1500 K.
QHA_VOLUME_RANGE: float = 0.06

#: Equation of state model used to fit E(V) in the QHA.
#: Supported by phonopy: "vinet", "birch_murnaghan", "murnaghan".
QHA_EOS: str = "vinet"

# ---------------------------------------------------------------------------
# Run manifest — traceability
# ---------------------------------------------------------------------------
#
# Every output file written by HeatUp is accompanied by a ``manifest.json``
# that records the exact configuration used to produce it.  This ensures
# full reproducibility and makes it easy to compare results from different
# runs (e.g. HA vs QHA, NPT vs NVT).
#
# The manifest is written automatically by heatup.manifest.write_manifest().
# Set WRITE_MANIFEST = False to disable (not recommended).
WRITE_MANIFEST: bool = True


# ---------------------------------------------------------------------------
# Subprocess config propagation
# ---------------------------------------------------------------------------
# When the pipeline spawns child processes (for GPU memory isolation), all
# in-memory config overrides must be forwarded via environment variables,
# because the child process re-imports config.py with its default values.
#
# Usage (inside _cuda_env() of each pipeline module):
#   env.update(config_to_env())
#
# Usage (at the top of each __main__ entry point):
#   import heatup.config as _cfg; _cfg.config_from_env()

#: Prefix used for every config env-var forwarded to child processes.
_ENV_PREFIX = "HEATUP_CFG_"

#: Names of module-level config variables that are forwarded to subprocesses.
#: Extend this list if you add new parameters that subprocesses need.
_FORWARDED_VARS: list[str] = [
    # Calculator
    "CALC_BACKEND", "MACE_MODEL", "MACE_DISPERSION", "MACE_DEFAULT_DTYPE",
    # MD
    "MD_ENSEMBLE", "MD_TIMESTEP_FS", "MD_N_STEPS", "MD_NBLOCK",
    "MD_STEP_EQUIV", "MD_STEP_SKIP", "MD_TTIME_FS", "MD_PTIME_FS",
    "MD_PRESSURE_GPA",
    # Relaxation
    "RELAX_FMAX", "RELAX_MAX_STEPS", "RELAX_CELL", "RELAX_CONSTANT_VOLUME",
    # Phonons
    "PHONON_MODE", "PHONON_BACKEND", "FORCE_CONSTANT_ORDER",
    "PHONON_SUPERCELL", "PHONON_DELTA", "PHONON_NPOINTS",
    "PHONON_DOS_NPTS", "PHONON_DOS_WIDTH",
    # QHA
    "QHA_N_VOLUMES", "QHA_VOLUME_RANGE", "QHA_EOS",
    # AIMD / elastic
    "AIMD_MIN_CELL_ANG", "ELASTIC_DELTA",
    # Paths
    "DATABASE_ROOT", "CANDIDATES_ROOT", "DEFAULT_DEVICE",
    # Manifests
    "WRITE_MANIFEST",
]


def config_to_env() -> dict[str, str]:
    """Serialise live config values to a flat dict of ``HEATUP_CFG_*`` env vars.

    Called by the subprocess helpers in :mod:`heatup.structure_pipeline` and
    :mod:`heatup.md_pipeline` to propagate any in-process overrides (e.g. set
    via ``cfg.CALC_BACKEND = 'chgnet'`` in a notebook) to the child process.

    Returns:
        Dict mapping ``HEATUP_CFG_<NAME>`` → str(value) for every variable
        listed in :data:`_FORWARDED_VARS`.
    """
    import sys as _sys
    this_module = _sys.modules[__name__]
    env: dict[str, str] = {}
    for name in _FORWARDED_VARS:
        val = getattr(this_module, name, None)
        if val is not None:
            env[_ENV_PREFIX + name] = str(val)
    return env


def config_from_env() -> None:
    """Apply ``HEATUP_CFG_*`` environment variables to the live config module.

    Called at the start of each subprocess ``__main__`` entry point so that
    parent-process overrides are honoured inside the child.

    Type coercion is performed using the type of the current default value so
    that, e.g., ``HEATUP_CFG_MD_N_STEPS=5000`` sets an ``int``, not a string.
    Tuple values (e.g. ``PHONON_SUPERCELL``) are parsed from their ``str()``
    representation (e.g. ``"(3, 3, 3)"``).
    """
    import sys as _sys
    import ast as _ast
    this_module = _sys.modules[__name__]
    for name in _FORWARDED_VARS:
        env_key = _ENV_PREFIX + name
        raw = _os.environ.get(env_key)
        if raw is None:
            continue
        current = getattr(this_module, name, None)
        try:
            if isinstance(current, bool):
                coerced = raw.lower() in ("1", "true", "yes")
            elif isinstance(current, int):
                coerced = int(raw)
            elif isinstance(current, float):
                coerced = float(raw)
            elif isinstance(current, tuple):
                coerced = tuple(_ast.literal_eval(raw))
            elif isinstance(current, list):
                coerced = list(_ast.literal_eval(raw))
            else:
                coerced = raw  # str — use as-is
            setattr(this_module, name, coerced)
        except Exception:
            # If parsing fails, keep the default rather than crashing.
            pass
