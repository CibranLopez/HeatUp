"""thermophasepy.config
======================
Central configuration for all stability thresholds, file paths, and
physical constants.  **All** tuneable parameters live here so that
nothing is hardcoded deeper in the library.  Override any value before
calling the pipeline functions, e.g.::

    import thermophasepy.config as cfg
    cfg.THERMO_HULL_WARN_EV = 0.05   # tighter metastability window
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Paths (relative to the project working directory)
# ---------------------------------------------------------------------------

#: Root of the simulation database.
DATABASE_ROOT: str = "database"

#: Root of the MP / generated candidate POSCAR tree.
CANDIDATES_ROOT: str = "input/candidates"

# ---------------------------------------------------------------------------
# Compute
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
# Gate 1 — Mechanical stability thresholds
# ---------------------------------------------------------------------------

#: All eigenvalues of the 6×6 Voigt stiffness tensor C must exceed this
#: value (GPa) for the Born–Huang criterion to be satisfied.
MECH_BORN_EIGENVALUE_FAIL_GPa: float = 0.0

#: Bulk modulus (GPa) below which a WARNING is issued.
MECH_BULK_WARN_GPa: float = 10.0

#: Bulk modulus (GPa) below which the material FAILS the mechanical gate.
#: Negative B means the crystal expands under hydrostatic compression.
MECH_BULK_FAIL_GPa: float = 0.0

#: Shear modulus (GPa) below which a WARNING is issued.
MECH_SHEAR_WARN_GPa: float = 5.0

# ---------------------------------------------------------------------------
# Gate 2 — Vibrational stability thresholds (anharmonic VDOS)
# ---------------------------------------------------------------------------

#: Half-width of the soft-mode inspection window around ω = 0 (meV).
#: VDOS weight within |ω| < VIB_ZERO_WINDOW_MEV is considered potentially
#: pathological (acoustic broadening, anharmonic soft modes, pre-melting).
VIB_ZERO_WINDOW_MEV: float = 1.0

#: Fraction of integrated normalised VDOS within the zero window that
#: triggers a WARNING.  Purely acoustic broadening at Γ is typically < 1 %.
VIB_ZERO_FRAC_WARN: float = 0.02   # 2 %

#: Fraction above which the material FAILS the vibrational gate.
#: Significant zero-frequency weight indicates soft modes that survive at
#: finite temperature — characteristic of structural instability.
VIB_ZERO_FRAC_FAIL: float = 0.08   # 8 %

#: Minimum number of production trajectory frames required for a reliable
#: VDOS.  Fewer frames produce artefactual broadening near ω = 0.
#: At NBLOCK=20, TIMESTEP=1 fs → 500 frames ≈ 10 ps production.
VIB_MIN_FRAMES: int = 500

#: Path to stoichiometry hints JSON file for PyXtal phase generation.
#: When present, non-trivial stoichiometries are added to the 1:1 search.
STOICHIOMETRY_HINTS_PATH: str = "input/stoichiometry_hints.json"

# ---------------------------------------------------------------------------
# Gate 3 — Thermodynamic stability thresholds
# ---------------------------------------------------------------------------

#: E_above_hull (eV/atom) ≤ this value → material is ON the hull (stable).
THERMO_HULL_STABLE_EV: float = 0.0       # exactly on hull (tolerance 1e-4 applied)

#: E_above_hull (eV/atom) ≤ this → metastable WARNING.
THERMO_HULL_WARN_EV: float = 0.10        # 100 meV/atom

#: Fractional spread of F(T) across multiple AIMD temperatures that triggers
#: a free-energy consistency WARNING (may indicate a phase transition between
#: the simulated temperatures).
THERMO_FE_CONSISTENCY_THRESHOLD: float = 0.05   # 5 %

#: Temperature grid (K) for convex-hull construction.
HULL_TEMPERATURES: list[int] = list(range(0, 1501, 50))

#: AIMD temperature used when no MD simulation exists and one must be triggered.
AIMD_TRIGGER_TEMPERATURE_K: float = 900.0

# ---------------------------------------------------------------------------
# Secondary-phase generation (PyXtal)
# ---------------------------------------------------------------------------

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
#: fftfreq (cycles/fs) × 1e3 × H_FS → meV.
H_FS_MEV: float = 4.135667696   # meV·fs

#: Minimum physical frequency (meV) below which VDOS modes are considered
#: acoustic/numerical noise and excluded from thermodynamic integrals.
OMEGA_MIN_MEV: float = 0.1
HULL_PRESSURE_GPa: float = 0.0
