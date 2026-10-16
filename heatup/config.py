"""heatup.config
=================
Single, typed configuration object for the whole package.

All tunable parameters live in nested dataclasses under the global ``cfg``
object.  Nothing physical is hard-coded elsewhere::

    from heatup import cfg

    cfg.calc.model = "medium-mpa-0"
    cfg.calc.model_overrides = {"BaTiO3": "BTO_MACE.model"}
    cfg.md.n_steps = 40_000
    cfg.gates.mobile_species = ["Li"]

Why a dataclass tree and not module globals (v0.2)?

* it can be **serialised as one JSON blob** and forwarded to child processes
  (``HEATUP_CONFIG``) - the old per-variable ``HEATUP_CFG_*`` scheme silently
  dropped every variable that was not in a hand-maintained list;
* it can be **snapshotted** (``cfg.snapshot()``) for manifests and restored
  with ``config.override(...)``; no more mutate-and-restore hacks;
* typos (``cfg.md.n_stpes = 1``) raise ``AttributeError`` instead of creating
  a new, ignored attribute.

Units: eV, Å, K, fs, GPa unless a field name says otherwise.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses as dc
import json
import os
from typing import Any, Callable, Iterator

# ---------------------------------------------------------------------------
# Status codes (used as dict values in every gate result)
# ---------------------------------------------------------------------------
STATUS_OK = "ok"
STATUS_WARN = "warn"
STATUS_FAIL = "fail"
STATUS_MISSING = "missing"
STATUSES = (STATUS_OK, STATUS_WARN, STATUS_FAIL, STATUS_MISSING)

ENV_VAR = "HEATUP_CONFIG"


class _Frozen:
    """Mixin: forbid creation of unknown attributes (catches typos)."""

    def __setattr__(self, name: str, value: Any) -> None:
        if name not in {f.name for f in dc.fields(self)}:  # type: ignore[arg-type]
            raise AttributeError(
                f"{type(self).__name__} has no option {name!r}. "
                f"Valid options: {[f.name for f in dc.fields(self)]}"  # type: ignore[arg-type]
            )
        object.__setattr__(self, name, value)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

#: names that circulate in notebooks / other repos -> the valid ``mace_mp`` name
#: what ``min_length_A = None`` / ``min_cell_A = None`` mean for each step (A)
AUTO_MIN_LENGTH_A = {"phonon": 12.0, "fc3": 8.0, "md": 15.0}

MODEL_ALIASES = {"mace-mpa-0-medium": "medium-mpa-0", "mace_mpa_0_medium": "medium-mpa-0"}


@dc.dataclass
class CalcCfg(_Frozen):
    """Interatomic-potential backend."""
    #: "mace-mp" | "chgnet" | "m3gnet" | "emt" (tests only) | "custom"
    backend: str = "mace-mp"
    #: ``mace_mp`` model name (``"medium-mpa-0"``, ``"large"``, ...) or a path
    #: to a ``.model`` file.  NB: ``"mace-mpa-0-medium"`` (the v0.2 default)
    #: is *not* a valid mace_mp name.
    model: str = os.environ.get("MACE_MODEL_PATH", "medium-mpa-0")      # THE owner of the default model name
    #: material folder name -> model path (fine-tuned potentials)
    model_overrides: dict = dc.field(default_factory=dict)
    dispersion: bool = False
    dtype: str = "float64"
    #: "auto" -> cuda if usable else cpu
    device: str = "auto"

    def model_for(self, material: str | None) -> str:
        """Model for *material*: per-material override > ``model``; legacy aliases are normalised
        (``mace-mpa-0-medium`` -> ``medium-mpa-0``) so every caller ends up with one valid name."""
        m = self.model_overrides[material] if (material and material in self.model_overrides) else self.model
        return MODEL_ALIASES.get(m, m)


@dc.dataclass
class RelaxCfg(_Frozen):
    #: Tight on purpose: phonons / elastic constants are meaningless on a
    #: loosely relaxed cell (v0.2 used 0.05 eV/Å -> spurious imaginary modes).
    fmax: float = 0.005
    max_steps: int = 1000
    relax_cell: bool = True
    constant_volume: bool = False
    #: Keep the input space group during relaxation (assess *this* phase, do
    #: not let it silently roll downhill into another one).
    fix_symmetry: bool = True
    symprec: float = 1e-3
    #: Hydrostatic pressure of the whole assessment (hull is then an enthalpy hull).
    pressure_GPa: float = 0.0


@dc.dataclass
class PhononCfg(_Frozen):
    """Harmonic phonons (finite displacements, phonopy)."""
    #: Sizing: every entry is ``None`` = "as needed".  An explicit ``supercell`` wins (leave ``min_length_A``
    #: ``None``); otherwise the smallest supercell with perpendicular widths >= ``min_length_A`` is built, and
    #: ``min_length_A = None`` means the built-in 12 A.  ``max_atoms = None`` = no limit (a number = guard).
    supercell: tuple | None = None
    min_length_A: float | None = None
    max_atoms: int | None = None
    delta_A: float = 0.01
    #: q-mesh density: n_i = ceil(mesh_length_A / |a_i|)  (phonopy convention)
    mesh_length_A: float = 80.0
    #: Gaussian smearing used only for *plotting* the DOS (meV)
    dos_sigma_meV: float = 0.5
    band_npoints: int = 61
    symprec: float = 1e-4
    #: Imaginary modes shallower than this (meV, absolute) are treated as numerical noise
    imag_tol_meV: float = 0.5
    #: 2 = harmonic only; 3 = also compute third-order force constants (phono3py, step ``fc3``): kappa(T)
    #: and phonon linewidths for the target, see ``cfg.fc3``.  Needs ``pip install phono3py``.
    fc_order: int = 2


@dc.dataclass
class FC3Cfg(_Frozen):
    """Third-order force constants (phono3py) - used when ``phonon.fc_order == 3``."""
    #: Sizing, same rule as ``phonon``: explicit ``supercell`` wins; else ``min_length_A`` (None = built-in 8 A);
    #: ``max_atoms = None`` = no limit.
    supercell: tuple | None = None
    min_length_A: float | None = None
    max_atoms: int | None = None
    #: displacement amplitude (phono3py default 0.03 A: larger than for fc2 to beat numerical noise)
    delta_A: float = 0.03
    #: ignore atom pairs farther apart than this (A) -> far fewer displaced cells; None = no cutoff
    cutoff_pair_A: float | None = None
    #: compute fc2 on the harmonic supercell (``cfg.phonon``) instead of the small fc3 supercell
    separate_fc2: bool = True
    #: q-mesh density for kappa: n_i = ceil(kappa_mesh_length_A / |a_i|)
    kappa_mesh_length_A: float = 30.0
    temperatures: list = dc.field(default_factory=lambda: [100.0, 200.0, 300.0, 400.0, 600.0, 800.0, 1000.0])
    symprec: float = 1e-5
    #: modes below this frequency (THz) are excluded from the gamma/nu statistics
    min_freq_THz: float = 0.5
    #: a flag is added to the report when <gamma/nu> at the operating T exceeds this
    ratio_warn: float = 0.05


@dc.dataclass
class QHACfg(_Frozen):
    n_volumes: int = 7          # odd, >= 5 recommended
    volume_range: float = 0.06  # +/- fraction of V0
    #: polynomial order of the F(V) fit at each temperature
    poly_order: int = 4
    relax_internal: bool = True # relax ions at every fixed volume (required for QHA)


@dc.dataclass
class ElasticCfg(_Frozen):
    strain: float = 0.004
    #: Relax internal coordinates at every strained cell (relaxed-ion
    #: constants).  v0.2 used clamped ions, which over-stiffens every crystal
    #: with internal degrees of freedom (all ferroelectrics, wurtzite, ...).
    relax_ions: bool = True
    fmax: float = 0.002


@dc.dataclass
class MDCfg(_Frozen):
    ensemble: str = "NPT"            # "NPT" | "NVT"
    timestep_fs: float = 1.0
    n_steps: int = 30_000
    #: Write a frame every ``nblock`` steps.  The VDOS Nyquist limit is
    #: ``h / (2 nblock dt)``: 5 fs -> 413 meV.  (v0.2: 20 fs -> 103 meV, which
    #: aliases every O/H/Li-containing material.)
    nblock: int = 5
    #: Equilibration discarded before analysis (ps)
    equil_ps: float = 3.0
    ttime_fs: float = 50.0
    ptime_fs: float = 100.0
    pressure_GPa: float = 0.0
    #: Bulk modulus used to build the barostat mass ``pfactor = ptime^2 B``.
    #: Overridden by the elastic result when available.
    default_bulk_modulus_GPa: float = 100.0
    #: Sizing, same rule as ``phonon``: explicit ``supercell`` wins; else ``min_cell_A`` (None = built-in 15 A);
    #: ``max_atoms = None`` = no limit.
    supercell: tuple | None = None
    min_cell_A: float | None = None
    max_atoms: int | None = None
    seed: int = 12345
    #: Read stride when analysing trajectories
    stride: int = 1


GATES = ("mechanical", "structural_transition", "vibrational", "thermodynamic")
STEPS = ("relax", "elastic", "phonons", "qha", "fc3", "md", "analysis")


@dc.dataclass
class RunCfg(_Frozen):
    """Execution control."""
    #: Steps to recompute even if cached: any of ``"relax" "elastic" "phonons" "qha" "fc3" "md" "analysis"``
    #: (``analysis`` = VDOS + Gate-0 results derived from the MD) or ``"all"``.  Applies to the target
    #: *and* to the competing phases.  Downstream steps follow automatically: they are invalidated
    #: when the input they consume changes (manifest hashes).
    force: list = dc.field(default_factory=list)

    def is_forced(self, step: str) -> bool:
        return "all" in self.force or step in self.force


def normalize_force(force) -> list:
    """``True`` -> ``["all"]``, ``False``/None -> ``[]``, a step name or list stays a list."""
    if force is True:
        return ["all"]
    if not force:
        return []
    return [force] if isinstance(force, str) else list(force)


def forced(step: str, explicit: bool | None = None) -> bool:
    """Effective force flag of *step*: an explicit argument wins, else ``cfg.run.force``."""
    return bool(explicit) if explicit is not None else cfg.run.is_forced(step)


@dc.dataclass
class GateCfg(_Frozen):
    #: Gates that are evaluated.  They are independent: a failing gate never prevents another
    #: enabled gate from running (only a missing relaxation makes every gate "missing").
    enabled: list = dc.field(default_factory=lambda: list(GATES))
    # Gate 0 - structural transition (block-averaged structure comparison)
    block_ps: float = 2.0             # length of first/last averaging window
    transition_rms_A: float = 0.30    # rms displacement of block-mean positions
    transition_max_A: float = 0.80    # largest single-atom displacement
    cell_shape_strain: float = 0.04   # max |eps_ij| of deviatoric strain (first -> last block)
    cell_volume_change: float = 0.10  # |dV/V|
    symmetry_break_A: float = 0.06    # rms symmetry-breaking of block-mean structure -> warn
    #: Species allowed to diffuse (superionic conductors).  Excluded from the
    #: displacement test and from the soft-mode test.
    mobile_species: list = dc.field(default_factory=list)
    # Gate 1 - mechanical
    born_min_eigenvalue_GPa: float = 0.0
    bulk_warn_GPa: float = 10.0
    bulk_fail_GPa: float = 0.0
    shear_warn_GPa: float = 5.0
    # Gate 2 - vibrational
    zero_window_meV: float = 2.0
    zero_frac_warn: float = 0.01
    zero_frac_fail: float = 0.05
    min_production_ps: float | None = 10.0     # None -> no minimum-length warning
    #: HA: fraction of mode weight that is imaginary (> imag_tol) -> warn / fail
    imag_frac_warn: float = 0.002
    imag_frac_fail: float = 0.02
    # Gate 3 - thermodynamic
    hull_ok_eV: float = 0.010
    hull_fail_eV: float = 0.100
    #: Spread of F_vib(T) between MD temperatures that triggers a warning
    fe_consistency: float = 0.05


@dc.dataclass
class HullCfg(_Frozen):
    #: temperature at which Gate 3 is judged (K); must lie inside ``temperatures``
    operating_T: float = 300.0
    temperatures: list = dc.field(default_factory=lambda: list(range(0, 1501, 50)))
    #: ``"HA"`` for *every* phase (consistent, default) | ``"QHA"`` | ``"VDOS"``
    #: (VDOS only where MD exists; the hull then mixes methods -> flagged).
    phonon_method: str = "HA"
    #: Extra anharmonic overlay computed for the target only (reported next to the consistent hull).
    target_overlay: str = "VDOS"
    include_electronic: bool = True
    include_magnetic: bool = True
    include_configurational: bool = True
    #: Competitors whose 0 K energy lies more than this above the 0 K hull
    #: (eV/atom) are dropped before any phonon calculation.
    prune_window_eV: float | None = 0.15       # None -> no pruning: every competitor gets phonons
    #: O2/N2/F2... gas-phase corrections are NOT applied; see docs/AUDIT.md.
    reference_energy_shift: dict = dc.field(default_factory=dict)


@dc.dataclass
class PhasesCfg(_Frozen):
    #: any of "database", "candidates", "competing", "mp-api", "generate"
    sources: list = dc.field(default_factory=lambda: ["database", "candidates", "competing", "mp-api"])
    mp_api_key: str = os.environ.get("MP_API_KEY", "")
    mp_max_e_above_hull: float | None = 0.15   # query window on MP's own hull; None = no window
    mp_max_per_chemsys: int | None = 25        # None = all phases of every sub-system
    #: your calculations of *targets* (read/write)
    database_root: str = "database"
    #: your own input structures (READ-ONLY: HeatUp never writes here; targets generated by
    #: ``heatup.generation`` go here only when you ask for it)
    candidates_root: str = "input/candidates"
    #: everything HeatUp itself fetches, generates or mirrors for use as *competing phases*
    #: (Materials Project downloads, generated structures, copies of ``candidates``), together
    #: with their relaxation / phonon results.  Safe to share between projects and to delete.
    competing_root: str = "competing_phases"


@dc.dataclass
class GenerationCfg(_Frozen):
    #: "pyxtal" | "matgen" (MatterGen CLI) | "custom"
    method: str = "pyxtal"
    n_per_composition: int = 20
    max_atoms: int = 24
    space_groups: list | None = None      # None -> all
    #: StructureMatcher tolerances for de-duplication
    ltol: float = 0.2
    stol: float = 0.3
    angle_tol: float = 5.0
    #: MatterGen
    matgen_model: str = "mattergen_base"
    matgen_command: str = "mattergen-generate"
    matgen_extra_args: list = dc.field(default_factory=list)
    #: pre-screen: keep candidates whose 0 K E_hull (MLIP) is below this
    prescreen_e_hull_eV: float = 0.10
    seed: int = 0


@dc.dataclass
class Config(_Frozen):
    calc: CalcCfg = dc.field(default_factory=CalcCfg)
    relax: RelaxCfg = dc.field(default_factory=RelaxCfg)
    phonon: PhononCfg = dc.field(default_factory=PhononCfg)
    qha: QHACfg = dc.field(default_factory=QHACfg)
    fc3: FC3Cfg = dc.field(default_factory=FC3Cfg)
    elastic: ElasticCfg = dc.field(default_factory=ElasticCfg)
    md: MDCfg = dc.field(default_factory=MDCfg)
    gates: GateCfg = dc.field(default_factory=GateCfg)
    run: RunCfg = dc.field(default_factory=RunCfg)
    hull: HullCfg = dc.field(default_factory=HullCfg)
    phases: PhasesCfg = dc.field(default_factory=PhasesCfg)
    generation: GenerationCfg = dc.field(default_factory=GenerationCfg)
    #: Factory ``device -> ASE calculator`` for backend="custom" (not serialised).
    custom_calc_factory: Callable | None = None

    # -- serialisation -----------------------------------------------------
    def to_dict(self) -> dict:
        d = {}
        for f in dc.fields(self):
            v = getattr(self, f.name)
            if f.name == "custom_calc_factory":
                continue
            d[f.name] = dc.asdict(v)
        return d

    def snapshot(self) -> dict:
        return json.loads(json.dumps(self.to_dict(), default=_json_default))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=_json_default)

    def update_from_dict(self, d: dict) -> None:
        for sec, vals in d.items():
            if not hasattr(self, sec):
                raise KeyError(f"unknown config section {sec!r}")
            section = getattr(self, sec)
            for k, v in vals.items():
                cur = getattr(section, k)
                if isinstance(cur, tuple) and isinstance(v, list):
                    v = tuple(v)
                setattr(section, k, v)

    def get(self, dotted: str) -> Any:
        sec, key = dotted.split(".", 1)
        return getattr(getattr(self, sec), key)


def _json_default(o: Any) -> Any:
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


#: the global configuration
cfg = Config()


# ---------------------------------------------------------------------------
# Forwarding to child processes
# ---------------------------------------------------------------------------

def to_env(material: str | None = None) -> dict[str, str]:
    """Env-var dict that reproduces the live config in a child process.

    The per-material MACE model is resolved *here* (``calc.model`` is replaced,
    ``model_overrides`` emptied) so the child never needs to know the material.
    """
    d = cfg.snapshot()
    d["calc"]["model"] = cfg.calc.model_for(material)
    d["calc"]["model_overrides"] = {}
    return {ENV_VAR: json.dumps(d)}


def load_from_env() -> bool:
    """Apply ``HEATUP_CONFIG`` (if present) to the live config. Returns True if applied."""
    raw = os.environ.get(ENV_VAR)
    if not raw:
        return False
    cfg.update_from_dict(json.loads(raw))
    return True


def load_json(path: str) -> None:
    """Load a JSON file produced by ``cfg.to_json()`` / hand-written (partial allowed)."""
    with open(path) as fh:
        cfg.update_from_dict(json.load(fh))


@contextlib.contextmanager
def override(**dotted: Any) -> Iterator[Config]:
    """Temporarily change options: ``with override(**{"md.n_steps": 10}): ...``"""
    saved = copy.deepcopy(cfg.to_dict())
    saved_factory = cfg.custom_calc_factory
    try:
        for k, v in dotted.items():
            sec, key = k.split(".", 1)
            setattr(getattr(cfg, sec), key, v)
        yield cfg
    finally:
        cfg.update_from_dict(saved)
        cfg.custom_calc_factory = saved_factory


def reset() -> None:
    """Restore factory defaults (used by tests)."""
    fresh = Config()
    for f in dc.fields(cfg):
        object.__setattr__(cfg, f.name, getattr(fresh, f.name))
