"""heatup.core.config
====================
Single, typed configuration object for the whole package (+ physical constants).

All tunable parameters live in nested dataclasses under the global ``cfg``
object.  Nothing physical is hard-coded elsewhere::

    from heatup import cfg

    cfg.calc.model = "medium-mpa-0"
    cfg.calc.model_overrides = {"BaTiO3": "BTO_MACE.model"}
    cfg.md.n_steps = 40_000
    cfg.analysis.mobile_species = ["Li"]

Why a dataclass tree and not module globals (v0.2)?

* it can be **serialised as one JSON blob** and forwarded to child processes
  (``HEATUP_CONFIG``) - the old per-variable ``HEATUP_CFG_*`` scheme silently
  dropped every variable that was not in a hand-maintained list;
* it can be **snapshotted** (``cfg.snapshot()``) for manifests and restored
  with ``config.override(...)``; no more mutate-and-restore hacks;
* typos (``cfg.md.n_stpes = 1``) raise ``AttributeError`` instead of creating
  a new, ignored attribute.

Who owns which section (the three notebooks)
--------------------------------------------
============  ==========================================================  =========
notebook      sections                                                    stored in
============  ==========================================================  =========
1 prepare     ``relax elastic harmonic quasiharmonic perturbative md``    ``plan.json`` of every phase
              ``vibrational terms plan phases generation``
2 compute     ``calc`` (MLIP, device) ``run``                             manifests / ``energy.json``
3 analyze     ``analysis gates hull`` (+ ``terms`` / ``vibrational`` as prepared)  reports
============  ==========================================================  =========
Notebook 2 applies the options stored in each ``plan.json`` (``ledger.applied``), so the calculation
settings are written down exactly once.

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
# Physical constants and unit conversions (single source)
# ---------------------------------------------------------------------------
KB_EV = 8.617333262e-5          # Boltzmann constant, eV / K
KB_MEV = KB_EV * 1e3            # meV / K
EV_A3_TO_GPA = 160.21766208     # eV/A^3 -> GPa
GPA_TO_EV_A3 = 1.0 / EV_A3_TO_GPA
H_MEV_FS = 4135.667696          # Planck constant in meV*fs:  E[meV] = H_MEV_FS * nu[cycles/fs]
MEV_PER_THZ = 4.135667696       # 1 THz in meV
MODES_PER_ATOM = 3.0            # vibrational modes per atom (3 for a crystal)
OMEGA_MIN_MEV = 0.1             # |omega| below this = acoustic/numerical zero mode, excluded from F_vib (meV)

# ---------------------------------------------------------------------------
# Status codes (used as dict values in every gate result)
# ---------------------------------------------------------------------------
STATUS_OK = "ok"
STATUS_WARN = "warn"
STATUS_FAIL = "fail"
STATUS_MISSING = "missing"
STATUSES = (STATUS_OK, STATUS_WARN, STATUS_FAIL, STATUS_MISSING)

ENV_VAR = "HEATUP_CONFIG"


class Frozen:
    """Mixin: forbid creation of unknown attributes (catches typos).  Every configuration section is a ``@dataclass`` of this kind."""

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
AUTO_MIN_LENGTH_A = {"harmonic": 12.0, "perturbative": 8.0, "md": 15.0}

MODEL_ALIASES = {"mace-mpa-0-medium": "medium-mpa-0", "mace_mpa_0_medium": "medium-mpa-0"}


@dc.dataclass
class CalcCfg(Frozen):
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
        if os.path.isfile(m):
            return os.path.abspath(m)                     # a model FILE is always passed as a full path (the same file for every package)
        return MODEL_ALIASES.get(m, m)


@dc.dataclass
class RelaxCfg(Frozen):
    #: Tight on purpose: phonons / elastic constants are meaningless on a
    #: loosely relaxed cell (v0.2 used 0.05 eV/Å -> spurious imaginary modes).
    fmax: float = 0.005
    max_steps: int = 1000
    #: False: the structure is used as given (already relaxed with another code/model): nothing is optimised, only
    #: its energy, forces and stress are evaluated with the active calculator (the hull needs energies from the same
    #: calculator as the phonons).  In notebook 1 this follows from ``"relax"`` being in ``cfg.plan.gates``.
    optimize: bool = True
    #: Relax the cell too (False: ions only, the given cell is kept).
    relax_cell: bool = True
    constant_volume: bool = False
    #: Keep the input space group during relaxation (assess *this* phase, do
    #: not let it silently roll downhill into another one).
    fix_symmetry: bool = True
    symprec: float = 1e-3
    #: Hydrostatic pressure of the whole assessment (hull is then an enthalpy hull).
    pressure_GPa: float = 0.0


@dc.dataclass
class HarmonicCfg(Frozen):
    """HARMONIC level: second-order force constants at the relaxed 0 K cell (finite displacements, phonopy).  The same
    settings (supercell, displacement, q-mesh) are used by the quasi-harmonic level at every volume."""
    #: Sizing.  An explicit ``supercell`` wins (leave ``min_length_A`` ``None``).  Default (3, 3, 3): it is the number of
    #: replications that sets the commensurate q-grid on which the force constants are exact, and it is the same for
    #: every phase and, in ``quasiharmonic``, at every volume.  ``supercell = None`` = automatic: the smallest supercell with
    #: perpendicular widths >= ``min_length_A`` (``None`` = the built-in 12 A).  ``max_atoms = None`` = no limit.
    #: NB: a phase is computed with the options stored in its ``plan.json``, so a changed default reaches phases that
    #: are already planned only after their plans are updated.
    supercell: tuple | None = (3, 3, 3)
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


@dc.dataclass
class PerturbativeCfg(Frozen):
    """PERTURBATIVE ANHARMONIC level: higher-order force constants on top of the harmonic crystal (step ``perturbative``).
    Implemented: third order (phono3py) -> kappa(T) and phonon linewidths, a DIAGNOSTIC.  Fourth order is reserved (a
    free-energy correction needs fc3 AND fc4 at the same order of perturbation theory)."""
    #: highest order of the force constants.  3 = fc3 (implemented); 4 = fc4 (not implemented yet: raises).
    order: int = 3
    #: Sizing, same rule as ``harmonic``: explicit ``supercell`` wins; else ``min_length_A`` (None = built-in 8 A);
    #: ``max_atoms = None`` = no limit.
    supercell: tuple | None = None
    min_length_A: float | None = None
    max_atoms: int | None = None
    #: displacement amplitude (phono3py default 0.03 A: larger than for fc2 to beat numerical noise)
    delta_A: float = 0.03
    #: ignore atom pairs farther apart than this (A) -> far fewer displaced cells; None = no cutoff
    cutoff_pair_A: float | None = None
    #: compute fc2 on the harmonic supercell (``cfg.harmonic``) instead of the small fc3 supercell
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
class QuasiHarmonicCfg(Frozen):
    """QUASI-HARMONIC level: the harmonic fc2 at several volumes (ions relaxed at each), F(V,T) minimised at every T."""
    n_volumes: int = 7          # odd, >= 5 recommended
    volume_range: float = 0.06  # +/- fraction of V0
    #: polynomial order of the F(V) fit at each temperature
    poly_order: int = 4
    relax_internal: bool = True # relax ions at every fixed volume (required for QHA)
    #: temperature grid of G(T) stored in ``quasiharmonic.json``.  The hull grid (``cfg.hull.temperatures``) must lie inside it.
    temperatures: list = dc.field(default_factory=lambda: list(range(0, 1501, 50)))


@dc.dataclass
class ElasticCfg(Frozen):
    strain: float = 0.004
    #: Relax internal coordinates at every strained cell (relaxed-ion
    #: constants).  v0.2 used clamped ions, which over-stiffens every crystal
    #: with internal degrees of freedom (all ferroelectrics, wurtzite, ...).
    relax_ions: bool = True
    fmax: float = 0.002


@dc.dataclass
class MDCfg(Frozen):
    ensemble: str = "NPT"            # "NPT" | "NVT"
    timestep_fs: float = 1.0
    n_steps: int = 30_000
    #: Write a frame every ``nblock`` steps.  The VDOS Nyquist limit is
    #: ``h / (2 nblock dt)``: 10 fs -> 207 meV (covers oxides, sulfides, halides; not O-H/N-H stretches),
    #: 5 fs -> 413 meV.  (v0.2: 20 fs -> 103 meV, which aliases every O/H/Li-containing material.)
    #: The same run is also the source of the diffusivity labels, which need 20 fs or finer.
    nblock: int = 10
    ttime_fs: float = 50.0
    ptime_fs: float = 100.0
    pressure_GPa: float = 0.0
    #: Bulk modulus used to build the barostat mass ``pfactor = ptime^2 B``.
    #: Overridden by the elastic result when available.
    default_bulk_modulus_GPa: float = 100.0
    #: Sizing, same rule as ``harmonic``: explicit ``supercell`` wins; else perpendicular widths >= ``min_cell_A``
    #: (None = built-in 15 A).  25 A: the cell must also give converged diffusivities, not only a stability verdict.
    #: ``max_atoms`` caps the cell (None = no limit); when the cap does not allow ``min_cell_A`` the target is lowered
    #: in steps of 1 A down to ``min_cell_floor_A``, and only below that is the structure refused.
    supercell: tuple | None = None
    min_cell_A: float | None = 25.0
    max_atoms: int | None = 1500
    min_cell_floor_A: float | None = 12.0
    seed: int = 12345


GATES = ("mechanical", "structural_transition", "vibrational", "thermodynamic")   # the one definition
REQUESTS = ("relax", *GATES)         # what ``cfg.plan.gates`` may contain


@dc.dataclass
class RunCfg(Frozen):
    """Execution control."""
    #: Steps to recompute even if cached: step names of ``ledger.STEPS`` (``relax elastic harmonic quasiharmonic
    #: perturbative md vdos transition mdstats``), the group ``"analysis"`` (= vdos + transition + mdstats) or ``"all"``.  Applies to
    #: every phase of the database (targets and competing phases).  Downstream steps follow automatically: they are invalidated
    #: when the input they consume changes (manifest hashes).
    force: list = dc.field(default_factory=list)

    def is_forced(self, step: str) -> bool:
        from heatup.core.ledger import STEPS          # lazy: ledger imports this module
        spec = STEPS.get(step)
        return "all" in self.force or step in self.force or bool(spec and spec.group and spec.group in self.force)


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
class AnalysisCfg(Frozen):
    """How trajectories are read and reduced (notebook 3).  Changing these re-runs only the cheap analysis steps."""
    #: Equilibration discarded before analysis (ps)
    equil_ps: float = 3.0
    #: Read stride when analysing trajectories
    stride: int = 1
    #: A species counts as mobile (diffusing) when its final mean-square displacement exceeds this (A^2).  Vibrations in
    #: a solid give 0.1-0.5 A^2, a hop is (2-3 A)^2: a model-free criterion, independent of any assumed ionic charge.
    msd_mobile_A2: float = 4.0
    #: length of the first/last averaging window of the structural-transition check (ps)
    block_ps: float = 2.0
    #: longest VACF lag used for the VDOS (ps)
    vdos_max_lag_ps: float = 10.0
    #: Species allowed to diffuse (superionic conductors).  Excluded from the displacement test (structural transition)
    #: and from the soft-mode test (vibrational).
    mobile_species: list = dc.field(default_factory=list)


@dc.dataclass
class GateCfg(Frozen):
    #: Gates that are reported.  None (default) -> the gates each phase's plan asked for (``cfg.plan.gates``),
    #: else all four.  They are independent and unordered: any one can be asked for alone, a failing gate never
    #: prevents another from running (only a missing relaxation makes every gate "missing").
    enabled: list | None = None
    # structural transition (block-averaged structure comparison)
    transition_rms_A: float = 0.30    # rms displacement of block-mean positions
    transition_max_A: float = 0.80    # largest single-atom displacement
    cell_shape_strain: float = 0.04   # max |eps_ij| of deviatoric strain (first -> last block)
    cell_volume_change: float = 0.10  # |dV/V|
    symmetry_break_A: float = 0.06    # rms symmetry-breaking of block-mean structure -> warn
    # mechanical
    born_min_eigenvalue_GPa: float = 0.0
    bulk_warn_GPa: float = 10.0
    bulk_fail_GPa: float = 0.0
    shear_warn_GPa: float = 5.0
    # vibrational
    zero_window_meV: float = 2.0
    zero_frac_warn: float = 0.01
    zero_frac_fail: float = 0.05
    min_production_ps: float | None = 10.0     # None -> no minimum-length warning
    #: harmonic / quasi-harmonic: fraction of mode weight that is imaginary (> imag_tol) -> warn / fail
    imag_frac_warn: float = 0.002
    imag_frac_fail: float = 0.02
    # thermodynamic
    hull_ok_eV: float = 0.010
    hull_fail_eV: float = 0.100
    #: Spread of F_vib(T) between MD temperatures that triggers a warning
    fe_consistency: float = 0.05


@dc.dataclass
class HullCfg(Frozen):
    """The temperature-dependent hull.  WHICH free-energy terms enter it is ``cfg.terms`` (one place)."""
    #: default temperature at which the thermodynamic check is judged (K); must lie inside ``temperatures``.  A phase's
    #: own ``plan.json`` value (set in notebook 1) wins.
    operating_T: float = 300.0
    temperatures: list = dc.field(default_factory=lambda: list(range(0, 1501, 50)))
    #: Extra anharmonic overlay computed for the target only (reported next to the consistent hull): ``"anharmonic"``
    #: (needs MD of the target) | ``"off"``.  It is an ESTIMATE, never the verdict.
    target_overlay: str = "anharmonic"
    #: O2/N2/F2... gas-phase corrections are NOT applied; see docs/AUDIT.md.
    reference_energy_shift: dict = dc.field(default_factory=dict)


#: the terms of the free energy (``heatup.core.contributions`` declares their models, steps, files and rules)
TERM_KEYS = ("vibrational", "electronic", "magnetic", "configurational", "orientational", "defects")


@dc.dataclass
class TermsCfg(Frozen):
    """Which terms make up G(T) = H0 + sum of the selected terms.  ``None`` = as prepared in notebook 1 (stored in each
    ``plan.json``), and if nothing was prepared, the default of the term (``contributions.TERMS``).  ONE model per term,
    the SAME for every phase of a hull.

    ``vibrational``: ``"off"`` static 0 K hull | ``"ZPE"`` zero-point energy only | ``"harmonic"`` | ``"quasiharmonic"``
    | ``"anharmonic"`` (effective-harmonic F from the MD spectra; experimental).  Old names ``HA`` / ``QHA`` / ``VDOS`` are
    still read."""
    vibrational: str | None = None
    electronic: str | None = None        # "off" | "dos"
    magnetic: str | None = None          # "off" | "ihj"
    configurational: str | None = None   # "off" | "ideal" | "enumerated" | "monte_carlo"
    orientational: str | None = None     # "off" | "rotors"
    defects: str | None = None           # "off" | "pairs"

    def only(self, **on: str) -> "TermsCfg":
        """The listed terms on (with the given model), every other term off:  ``cfg.terms.only(configurational="ideal")``."""
        bad = [k for k in on if k not in TERM_KEYS]
        if bad:
            raise ValueError(f"unknown term(s) {bad}; terms: {list(TERM_KEYS)}")
        for k in TERM_KEYS:
            setattr(self, k, on.get(k, "off"))
        return self

    def show(self) -> None:
        """Print the selection with what it needs (same as ``heatup.core.contributions.show()``)."""
        from heatup.core import contributions
        contributions.show()


@dc.dataclass
class VibrationalCfg(Frozen):
    """The vibrational check (stability against soft / imaginary modes): WHICH evidence it reads, plans and plots.
    Choose any of ``"harmonic"`` (phonon bands / mesh at 0 K), ``"quasiharmonic"`` (the same at the volumes the crystal
    visits), ``"anharmonic"`` (VDOS of the MD runs).  They are listed in order of increasing physical completeness: when
    several are available the most complete one decides and the others are reported as information (a 0 K harmonic
    instability is not a failure if the finite-temperature evidence is clean)."""
    sources: list = dc.field(default_factory=lambda: ["harmonic", "anharmonic"])
    #: Save the figures of the vibrational analysis as PNG next to the data: ``phonons/anharmonic/<T>K/vdos.png`` (every VDOS), the figure of each level in its phonons folder and
    #: ``stability/vibrational.png`` (bands / volumes / VDOS of the selected sources, imaginary parts in red).  Switch off to
    #: analyse without matplotlib.  Display-only: never makes a result stale.
    save_figures: bool = True


@dc.dataclass
class PlanCfg(Frozen):
    """WHAT notebook 2 will compute for every phase (notebook 1).  Ask for gates and everything they need is
    planned; the settings of each calculation are the ``relax/elastic/harmonic/quasiharmonic/perturbative/md`` sections."""
    #: What you ask for: the gates and/or ``"relax"``.  The gates decide the calculations (``ledger.steps_for``:
    #: mechanical -> elastic, structural_transition -> md, vibrational -> the steps of ``cfg.vibrational.sources``,
    #: thermodynamic -> the steps of the selected ``cfg.terms.vibrational`` model for target and competitors).  Leave ``"relax"`` out when the structures are already relaxed (other code/model): they
    #: are then used as given - only energy, forces and stress are evaluated (``relax.optimize = False``).
    gates: list = dc.field(default_factory=lambda: list(REQUESTS))
    #: Override the derived steps (names of ``ledger.STEPS``; ``relax`` is always added), e.g. to add
    #: ``quasiharmonic`` / ``perturbative`` or ``md`` for competitors.  None -> derived from ``gates``.
    target_steps: list | None = None
    competitor_steps: list | None = None
    #: MD temperatures of the targets, K.  None -> the ``temperature`` of each run-list entry.
    md_temperatures: list | None = None
    #: Opt-in cost saver: competitors whose 0 K enthalpy lies more than this (eV/atom) above the 0 K hull get no
    #: harmonic / MD steps and are left out of the hull.  None (default) -> every competitor is computed.
    prune_window_eV: float | None = None


@dc.dataclass
class PhasesCfg(Frozen):
    #: Where competing phases are looked for: "database" and "candidates" (always: they are the two roots) plus,
    #: optionally, "mp-api" (Materials Project; needs a key) and "generate" (random/AI structures).  The last two only
    #: *add structures to* ``candidates_root``.
    sources: list = dc.field(default_factory=lambda: ["database", "candidates", "mp-api"])
    #: NAME of the environment variable that holds the Materials Project API key.  The key itself is never part of the
    #: configuration: it would be copied into every snapshot, report and child-process environment.
    mp_api_key_env: str = "MP_API_KEY"
    #: query window on MP's own hull (eV/atom); None = no window
    mp_ehull_max_eV: float | None = 0.15
    #: None = all phases of every sub-system
    mp_max_per_chemsys: int | None = 25
    #: EVERYTHING that is computed: targets and competing phases alike (read/write).  A structure is copied here
    #: from ``candidates_root`` the moment it is needed; its plan, results and manifests live here.
    database_root: str = "database"
    #: STRUCTURES only: your own inputs, Materials Project downloads, generated candidates.  Nothing is ever
    #: computed here and existing structures are never modified.
    candidates_root: str = "input/candidates"


@dc.dataclass
class GenerationCfg(Frozen):
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


class Settings(Frozen):
    """Base of a package's global configuration object (HeatUp's ``cfg``, SSE-AL's ``cfg``): the ONE API both share.

    A configuration is a dataclass whose fields are *sections* (dataclasses of :class:`Frozen`, documented option by option
    with ``#:`` comments) and, optionally, a few plain top-level options.  What every configuration can do:

    ``to_dict()`` / ``snapshot()`` / ``to_json()``  the live values (``snapshot`` is JSON-safe);
    ``save(path)`` / ``load(path)``                  JSON file; a partial file is allowed, unknown keys raise;
    ``update_from_dict(d)``  ``get("md.nblock")``    set / read by section;
    ``override(**{"md.n_steps": 10})``               context manager: temporary change;
    ``reset()``                                      factory defaults;  ``show(*sections)``  print;  ``check()``  validate.

    A package can also *expose sections that belong to another package's configuration* (``_foreign``): they are looked up
    live, are saved and loaded with this configuration, and are THE objects of the other package - one definition.
    """

    #: top-level fields that are neither sections nor plain options (callables, ...): never serialised
    _NOT_SERIALISED: tuple = ()

    # -- which fields are what ------------------------------------------------------------------
    def _foreign(self) -> dict:
        """``{name: section}`` of sections that belong to another package's configuration (default: none)."""
        return {}

    def _sections(self) -> dict:
        s = {f.name: getattr(self, f.name) for f in dc.fields(self) if dc.is_dataclass(getattr(self, f.name))}
        s.update(self._foreign())
        return s

    def _scalars(self) -> dict:
        return {f.name: getattr(self, f.name) for f in dc.fields(self)
                if f.name not in self._NOT_SERIALISED and not dc.is_dataclass(getattr(self, f.name))
                and isinstance(getattr(self, f.name), (str, int, float, bool, type(None)))}

    # -- serialisation --------------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {**{n: dc.asdict(sec) for n, sec in self._sections().items()}, **self._scalars()}

    def snapshot(self) -> dict:
        return json.loads(json.dumps(self.to_dict(), default=_json_default))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=_json_default)

    def update_from_dict(self, d: dict) -> None:
        sections, scalars = self._sections(), self._scalars()
        for key, vals in d.items():
            if key in sections:
                section = sections[key]
                for k, v in vals.items():
                    cur = getattr(section, k)
                    if isinstance(cur, tuple) and isinstance(v, list):
                        v = tuple(v)
                    setattr(section, k, v)
            elif key in scalars:
                setattr(self, key, vals)
            else:
                raise KeyError(f"unknown config section {key!r}")

    def get(self, dotted: str) -> Any:
        sec, key = dotted.split(".", 1)
        return getattr(getattr(self, sec), key)

    def save(self, path: str) -> str:
        """Write the live values as JSON (e.g. next to a result, so it can be traced to the settings that made it)."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.snapshot(), fh, indent=2)
        return path

    def load(self, path: str) -> "Settings":
        """Apply a JSON written by :meth:`save` (or by hand; partial is fine)."""
        with open(path) as fh:
            self.update_from_dict(json.load(fh))
        return self

    def show(self, *sections: str) -> None:
        d = self.snapshot()
        print(json.dumps({s: d[s] for s in sections} if sections else d, indent=2))

    def check(self) -> "Settings":
        """Fail early on invalid values (override in a subclass); returns ``self``."""
        return self

    # -- changing --------------------------------------------------------------------------------
    @contextlib.contextmanager
    def override(self, **dotted: Any) -> Iterator["Settings"]:
        """Temporarily change options: ``with cfg.override(**{"md.n_steps": 10}): ...``"""
        saved = copy.deepcopy(self.to_dict())
        sections, scalars = self._sections(), self._scalars()
        extras = {f.name: getattr(self, f.name) for f in dc.fields(self) if f.name not in sections and f.name not in scalars}
        try:
            for k, v in dotted.items():
                if "." not in k:
                    setattr(self, k, v)
                    continue
                sec, key = k.split(".", 1)
                section = getattr(self, sec)
                if isinstance(getattr(section, key), tuple) and isinstance(v, list):    # JSON round trip
                    v = tuple(v)
                setattr(section, key, v)
            yield self
        finally:
            self.update_from_dict(saved)
            for n, v in extras.items():
                object.__setattr__(self, n, v)

    def reset(self) -> "Settings":
        """Restore the factory defaults."""
        fresh = type(self)()
        for f in dc.fields(self):
            object.__setattr__(self, f.name, getattr(fresh, f.name))
        return self


@dc.dataclass
class Config(Settings):
    calc: CalcCfg = dc.field(default_factory=CalcCfg)
    relax: RelaxCfg = dc.field(default_factory=RelaxCfg)
    harmonic: HarmonicCfg = dc.field(default_factory=HarmonicCfg)
    quasiharmonic: QuasiHarmonicCfg = dc.field(default_factory=QuasiHarmonicCfg)
    perturbative: PerturbativeCfg = dc.field(default_factory=PerturbativeCfg)
    elastic: ElasticCfg = dc.field(default_factory=ElasticCfg)
    md: MDCfg = dc.field(default_factory=MDCfg)
    plan: PlanCfg = dc.field(default_factory=PlanCfg)
    analysis: AnalysisCfg = dc.field(default_factory=AnalysisCfg)
    gates: GateCfg = dc.field(default_factory=GateCfg)
    run: RunCfg = dc.field(default_factory=RunCfg)
    hull: HullCfg = dc.field(default_factory=HullCfg)
    terms: TermsCfg = dc.field(default_factory=TermsCfg)
    vibrational: VibrationalCfg = dc.field(default_factory=VibrationalCfg)
    phases: PhasesCfg = dc.field(default_factory=PhasesCfg)
    generation: GenerationCfg = dc.field(default_factory=GenerationCfg)
    #: Factory ``device -> ASE calculator`` for backend="custom" (not serialised).
    custom_calc_factory: Callable | None = None
    _NOT_SERIALISED = ("custom_calc_factory",)


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
    """Load a JSON file produced by ``cfg.save()`` / hand-written (partial allowed)."""
    cfg.load(path)


def override(**dotted: Any):
    """Temporarily change options: ``with override(**{"md.n_steps": 10}): ...``  (same as ``cfg.override``)."""
    return cfg.override(**dotted)


def reset() -> None:
    """Restore factory defaults (used by tests)."""
    cfg.reset()
