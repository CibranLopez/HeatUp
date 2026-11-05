"""heatup.core.contributions  --  the ONE place where the terms of the hull's free energy are declared and selected
=================================================================================================================
    G(T) = H0  +  sum over the SELECTED terms  [ F_term(T) ]                 (H0 = relaxed 0 K enthalpy: always there)

Every term is a row of ``TERMS`` with a handful of selectable MODELS.  The selection is one string per term::

    cfg.terms.vibrational     = "QHA"            # "off" | "ZPE" | "HA" | "QHA" | "VDOS"
    cfg.terms.anharmonic      = "off"            # "off" | "isochore"
    cfg.terms.electronic      = "off"            # "off" | "dos"
    cfg.terms.magnetic        = "off"            # "off" | "ihj"
    cfg.terms.configurational = "ideal"          # "off" | "ideal" | "enumerated" | "monte_carlo"
    cfg.terms.orientational   = "off"            # "off" | "rotors"
    cfg.terms.defects         = "off"            # "off" | "pairs"
    cfg.terms.only(configurational="ideal")      # ONE call: these on, everything else off  (a hull of configurational entropy only)

``None`` (the default of every field) means *as prepared in notebook 1* (stored in each phase's ``plan.json``), and if
nothing was prepared, the term's default.  So notebook 1 chooses, notebook 3 follows - or overrides any single term.

What a selection drives (all from this table, so no term is special):
    * notebook 1 / ``plan.py``   -> the compute steps every phase (target AND competitors) needs: ``required_steps``
    * notebook 2 / ``execute.py``-> runs those steps (settings: the config sections listed per model)
    * notebook 3 / ``thermo.py`` -> ``assemble`` builds the Gibbs assembler from the selected models only
    * ``check`` -> explicit double-counting / compatibility rules, enforced before anything is computed
    * ``show`` / ``inventory``   -> what is selected, what it needs, and which phases have it

Static energy and "finite-T energy": H0 is always the base.  The thermal part of the *energy* belongs to the nuclear
model, not to a separate switch: HA keeps the 0 K cell, QHA contains E0(V_T), the effective-harmonic route contains U0(T).
``vibrational = "off"`` is the static (0 K) hull; ``"ZPE"`` adds only the zero-point energy; at T = 0 every other term
vanishes.

Adding a term = adding one ``Term`` below (function, steps, files, settings, rules).  Nothing else needs to change.
"""
from __future__ import annotations

import dataclasses as dc
import importlib
import logging
import os
from typing import Callable

import numpy as np

from heatup.core.config import cfg

log = logging.getLogger("heatup")


# =================================================================================================================
# declarations
# =================================================================================================================

@dc.dataclass(frozen=True)
class Model:
    name: str
    func: str | None = None                     # "module:function"  ``fn(phase, T) -> eV/atom`` (None for "off")
    tag: str | None = None                      # name of the contribution in the results (default F_<term>)
    steps: tuple = ()                           # compute steps (``ledger.STEPS``) every phase needs for this model
    files: tuple = ()                           # data files (relative to the phase) the model reads; absent -> contributes 0
    settings: tuple = ()                        # config sections that parametrise it (shown by ``show``)
    requires: tuple = ()                        # ((term, (allowed models, ...)), ...)
    status: str = "ok"                          # "ok" | "experimental"
    doc: str = ""


@dc.dataclass(frozen=True)
class Term:
    key: str
    label: str
    default: str
    models: tuple
    doc: str = ""

    def model(self, name: str) -> Model:
        for m in self.models:
            if m.name.lower() == str(name).lower():
                return m
        raise ValueError(f"cfg.terms.{self.key} = {name!r}: choose from {[m.name for m in self.models]}")


_OFF = lambda why: Model("off", doc=why)

TERMS: dict[str, Term] = {t.key: t for t in [
    Term("vibrational", "Lattice vibrations (nuclear free energy)", "HA", (
        _OFF("static hull: G = H0 (relaxed 0 K enthalpy)"),
        Model("ZPE", "heatup.analyze.thermo:f_vib_zpe", "F_vib[ZPE]", steps=("phonons",), settings=("phonon",),
              doc="zero-point energy of the harmonic phonons, temperature independent"),
        Model("HA", "heatup.analyze.thermo:f_vib_ha", "F_vib[HA]", steps=("phonons",), settings=("phonon",),
              doc="harmonic phonons at the 0 K cell"),
        Model("QHA", "heatup.analyze.thermo:f_vib_qha", "F_vib[QHA]", steps=("qha",), settings=("phonon", "qha"),
              doc="quasi-harmonic: E0(V) + F_vib(V,T), volume minimised at every T (contains thermal expansion)"),
        Model("VDOS", "heatup.analyze.thermo:f_vib_vdos", "F_vib[VDOS]", steps=("md",), settings=("md", "analysis"),
              status="experimental",
              doc="effective-harmonic F_vib from the MD velocity spectrum; NOT validated (VACF bias, finite q-mesh, "
                  "see heatup_terms/validation)"),
    ), "The nuclear free energy: ONE model per phase; they approximate the same quantity and replace each other."),

    Term("anharmonic", "Explicit anharmonic free energy (classical, fixed cell)", "off", (
        _OFF("harmonic (or quasi-harmonic) only"),
        Model("isochore", "heatup.compute.anharmonic:f_anh", "F_anh", steps=("anharmonic",), settings=("anharmonic",),
              requires=(("vibrational", ("QHA",)),),
              doc="F_true - F_harm at the QHA cell from NVT runs, temperature integration of the equipartition estimator"),
    ), "Added to QHA only: it is defined relative to the harmonic crystal at the same cell."),

    Term("electronic", "Electronic excitations (Mermin)", "dos", (
        _OFF("no electronic entropy"),
        Model("dos", "heatup.analyze.thermo:f_el", "F_el", files=("electronic/edos.json",),
              doc="F_el(T) - F_el(0) from the electronic DOS at fixed ions; metals only"),
    ), "Negligible for gapped materials; needs a DOS (DFT), an MLIP has none."),

    Term("magnetic", "Magnetic entropy (Inden-Hillert-Jarl, thermal part)", "ihj", (
        _OFF("no magnetic term"),
        Model("ihj", "heatup.analyze.thermo:f_mag", "F_mag", files=("magnetic/moments.json",),
              doc="IHJ thermal magnetic free energy with S(0) = 0; needs Tc and beta per phase"),
    ), "Empirical (CALPHAD) model; competitors need their own Tc and beta."),

    Term("configurational", "Configurational disorder", "ideal", (
        _OFF("ordered phases only"),
        Model("ideal", "heatup.analyze.thermo:f_conf", "F_conf", files=("disorder/site_occupancies.json",),
              doc="-T S of ideal mixing on the sites (upper bound)"),
        Model("enumerated", "heatup.analyze.ensembles:f_conf_enumerated", "F_conf[enumerated]",
              files=("disorder/configurations.json",),
              doc="Boltzmann ensemble of enumerated configurations of one cell, each with its own F_vib"),
        Model("monte_carlo", "heatup.analyze.ensembles:f_conf_monte_carlo", "F_conf[monte_carlo]",
              files=("disorder/monte_carlo.json",),
              doc="lattice-model Monte Carlo (cluster expansion) free energy by integration of <E> over beta"),
    ), "ONE model: they describe the same degrees of freedom."),

    Term("orientational", "Orientational disorder / hindered rotors", "off", (
        _OFF("molecular units are static"),
        Model("rotors", "heatup.analyze.ensembles:f_orientational", "F_orientational", files=("disorder/rotors.json",),
              doc="hindered rotors (quantum levels) and independent discrete orientations"),
    ), ""),

    Term("defects", "Point defects (composition-conserving pairs)", "off", (
        _OFF("perfect crystal"),
        Model("pairs", "heatup.analyze.ensembles:f_defects", "F_defects", files=("disorder/defects.json",),
              doc="Frenkel/Schottky/antisite pairs, dilute and non-interacting"),
    ), ""),
]}


# =================================================================================================================
# selection
# =================================================================================================================

#: legacy ``cfg.hull`` switches (kept working): ``False`` means "off" when ``cfg.terms.<term>`` is None
_LEGACY_OFF = {"electronic": "include_electronic", "magnetic": "include_magnetic",
               "configurational": "include_configurational"}


def resolve(phase=None, **explicit) -> dict[str, str]:
    """The model chosen for every term:  explicit argument > ``cfg.terms.<term>`` > legacy ``cfg.hull`` switch >
    what *phase* was prepared with (``plan.json``) > the term's default."""
    opts: dict = {}
    if phase is not None:
        from heatup.core import ledger
        plan = ledger.read_plan(phase)
        opts = plan.options if plan else {}
    unknown = [k for k in explicit if k not in TERMS]
    if unknown:
        raise ValueError(f"unknown term(s) {unknown}; terms: {list(TERMS)}")
    sel: dict[str, str] = {}
    for key, term in TERMS.items():
        v = explicit.get(key)
        if v is None:
            v = getattr(cfg.terms, key)
        if v is None and key == "vibrational":
            v = cfg.hull.phonon_method
        if v is None and key in _LEGACY_OFF and getattr(cfg.hull, _LEGACY_OFF[key]) is False:
            v = "off"
        if v is None:
            v = opts.get(f"terms.{key}")
        if v is None and key == "vibrational":
            v = opts.get("hull.phonon_method")
        sel[key] = term.model(term.default if v is None else v).name
    return sel


def project_selection(phases=None) -> tuple[dict, dict]:
    """``(selection, disagreements)`` for a whole database: every phase resolves ITS OWN selection (so a term left at
    None follows what that phase was prepared with); ``disagreements = {term: [models]}`` lists the terms on which
    phases differ - such phases must not be mixed in one hull (``ledger.plan_conflicts`` refuses them)."""
    from heatup.core.layout import PhaseDir, iter_phase_dirs
    ph = [PhaseDir.of(p) for p in phases] if phases is not None else list(iter_phase_dirs(cfg.phases.database_root))
    sels = [resolve(p) for p in ph] or [resolve()]
    diff = {k: sorted({s[k] for s in sels}) for k in TERMS if len({s[k] for s in sels}) > 1}
    return dict(sels[0]), diff


def model_of(key: str, name: str) -> Model:
    return TERMS[key].model(name)


def plan_options(sel: dict | None = None) -> dict:
    """``{"terms.<term>": model}`` stored in every ``plan.json`` ('as prepared'; also valid config keys)."""
    sel = sel or resolve()
    return {f"terms.{k}": v for k, v in sel.items()}


# =================================================================================================================
# rules
# =================================================================================================================

def check(sel: dict | None = None) -> list[tuple[str, str]]:
    """``[(level, message)]``: ``"error"`` (cannot be assembled) | ``"warn"`` (allowed, logged) | ``"note"`` (shown by ``show``)."""
    sel = sel or resolve()
    out: list[tuple[str, str]] = []
    for key, name in sel.items():
        m = model_of(key, name)
        for term, allowed in m.requires:
            if sel[term] not in allowed:
                out.append(("error", f"{key}={name} requires {term} in {list(allowed)} (selected: {sel[term]})"))
        if m.status == "experimental":
            out.append(("warn", f"{key}={name} is experimental: {m.doc}"))
    if sel["orientational"] != "off" and sel["configurational"] in ("enumerated", "monte_carlo"):
        out.append(("warn", "orientational + configurational: count each orientation ONCE (rotors.json "
                            "'orientational_entropy': false, or leave the orientations out of the configurations)"))
    if sel["orientational"] != "off" and sel["vibrational"] == "off":
        out.append(("warn", "orientational with vibrational=off: a rotor subtracts the harmonic libration that "
                            "F_vib would contain - set 'remove_libration': false in rotors.json"))
    if sel["configurational"] == "ideal" and sel["defects"] != "off":
        out.append(("warn", "ideal mixing + defects: vacancies/interstitials must not be both site occupancies and defects"))
    if sel["electronic"] != "off" and sel["magnetic"] != "off":
        out.append(("note", "electronic + magnetic: a DOS of the ordered state already contains Stoner-type "
                            "excitations; use one source for the longitudinal spin part"))
    return out


def raise_on_errors(sel: dict | None = None) -> None:
    errs = [m for lvl, m in check(sel) if lvl == "error"]
    if errs:
        raise ValueError("incompatible terms:\n  " + "\n  ".join(errs))


# =================================================================================================================
# what a selection needs / builds
# =================================================================================================================

def required_steps(sel: dict | None = None, phase=None, exclude=()) -> tuple:
    """Compute steps (``ledger.COMPUTE_STEPS`` order) the selected models need - for the target AND every competitor."""
    from heatup.core import ledger
    sel = sel or resolve(phase)
    need: set = set()
    for key, name in sel.items():
        if key not in exclude:
            need.update(model_of(key, name).steps)
    return tuple(s for s in ledger.COMPUTE_STEPS if s in need)


def _load(path: str) -> Callable:
    mod, fn = path.split(":")
    return getattr(importlib.import_module(mod), fn)


def assemble(sel: dict | None = None, phase=None, drop_invalid: bool = False):
    """A ``GibbsAssembler`` with exactly the selected models.  ``drop_invalid`` skips (with a warning) the terms whose
    requirements are not met instead of raising - used for diagnostics that force another vibrational method."""
    from heatup.analyze.thermo import GibbsAssembler
    sel = dict(sel or resolve(phase))
    for lvl, msg in check(sel):
        if lvl == "warn":
            log.warning("terms: %s", msg)
    if drop_invalid:
        for key, name in list(sel.items()):
            for term, allowed in model_of(key, name).requires:
                if sel[term] not in allowed:
                    log.warning("terms: %s=%s dropped (requires %s in %s)", key, name, term, list(allowed))
                    sel[key] = "off"
    raise_on_errors(sel)
    asm = GibbsAssembler()
    for key, name in sel.items():
        m = model_of(key, name)
        if m.func:
            asm.register(m.tag or f"F_{key}", _load(m.func))
    return asm


# =================================================================================================================
# reporting (notebooks)
# =================================================================================================================

def _section(name: str) -> dict:
    return dc.asdict(getattr(cfg, name))


def describe(sel: dict | None = None, phase=None) -> list[dict]:
    """One row per term: selection, what it needs, its settings sections (feed to ``pandas.DataFrame``)."""
    sel = sel or resolve(phase)
    rows = []
    for key, term in TERMS.items():
        m = model_of(key, sel[key])
        rows.append({"term": key, "model": m.name, "options": " | ".join(x.name for x in term.models),
                     "compute steps": ", ".join(m.steps) or "-", "data files": ", ".join(m.files) or "-",
                     "settings": ", ".join(f"cfg.{s}" for s in m.settings) or "-", "status": m.status,
                     "meaning": m.doc})
    return rows


def show(sel: dict | None = None, phase=None, settings: bool = True) -> None:
    """Print the selection, its compute steps, the rules that apply and the current values of the settings."""
    diff: dict = {}
    if sel is None:
        sel, diff = (resolve(phase), {}) if phase is not None else project_selection()
    print("selected terms (cfg.terms; None = as prepared in notebook 1, else the default)")
    for r in describe(sel):
        on = "  " if r["model"] != "off" else "  "
        print(f"{on}{r['term']:16s} = {r['model']:12s} [{r['options']}]")
    print("compute steps for every phase:", ", ".join(("relax",) + required_steps(sel)))
    for term, models in diff.items():
        print(f"  WARN: phases were prepared with different {term} models {models}: they cannot share one hull")
    for lvl, msg in check(sel):
        print(f"  {lvl.upper()}: {msg}")
    if settings:
        for key, name in sel.items():
            for s in model_of(key, name).settings:
                print(f"  cfg.{s} (for {key}={name}):", {k: v for k, v in _section(s).items()})


def inventory(phases) -> list[dict]:
    """For every phase and selected term: are its compute steps done, does it have its data file?  A hull is only
    consistent if every phase that should carry a term has it: mixed presence is flagged ``inconsistent``."""
    from heatup.core import ledger
    from heatup.core.layout import PhaseDir
    rows = []
    for p in phases:
        ph = PhaseDir.of(p)
        for key, name in resolve(ph).items():                       # each phase as it was prepared
            m = model_of(key, name)
            if name == "off":
                continue
            steps_ok = True
            for s in m.steps:
                if ledger.STEPS[s].per_T:
                    steps_ok &= bool(ledger.md_temperatures(ph))
                else:
                    steps_ok &= ledger.step_state(ph, s, None, calc=False)[0] in ledger.GOOD_STATES
            has = [os.path.exists(ph.join(*f.split("/"))) for f in m.files]
            rows.append({"phase": ph.tag, "term": key, "model": name, "steps done": steps_ok,
                         "data": (all(has) if has else None)})
    for key in {r["term"] for r in rows}:
        d = [r["data"] for r in rows if r["term"] == key and r["data"] is not None]
        if d and any(d) and not all(d):
            for r in rows:
                if r["term"] == key:
                    r["inconsistent"] = True
    return rows
