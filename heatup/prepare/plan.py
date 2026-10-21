"""heatup.prepare.plan
======================
Notebook 1: put every structure where it is computed and write down what is to be computed for it.

``prepare_project(run_list)``
    1. *targets*: every ``{"material", "symmetry", "temperature"}`` entry of the run list is located: a POSCAR in
       ``database_root`` is used as is; one found only in ``candidates_root`` is copied to the database;
    2. *competing phases*: for every chemical subsystem of the targets, looked up in the **database** (anything already
       there) and in **candidates_root** (yours, Materials Project, generated - see ``cfg.phases.sources``).
       Those from the candidates are copied into the database.  A phase that is already in the database is reused
       **only if its plan has the settings this project wants**; otherwise it is left untouched, reported in
       ``Prepared.refused`` with a warning, and never used in a hull (compare ``ledger.plan_conflicts``);
    3. a ``plan.json`` next to every POSCAR (:class:`heatup.core.ledger.Plan`): the gates you asked for
       (``cfg.plan.gates``) and the steps they need (or ``cfg.plan.target_steps`` / ``competitor_steps`` if you list
       them), the temperatures, and the resolved options of those steps.  A competitor that already has a plan keeps
       it and only gains the steps it lacks.  From then on the folders are self-describing: notebook 2 executes the
       plans, notebook 3 judges the results against them, and neither needs the calculation settings again.

Temperatures: ``cfg.plan.md_temperatures`` (if set) gives every target's MD runs, else the ``temperature`` of its run-list
entry (scalar or list).  The operating temperature (where Gate 3 judges the phase) is the entry's ``operating_T`` if
given, else its first temperature, else ``cfg.hull.operating_T``.

Nothing here needs a calculator or a GPU (the Materials-Project query needs ``MP_API_KEY`` and a network).
"""

from __future__ import annotations

import dataclasses as dc
import json
import logging
import os

from heatup.core import ledger
from heatup.core.config import AUTO_MIN_LENGTH_A, cfg
from heatup.core.layout import PhaseDir, iter_phase_dirs, read_structure, resolve_supercell, upper_triangular
from heatup.prepare.phases import Phase, find_reference_phases, locate_phase, phase_from_dir, same_structure

log = logging.getLogger("heatup")


@dc.dataclass
class Prepared:
    targets: list = dc.field(default_factory=list)       # PhaseDir
    competitors: list = dc.field(default_factory=list)   # Phase (in database_root, planned)
    missing: list = dc.field(default_factory=list)       # "material/label" without a structure anywhere
    refused: dict = dc.field(default_factory=dict)       # competitor tag -> {option: (existing, wanted)}: other settings


def _steps(names, what: str) -> list[str]:
    bad = [s for s in names if s not in ledger.COMPUTE_STEPS]
    if bad:
        raise ValueError(f"cfg.plan.{what}: unknown step(s) {bad}; choose from {list(ledger.COMPUTE_STEPS)}")
    keep = [s for s in ledger.COMPUTE_STEPS if s in names or s == "relax"]       # everything starts with the relaxation
    if "fc3" in keep and cfg.phonon.fc_order != 3:
        log.warning("'fc3' is planned but cfg.phonon.fc_order = %s: set it to 3", cfg.phonon.fc_order)
    return keep


def make_plan(role: str, source: str, steps: list[str], operating_T: float | None = None,
              md_temperatures: list[float] | None = None, gates: list[str] | None = None) -> ledger.Plan:
    opts = ledger.plan_options(steps)
    opts["plan.prune_window_eV"] = cfg.plan.prune_window_eV              # project-level; analysis reads it from here
    md_T = sorted({float(t) for t in (md_temperatures or [])}) if "md" in steps else []
    return ledger.Plan(role=role, steps=list(steps), gates=list(gates or []), operating_T_K=operating_T,
                       md_temperatures_K=md_T, options=opts, source=source)


def prepare_phase(phase_dir, role: str = "target", steps: list[str] | None = None, operating_T: float | None = None,
                  md_temperatures: list[float] | None = None, source: str = "database",
                  gates: list[str] | None = None) -> ledger.Plan:
    """Write ``plan.json`` of one phase (the building block of :func:`prepare_project` and of ``heatup.screen``).
    Steps: *steps* > ``cfg.plan.<role>_steps`` > derived from the gates (*gates* > ``cfg.plan.gates``)."""
    phase = PhaseDir.of(phase_dir)
    gates = list(cfg.plan.gates if gates is None else gates)
    if steps is None:
        steps = cfg.plan.target_steps if role == "target" else cfg.plan.competitor_steps
    if steps is None:
        steps = ledger.steps_for(gates, "HA", role)
    if role == "target":
        steps = _steps(steps, "target_steps")
        op = float(cfg.hull.operating_T if operating_T is None else operating_T)
        temps = md_temperatures if md_temperatures is not None else (cfg.plan.md_temperatures or [op])
    else:
        steps = _steps(steps, "competitor_steps")
        op, temps = None, (md_temperatures or [])
        if "md" in steps and not temps:
            log.warning("competitor step 'md' planned without temperatures for %s", phase.tag)
    plan = make_plan(role, source, steps, op, temps, gates)
    ledger.write_plan(phase, plan)
    return plan


def _parse_entry(e: dict) -> tuple[str, str, list[float], float | None]:
    t = e.get("temperature")
    temps = [float(x) for x in (t if isinstance(t, (list, tuple)) else [t])] if t is not None else []
    op = e.get("operating_T", temps[0] if temps else None)
    return e["material"], e["symmetry"], temps, None if op is None else float(op)


def prepare_project(run_list, competitors: bool = True, refresh_mp: bool = False) -> Prepared:
    """Targets + competing phases + ``plan.json`` for all of them.  ``run_list``: a list of dicts or a JSON file."""
    if isinstance(run_list, (str, os.PathLike)):
        run_list = json.load(open(run_list))
    out = Prepared()
    for e in run_list:
        mat, label, temps, op = _parse_entry(e)
        d = locate_phase(mat, label)
        if d is None:
            out.missing.append(f"{mat}/{label}")
            log.warning("[missing] %s/%s: put a POSCAR in %s/%s/%s/", mat, label, cfg.phases.database_root, mat, label)
            continue
        T_all = [float(t) for t in cfg.plan.md_temperatures] if cfg.plan.md_temperatures is not None else temps
        plan = prepare_phase(d, "target", operating_T=op, md_temperatures=T_all or None,
                             source="database")
        if not (min(cfg.hull.temperatures) <= plan.operating_T_K <= max(cfg.hull.temperatures)):
            log.warning("%s: operating T = %g K lies outside cfg.hull.temperatures", d.tag, plan.operating_T_K)
        out.targets.append(d)
    if competitors and out.targets and ("thermodynamic" in cfg.plan.gates or cfg.plan.competitor_steps is not None):
        out.competitors = prepare_competitors(out.targets, refresh_mp=refresh_mp, refused=out.refused)
        log.info("%d target(s), %d competing phase(s) prepared", len(out.targets), len(out.competitors))
    return out


def _refusal(ph: Phase, conflicts: dict) -> str:
    what = ", ".join(f"{k}: {a} -> {b}" for k, (a, b) in list(conflicts.items())[:4])
    return (f"[reuse refused] {ph.tag}: already in the database with other settings ({what}) - its results are not used. "
            "Delete its folder to recompute it with the current settings.")


def prepare_competitors(targets, refresh_mp: bool = False, steps: list[str] | None = None,
                        refused: dict | None = None) -> list[Phase]:
    """Competing phases of *targets* (database + candidates + optional MP / generated), all in ``database_root``
    with a ``plan.json`` (``steps`` default: ``cfg.plan.competitor_steps``, else derived from ``cfg.plan.gates``).
    Phases already in the database keep their plan (and gain missing steps) if their settings match; otherwise they are
    refused (warning, entry in *refused*) and left exactly as they are."""
    tph = [p for p in (phase_from_dir(t, "target") for t in targets) if p]
    if not tph:
        return []
    elements = frozenset().union(*(p.elements for p in tph))
    refs = find_reference_phases(elements, exclude=[PhaseDir.of(t) for t in targets], refresh_mp=refresh_mp)
    refs = [r for r in refs if not any(r.formula == t.formula and same_structure(r.phase_dir, t.phase_dir, use_relaxed=False)
                                       for t in tph)]                    # the target's own structure is no competitor
    gates = list(cfg.plan.gates)
    names = steps if steps is not None else (cfg.plan.competitor_steps if cfg.plan.competitor_steps is not None
                                              else ledger.steps_for(gates, "HA", "competitor"))
    wanted_steps = _steps(names, "competitor_steps")
    out = []
    for ph in refs:
        wanted = make_plan("competitor", ph.source, wanted_steps, None, cfg.plan.md_temperatures, gates)
        existing = ledger.read_plan(ph.phase_dir)
        merged = ledger.merge_plans(existing, wanted) if existing else wanted
        conflicts = ledger.plan_conflicts(existing, wanted.steps, wanted.options) if existing else {}
        if not conflicts:                                    # results made with other settings (no plan to compare)?
            with ledger.applied(merged):
                stale = [s for s, T in merged.tasks() if ledger.step_state(ph.phase_dir, s, T, calc=False)[0] == ledger.STALE]
            conflicts = {f"(results of {s})": ("other settings", "current") for s in stale}
        if conflicts:
            log.warning(_refusal(ph, conflicts))
            if refused is not None:
                refused[ph.tag] = conflicts
            continue
        ledger.write_plan(ph.phase_dir, merged)
        out.append(ph)
    return out


# ---------------------------------------------------------------------------
# what the plan will cost
# ---------------------------------------------------------------------------

def supercell_preview(phase_dir, steps=None) -> dict[str, str]:
    """Supercells the planned steps will build (from the *unrelaxed* structure: sizes may change slightly)."""
    phase = PhaseDir.of(phase_dir)
    plan = ledger.read_plan(phase)
    steps = steps if steps is not None else (plan.steps if plan else [])
    atoms = read_structure(phase.poscar)
    cell, n = atoms.cell.array, len(atoms)
    cases = {"phonons": ("phonon", cfg.phonon.supercell, cfg.phonon.min_length_A, cfg.phonon.max_atoms),
             "fc3": ("fc3", cfg.fc3.supercell, cfg.fc3.min_length_A, cfg.fc3.max_atoms),
             "md": ("md", cfg.md.supercell, cfg.md.min_cell_A, cfg.md.max_atoms)}
    out: dict[str, str] = {}
    for step, (key, sc, ml, mx) in cases.items():
        if step not in steps:
            continue
        cell_ = upper_triangular(atoms).cell.array if step == "md" else cell
        try:
            reps = resolve_supercell(cell_, n, supercell=sc, min_length=ml, max_atoms=mx,
                                     auto_min_length=AUTO_MIN_LENGTH_A.get(key, AUTO_MIN_LENGTH_A["phonon"]),
                                     what=f"{step} supercell")
            out[step] = "x".join(map(str, reps)) + f" ({n * reps[0] * reps[1] * reps[2]} atoms)"
        except ValueError as exc:
            out[step] = f"ERROR: {exc}"
    return out


def preview_plan(prepared: Prepared | None = None) -> list[dict]:
    """One row per phase of the project: role, planned steps, temperatures, supercells.  With no argument: every
    phase of ``database_root`` that has a ``plan.json``."""
    from heatup.core.layout import iter_phase_dirs
    if prepared is not None:
        dirs = list(prepared.targets) + [c.phase_dir for c in prepared.competitors]
    else:
        dirs = list(iter_phase_dirs(cfg.phases.database_root))
    rows = []
    for d in dirs:
        plan = ledger.read_plan(d)
        if plan is None:
            continue
        rows.append({"phase": d.tag, "role": plan.role, "source": plan.source, "steps": ", ".join(plan.steps),
                     "operating_T_K": plan.operating_T_K, "md_T_K": plan.md_temperatures_K or "",
                     **{f"{k} cell": v for k, v in supercell_preview(d, plan.steps).items()}})
    return rows


def save_project(path: str = "heatup_project.json") -> str:
    """Write the folder roots to *path*; notebooks 02/03 read it with ``heatup.load_json`` (so the roots are typed once)."""
    from heatup.core.layout import write_json
    p = cfg.phases
    return write_json(path, {"phases": {"database_root": p.database_root, "candidates_root": p.candidates_root}})
