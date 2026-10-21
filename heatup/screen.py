"""heatup.screen
================
The stable entry point for other software: :func:`screen` (one phase) and :func:`screen_many` (a funnel over
many candidates).  Both *compose* the three stages - prepare (plan), compute (``run_plan``), analyze (``evaluate``) -
for callers that want one call instead of three notebooks.

::

    from heatup import screen, ScreenOptions
    r = screen("work/BaTiO3/P4mm", stage="pre_md",
               options=ScreenOptions(operating_T=600, mobile_species=["Li"],
                                     thresholds={"hull_fail_eV": 0.08}))
    r.verdict      # "accept" | "reject" | "inconclusive"
    r.reasons      # list[str], human readable
    r.e_hull_eV    # 0 K hull energy (pre_md) / hull energy at operating T (post_md), eV/atom

Stages
------
``pre_md``   relax -> elastic (Gate 1) and the 0 K hull (Gate 3 at 0 K) - each only if enabled, and
             independently of each other.  No phonons, no MD: minutes per phase.
             ``reject``: Born-unstable, or 0 K hull energy > ``pre_md_hull_reject_eV``
             (default ``gates.hull_fail_eV``).  ``accept``: "worth paying for MD".
``post_md``  all four gates (1, then 0 and 2 from the MD, then the T-dependent hull).
             ``reject``: any gate failed.  ``accept``: every gate ok.

``inconclusive`` = the evidence does not decide: warnings (unless ``accept_warn``), missing
data (hull undefined, no MD yet, a step failed) - the reasons say which.

Guarantees
----------
* **Gates are independent.**  The gates to run come from ``ScreenOptions.gates`` / ``cfg.gates.enabled`` (default: all four);
  every enabled gate is evaluated whatever the others say (a Born-unstable phase still gets its MD,
  vibrational and hull results), and the verdict is decided from all of them.  ``force`` recomputes
  the named steps (or ``True`` = all) instead of reading their caches.  The steps that get computed are derived from
  the enabled gates (``ledger.steps_for``); nothing is computed with ``compute=False``.
* **Per-stage state.**  Reports are written to ``<phase>/stability/report_<stage>.json`` and the
  typed result to ``result_<stage>.json``; a stage never reads or overwrites another stage's
  files, and no report is ever *returned from cache*: verdicts are recomputed from the cached
  *step* outputs (relax / elastic / phonons / md / vdos), each of which is invalidated by its own
  manifest (config fingerprint + input hashes).  So a pre_md call never masks a later post_md one,
  and a changed option or structure is never answered with a stale verdict.
* **Options are arguments.**  Everything is passed through :class:`ScreenOptions` (mobile
  species, thresholds, model, roots, any ``section.key`` of the config via ``config={...}``).
  They are applied for the duration of the call only and restored afterwards; the process-wide
  ``cfg`` is not modified.  Calls are serialised with a lock (the config is process-global
  state); run several phases in parallel with several processes.
* **Never raises for compute problems** (failed step, missing data): returns ``inconclusive``
  with the reason.  Raises ``ValueError`` for invalid arguments (stage, unknown option name).
"""

from __future__ import annotations

import dataclasses as dc
import logging
import os
import threading
import traceback
from typing import Any, Literal

from heatup.core import ledger
from heatup.core.config import (GATES, STATUS_FAIL, STATUS_MISSING, STATUS_OK, STATUS_WARN, cfg, normalize_force,
                                override)
from heatup.core.layout import PhaseDir, write_json

log = logging.getLogger("heatup")
_LOCK = threading.RLock()

Stage = Literal["pre_md", "post_md"]
Verdict = Literal["accept", "reject", "inconclusive"]
STAGES = ("pre_md", "post_md")


# ---------------------------------------------------------------------------
# public types
# ---------------------------------------------------------------------------

@dc.dataclass
class ScreenOptions:
    """Everything a caller may want to vary; ``None`` = keep the HeatUp default."""
    operating_T: float | None = None                 # K, temperature of the hull judgement
    md_temperatures: list[float] | None = None       # K, MD runs evaluated by Gates 0/2 (default [operating_T])
    mobile_species: list[str] | None = None          # e.g. ["Li"]: allowed to diffuse (Gates 0 and 2)
    thresholds: dict[str, float] | None = None       # gate thresholds, e.g. {"hull_ok_eV": 0.02}
    pre_md_hull_reject_eV: float | None = None       # pre_md: reject above this 0 K hull energy
    accept_warn: bool = False                        # treat "warn" as acceptable
    gates: list[str] | None = None                   # which gates to evaluate (default: all four);
                                                     # any of mechanical/structural_transition/vibrational/thermodynamic
    # calculator
    backend: str | None = None
    model: str | None = None                         # THE model: mace_mp name or .model path
    model_overrides: dict[str, str] | None = None    # material folder -> model path
    device: str | None = None
    # trees
    database_root: str | None = None
    candidates_root: str | None = None               # structures (inputs, MP, generated); computed copies go to database_root
    sources: list[str] | None = None
    mp_api_key: str | None = None
    # run control
    compute: bool = True                             # False: evaluate cached results only
    force: bool | str | list[str] | None = None      # True/"all", or step names (step names of ledger.STEPS, or the group "analysis")
    save_figure: bool = False
    #: any other option as ``{"md.n_steps": 20000, "phonon.delta_A": 0.02}``
    config: dict[str, Any] | None = None


@dc.dataclass
class GateResult:
    name: str
    status: str                  # ok | warn | fail | missing
    message: str


@dc.dataclass
class ScreenResult:
    verdict: Verdict
    stage: Stage
    reasons: list[str]
    e_hull_eV: float | None                  # pre_md: 0 K; post_md: at operating_T (eV/atom, signed)
    e_hull_0K_eV: float | None
    operating_T_K: float
    gates: dict[str, GateResult]
    decomposition: dict[str, float]          # competing phases the target decomposes into (if above the hull)
    flags: list[str]
    phase: str                               # "<material>/<label>"
    phase_dir: str
    model: str
    report_file: str | None
    result_file: str | None
    complete: bool                           # every gate of this stage had its evidence

    def to_dict(self) -> dict:
        return dc.asdict(self)


# ---------------------------------------------------------------------------
# options -> config
# ---------------------------------------------------------------------------

_DIRECT = {"gates": "gates.enabled", "operating_T": "hull.operating_T", "mobile_species": "analysis.mobile_species",
           "backend": "calc.backend", "model": "calc.model", "model_overrides": "calc.model_overrides",
           "database_root": "phases.database_root", "candidates_root": "phases.candidates_root",
           "sources": "phases.sources", "mp_api_key": "phases.mp_api_key", "device": "calc.device"}


def _dotted(o: ScreenOptions) -> dict[str, Any]:
    d: dict[str, Any] = {}
    for f, key in _DIRECT.items():
        v = getattr(o, f)
        if v is not None:
            d[key] = v
    if o.force is not None:
        d["run.force"] = normalize_force(o.force)
    for k, v in (o.thresholds or {}).items():
        d[k if "." in k else f"gates.{k}"] = v
    for k, v in (o.config or {}).items():
        if "." not in k:
            raise ValueError(f"config key {k!r} must be 'section.option'")
        d[k] = v
    for k in d:                                                    # unknown names fail early, before any work
        sec, key = k.split(".", 1)
        if not hasattr(cfg, sec) or not hasattr(getattr(cfg, sec), key):
            raise ValueError(f"unknown option {k!r}")
    return d


# ---------------------------------------------------------------------------
# verdict logic (pure; unit-tested)
# ---------------------------------------------------------------------------

def decide(stage: str, gates: dict[str, GateResult], e_hull_0K: float | None, hull_available: bool,
           *, hull_reject_eV: float, accept_warn: bool, errors: list[str]) -> tuple[Verdict, list[str]]:
    """(verdict, reasons) from gate results.  Fail anywhere -> reject; otherwise all-ok -> accept."""
    if not gates and not errors:
        return "inconclusive", ["no gate was evaluated (check cfg.gates.enabled / ScreenOptions.gates)"]
    fails = [f"{g.name}: {g.message}" for g in gates.values() if g.status == STATUS_FAIL]
    if stage == "pre_md" and hull_available and e_hull_0K is not None and e_hull_0K > hull_reject_eV:
        fails.append(f"0 K hull energy {1e3 * e_hull_0K:.0f} meV/atom > {1e3 * hull_reject_eV:.0f} meV/atom")
    if fails:
        return "reject", fails
    reasons: list[str] = list(errors)
    for g in gates.values():
        if g.message and any(g.message in e for e in errors):          # already reported as an error
            continue
        if g.status == STATUS_MISSING:
            reasons.append(f"{g.name}: {g.message or 'not evaluated'}")
        elif g.status == STATUS_WARN:
            reasons.append(f"{g.name}: {g.message}")
    unresolved = bool(errors) or any(g.status == STATUS_MISSING for g in gates.values())
    warned = any(g.status == STATUS_WARN for g in gates.values())
    if unresolved or (warned and not accept_warn):
        return "inconclusive", reasons
    return "accept", reasons or ["all evaluated gates passed"]


def _gate(name: str, r: dict | None) -> GateResult:
    if not r:
        return GateResult(name, STATUS_MISSING, "not evaluated")
    return GateResult(name, r.get("status", STATUS_MISSING), r.get("message", ""))


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------

def _gates() -> list[str]:
    """Gates of this call: ``ScreenOptions.gates`` / ``cfg.gates.enabled``, default all four."""
    return list(cfg.gates.enabled or GATES)


def _plan_and_run(phase: PhaseDir, stage: str, o: ScreenOptions, T_op: float) -> list[str]:
    """Plan the steps the enabled gates need (target + competitors), run them; returns error messages."""
    from heatup.compute.execute import run_plan
    from heatup.prepare.plan import prepare_competitors, prepare_phase

    enabled, method = _gates(), cfg.hull.phonon_method
    if stage == "pre_md":
        tsteps = ["relax"] + (["elastic"] if "mechanical" in enabled else [])
        csteps = ["relax"]
    else:
        tsteps = ledger.steps_for(enabled, method, "target")
        csteps = ledger.steps_for(enabled, method, "competitor")
    prepare_phase(phase, "target", steps=tsteps, operating_T=T_op, md_temperatures=o.md_temperatures or [T_op],
                  gates=enabled)
    if "thermodynamic" in enabled:
        prepare_competitors([phase], steps=csteps)
    rows = run_plan([phase], device=o.device, include_competing="thermodynamic" in enabled)
    return [f"{r['step']} failed: {r['detail']}".strip(": ") for r in rows
            if r["phase"] == phase.tag and r["action"] == "failed"]


def _pre_md(phase: PhaseDir, o: ScreenOptions, T_op: float) -> dict:
    """Relax, then - independently - the mechanical gate and the 0 K hull (each only if enabled)."""
    from heatup.analyze.gates import assess_mechanical_stability
    from heatup.analyze.hull import zero_k_hull

    enabled = _gates()
    errors: list[str] = _plan_and_run(phase, "pre_md", o, T_op) if o.compute else []
    gates: dict[str, GateResult] = {}
    hull0: dict = {}
    mech = None
    if not phase.has_relaxation():
        errors.append("relaxation failed or not available")
        return {"gates": gates, "hull0": hull0, "errors": errors}
    if "mechanical" in enabled:
        mech = assess_mechanical_stability(phase)
        gates["mechanical"] = _gate("mechanical", mech)
    if "thermodynamic" in enabled:
        hull0 = zero_k_hull(phase)
        if not hull0.get("available"):
            errors.append(f"0 K hull: {hull0.get('message')}")
        gates["hull_0K"] = (GateResult("hull_0K", STATUS_OK, f"{1e3 * hull0['e_hull_0K_eV']:.1f} meV/atom above the 0 K hull")
                            if hull0.get("available") else
                            GateResult("hull_0K", STATUS_MISSING, hull0.get("message", "")))
    return {"gates": gates, "hull0": hull0, "errors": errors, "mech": mech}


def screen(sym_dir: "str | os.PathLike | PhaseDir", stage: Stage = "post_md",
           options: ScreenOptions | None = None, **kwargs: Any) -> ScreenResult:
    """Assess one phase directory (``<root>/<material>/<label>`` containing a ``POSCAR``).

    ``kwargs`` are shorthand for :class:`ScreenOptions` fields (``screen(d, "pre_md", operating_T=600)``).
    """
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    o = dc.replace(options or ScreenOptions(), **kwargs) if kwargs else (options or ScreenOptions())
    dotted = _dotted(o)                                     # validates names before anything runs
    phase = PhaseDir.of(sym_dir)

    with _LOCK, override(**dotted):
        T_op = float(cfg.hull.operating_T)
        model = cfg.calc.model_for(phase.material)
        hull_reject = float(o.pre_md_hull_reject_eV if o.pre_md_hull_reject_eV is not None
                            else cfg.gates.hull_fail_eV)
        if not os.path.exists(phase.poscar):
            return ScreenResult("inconclusive", stage, [f"no POSCAR in {phase.path}"], None, None, T_op, {},
                                {}, [], phase.tag, phase.path, model, None, None, False)
        errors: list[str] = []
        flags: list[str] = []
        e_T = e_0 = None
        decomp: dict = {}
        complete = False
        report_file = os.path.join(phase.stability_dir, f"report_{stage}.json")
        try:
            if stage == "pre_md":
                r = _pre_md(phase, o, T_op)
                gates, errors = r["gates"], r["errors"]
                h0 = r["hull0"]
                e_0 = h0.get("e_hull_0K_eV")
                e_T = e_0
                decomp = h0.get("decomposition") or {}
                flags = list(h0.get("flags", []))
                hull_ok = bool(h0.get("available"))
                complete = bool(gates) and all(g.status != STATUS_MISSING for g in gates.values())
                verdict, reasons = decide("pre_md", gates, e_0, hull_ok, hull_reject_eV=hull_reject,
                                          accept_warn=o.accept_warn, errors=[e for e in errors if e])
                os.makedirs(phase.stability_dir, exist_ok=True)
                write_json(report_file, {"stage": stage, "phase": phase.tag, "operating_T_K": T_op,
                                         "mechanical": r.get("mech"), "hull_0K": h0, "errors": errors,
                                         "enabled": _gates(), "complete": complete,
                                         "calculator": model})
            else:
                from heatup.analyze.evaluate import evaluate
                run_errors = _plan_and_run(phase, "post_md", o, T_op) if o.compute else []
                rep = evaluate(phase, operating_T=T_op, save_figure=o.save_figure)
                gates = {k: _gate(k, rep.get(k)) for k in GATES if k not in rep["skipped"]}
                th = rep.get("thermodynamic") or {}
                e_T = th.get("e_hull_at_T_eV")
                decomp = th.get("decomposition") or {}
                flags = list(rep["flags"])
                errors = [f for f in rep["flags"] if f.startswith("[ERROR]")] + run_errors
                complete = bool(rep["complete"])
                verdict, reasons = decide("post_md", gates, None, True, hull_reject_eV=hull_reject,
                                          accept_warn=o.accept_warn, errors=errors)
        except Exception as exc:  # noqa: BLE001
            log.error("%s: screen(%s) failed: %s", phase.tag, stage, exc)
            log.debug(traceback.format_exc())
            return ScreenResult("inconclusive", stage, [f"internal error: {exc}"], e_T, e_0, T_op, {}, {}, flags,
                                phase.tag, phase.path, model, None, None, False)

        res = ScreenResult(verdict=verdict, stage=stage, reasons=reasons, e_hull_eV=e_T, e_hull_0K_eV=e_0,
                           operating_T_K=T_op, gates=gates, decomposition=decomp, flags=flags,
                           phase=phase.tag, phase_dir=phase.path, model=model,
                           report_file=report_file if os.path.exists(report_file) else None,
                           result_file=os.path.join(phase.stability_dir, f"result_{stage}.json"),
                           complete=complete)
        os.makedirs(phase.stability_dir, exist_ok=True)
        write_json(res.result_file, res.to_dict())
        log.info("%s [%s] -> %s (%s)", phase.tag, stage, verdict.upper(), "; ".join(reasons)[:200])
        return res


# ---------------------------------------------------------------------------
# many candidates: the funnel
# ---------------------------------------------------------------------------

_RANK = {STATUS_OK: 0, STATUS_WARN: 1, STATUS_MISSING: 2, STATUS_FAIL: 3, "pruned": 4, "error": 5}


def screen_many(targets: "list[str | PhaseDir]", out_dir: str | None = None, device: str | None = None,
                operating_T: float | None = None, md_temperatures: list[float] | None = None,
                prescreen_only: bool = False, force=None, **evaluate_kw) -> list[dict]:
    """Screening funnel for *many* structures, cheapest first, every step cached::

        plan -> relax targets + references -> 0 K pre-screen (drop > generation.prescreen_e_hull_eV)
             -> remaining steps of the survivors -> evaluate()

    One row per target (also written to ``<out>/screening.json`` and ``.csv``), best first.  *references* are
    known phases (database / candidates / Materials Project) that span the hull and are never judged; *targets*
    are the structures under test (yours, or generated by :mod:`heatup.prepare.generation`).
    """
    import csv

    from heatup.analyze.evaluate import evaluate
    from heatup.analyze.hull import zero_k_hull
    from heatup.compute.execute import run_plan
    from heatup.prepare.phases import phase_from_dir, to_database
    from heatup.prepare.plan import prepare_competitors, prepare_phase

    tph = [p for p in (phase_from_dir(to_database(t), "target") for t in targets) if p]   # computed copies live in the database
    if not tph:
        raise ValueError("no readable target structures")
    with _LOCK, override(**({"run.force": normalize_force(force)} if force is not None else {})):
        T_op = float(cfg.hull.operating_T if operating_T is None else operating_T)
        enabled, method = _gates(), cfg.hull.phonon_method
        for t in tph:
            prepare_phase(t.phase_dir, "target", steps=ledger.steps_for(enabled, method, "target"),
                          operating_T=T_op, md_temperatures=md_temperatures or [T_op], gates=enabled)
        refs = prepare_competitors([t.phase_dir for t in tph], steps=ledger.steps_for(enabled, method, "competitor"))
        log.info("screening %d target(s) against %d reference phase(s)", len(tph), len(refs))
        dirs = [t.phase_dir for t in tph]
        run_plan(dirs, device=device, steps=["relax"])        # the cheap stage of the funnel: just the relaxations

        rows: list[dict] = []
        alive: list[PhaseDir] = []
        limit = cfg.generation.prescreen_e_hull_eV
        for t in tph:
            h = zero_k_hull(t.phase_dir)
            e0 = h.get("e_hull_0K_eV")
            row = {"tag": t.tag, "formula": t.formula, "source": t.source, "e_hull_0K_eV": e0,
                   "overall": "error", "gates": {}, "message": "" if e0 is not None else h.get("message", "relaxation failed")}
            if e0 is not None and e0 > limit:
                row["overall"], row["message"] = "pruned", f"0 K hull distance {1e3 * e0:.0f} meV/atom > {1e3 * limit:.0f}"
                plan = ledger.read_plan(t.phase_dir)
                if plan is not None:
                    plan.pruned = {"reason": row["message"], "e_hull_0K_eV": e0}
                    ledger.write_plan(t.phase_dir, plan)
            elif e0 is not None:
                alive.append(t.phase_dir)
                row["overall"], row["message"] = STATUS_MISSING, "passed 0 K pre-screen"
            rows.append(row)
        if alive and not prescreen_only:
            run_plan(alive, device=device)
            for row in rows:
                d = next((a for a in alive if a.tag == row["tag"]), None)
                if d is None:
                    continue
                rep = evaluate(d, operating_T=T_op, **evaluate_kw)
                row["overall"] = rep["overall"]
                row["gates"] = {g: (rep[g] or {}).get("status") for g in GATES}
                row["e_hull_T_eV"] = (rep.get("thermodynamic") or {}).get("e_hull_at_T_eV")
                row["message"] = "; ".join(f for f in rep["flags"] if f.startswith("[FAIL]"))[:300]
    rows.sort(key=lambda r: (_RANK.get(r["overall"], 9),
                             r.get("e_hull_T_eV") if r.get("e_hull_T_eV") is not None else 9,
                             r["e_hull_0K_eV"] if r["e_hull_0K_eV"] is not None else 9))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        write_json(os.path.join(out_dir, "screening.json"), rows)
        cols = ["tag", "formula", "source", "overall", "e_hull_0K_eV", "e_hull_T_eV", "message"]
        with open(os.path.join(out_dir, "screening.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    return rows


def generate_and_screen(composition: dict, n: int | None = None, method: str | None = None,
                        out_dir: str | None = None, **kw) -> list[dict]:
    """Generate *n* candidates for the integer *composition* (e.g. ``{"Ba":1,"Ti":1,"O":3}``) and screen them."""
    from heatup.prepare.generation import generate
    phases = generate(composition, n=n, method=method)
    return screen_many([p.phase_dir for p in phases], out_dir=out_dir, **kw)
