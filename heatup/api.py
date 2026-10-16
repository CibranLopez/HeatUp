"""heatup.api
==============
The stable entry point for other software: :func:`screen`.

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
* **Gates are independent.**  The gates to run come from ``ScreenOptions.gates`` / ``cfg.gates.enabled``;
  every enabled gate is evaluated whatever the others say (a Born-unstable phase still gets its MD,
  vibrational and hull results), and the verdict is decided from all of them.  ``force`` recomputes
  the named steps (or ``True`` = all) instead of reading their caches.
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

from heatup.config import (GATES, STATUS_FAIL, STATUS_MISSING, STATUS_OK, STATUS_WARN, cfg, normalize_force,
                           override)
from heatup.paths import PhaseDir, write_json

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
    gates: list[str] | None = None                   # which gates to evaluate (default: cfg.gates.enabled = all);
                                                     # any of mechanical/structural_transition/vibrational/thermodynamic
    # calculator
    backend: str | None = None
    model: str | None = None                         # THE model: mace_mp name or .model path
    model_overrides: dict[str, str] | None = None    # material folder -> model path
    device: str | None = None
    # trees
    database_root: str | None = None
    competing_root: str | None = None                # competing phases live here, not in candidates
    sources: list[str] | None = None
    mp_api_key: str | None = None
    # run control
    compute: bool = True                             # False: evaluate cached results only
    force: bool | str | list[str] | None = None      # True/"all", or step names ("relax" "elastic" "phonons" "qha" "fc3" "md" "analysis")
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

_DIRECT = {"gates": "gates.enabled", "operating_T": "hull.operating_T", "mobile_species": "gates.mobile_species",
           "backend": "calc.backend", "model": "calc.model", "model_overrides": "calc.model_overrides",
           "database_root": "phases.database_root", "competing_root": "phases.competing_root",
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

def _pre_md(phase: PhaseDir, o: ScreenOptions) -> dict:
    """Relax, then - independently - the mechanical gate and the 0 K hull (each only if enabled)."""
    from heatup.mechanical import assess_mechanical_stability
    from heatup.runtime import run_step
    from heatup.thermodynamic import zero_k_hull

    enabled = cfg.gates.enabled
    errors: list[str] = []
    gates: dict[str, GateResult] = {}
    hull0: dict = {}
    mech = None
    ok = phase.has_relaxation() if not o.compute else run_step("relax", phase, device=o.device)
    if not ok:
        errors.append("relaxation failed or not available")
        return {"gates": gates, "hull0": hull0, "errors": errors}
    if "mechanical" in enabled:
        if o.compute:
            run_step("elastic", phase, device=o.device)
        mech = assess_mechanical_stability(phase)
        gates["mechanical"] = _gate("mechanical", mech)
    if "thermodynamic" in enabled:
        hull0 = zero_k_hull(phase, device=o.device, prepare=o.compute)
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
                r = _pre_md(phase, o)
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
                                         "enabled": list(cfg.gates.enabled), "complete": complete,
                                         "calculator": model})
            else:
                from heatup.pipeline import run_pipeline
                rep = run_pipeline(phase, operating_T=T_op, device=o.device, md_temperatures=o.md_temperatures,
                                   compute=o.compute, save_figure=o.save_figure)
                gates = {k: _gate(k, rep.get(k)) for k in GATES if k not in rep["skipped"]}
                th = rep.get("thermodynamic") or {}
                e_T = th.get("e_hull_at_T_eV")
                decomp = th.get("decomposition") or {}
                flags = list(rep["flags"])
                errors = [f for f in rep["flags"] if f.startswith("[ERROR]")]
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
