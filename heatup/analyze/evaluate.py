"""heatup.analyze.evaluate
==========================
Notebook 3's driver: turn what is on disk into verdicts.  **Nothing here computes with a calculator.**

* :func:`status`        one row per (phase, step[, T]): done / stale / pending / unfinished / failed / ... -
                        :mod:`heatup.core.ledger` states, plus the 0 K-pruned competitors.  Notebook 2 uses the
                        same table (``calc=True``) to decide what to run.
* :func:`run_analysis`  the cheap, calculator-free analysis steps of every finished MD run
                        (``mdstats``, ``vdos``, ``transition``), cached by manifest.
* :func:`evaluate`      the enabled gates for one phase.  Gates are **independent**: a failing gate never stops
                        another, and a gate whose evidence is absent is ``missing`` with the reason.  The verdicts
                        are always recomputed from the cached step outputs; reports are written per stage
                        (``stability/report_<stage>.json``) and never read back as a result.  Evidence from steps
                        that are not ``done`` (unfinished MD, failed relaxation, ...) is flagged ``[incomplete]``;
                        ``strict=True`` turns such a gate into ``missing``.
* :func:`evaluate_all`, :func:`summary_rows`  the same for a whole database, as a flat table.
"""

from __future__ import annotations

import logging
import os
import traceback

from heatup import __version__
from heatup.analyze.gates import (assess_mechanical_stability, assess_structural_transition,
                                  assess_vibrational_stability)
from heatup.analyze.hull import assess_thermodynamic_stability, pruned_competitors
from heatup.analyze.thermo import anharmonic_summary, load_fc3
from heatup.core import ledger
from heatup.core.config import (GATES, STATUS_FAIL, STATUS_MISSING, STATUS_OK, STATUS_WARN, cfg)
from heatup.core.layout import PhaseDir, iter_phase_dirs, write_json

log = logging.getLogger("heatup")

_SEVERITY = {STATUS_FAIL: 3, STATUS_MISSING: 2, STATUS_WARN: 1, STATUS_OK: 0}
_ASSESS = {"mechanical": assess_mechanical_stability, "structural_transition": assess_structural_transition,
           "vibrational": assess_vibrational_stability, "thermodynamic": assess_thermodynamic_stability}


# ---------------------------------------------------------------------------
# status of the project
# ---------------------------------------------------------------------------

def status(calc: bool = False, include_competing: bool = True, problems_only: bool = False) -> list[dict]:
    """Rows ``{phase, role, step, T_K, state, detail, calculator}`` for every planned step of every phase in
    ``database_root``.  ``calc=True`` also compares the calculator (notebook 2); ``include_competing=False`` leaves
    out the competing phases; ``problems_only`` keeps everything that is not ``done`` / ``external`` / ``pruned``."""
    pruned = pruned_competitors()
    rows: list[dict] = []
    for pd in iter_phase_dirs(cfg.phases.database_root):
        if not include_competing and (ledger.read_plan(pd) or ledger.Plan()).role == "competitor":
            continue
        rows += ledger.phase_rows(pd, calc=calc, pruned=pruned)
    if problems_only:
        rows = [r for r in rows if r["state"] not in (*ledger.GOOD_STATES, ledger.PRUNED)]
    return rows


# ---------------------------------------------------------------------------
# analysis of finished MD runs
# ---------------------------------------------------------------------------

def run_analysis(phases=None, force=None) -> list[dict]:
    """Run ``mdstats``, ``vdos`` and ``transition`` for every finished MD run (planned temperatures) of *phases*
    (default: everything in ``database_root``).  Cached by manifest; a failure is logged
    and returned, never raised."""
    from heatup.analyze import trajectory
    pds = [PhaseDir.of(p) for p in phases] if phases is not None else list(iter_phase_dirs(cfg.phases.database_root))
    out: list[dict] = []
    for pd in pds:
        for T in ledger.md_temperatures(pd):
            for step in ledger.ANALYSIS_STEPS:
                fn = getattr(trajectory, ledger.STEPS[step].func.split(":")[1])
                try:
                    fn(pd, T, force=force)
                    out.append({"phase": pd.tag, "step": step, "T_K": T, "ok": True, "error": ""})
                except Exception as exc:  # noqa: BLE001
                    log.warning("%s %dK %s: %s", pd.tag, T, step, exc)
                    log.debug(traceback.format_exc())
                    out.append({"phase": pd.tag, "step": step, "T_K": T, "ok": False, "error": str(exc)})
    return out


# ---------------------------------------------------------------------------
# gates of one phase
# ---------------------------------------------------------------------------

def overall_status(report: dict) -> str:
    skipped = set(report.get("skipped", []))
    st = [report[g].get("status", STATUS_MISSING) for g in GATES if g not in skipped and report.get(g) is not None]
    if STATUS_FAIL in st:
        return STATUS_FAIL
    if STATUS_WARN in st or STATUS_MISSING in st or not st:
        return STATUS_WARN
    return STATUS_OK


def _describe(r: dict) -> str:
    at = "" if r["T_K"] is None else "@%gK" % r["T_K"]
    return f"{r['step']}{at}: {r['state']}" + (f" ({r['detail']})" if r["detail"] else "")


def _incomplete(phase: PhaseDir, gates: list[str]) -> dict[str, list[str]]:
    """``{gate: ["md@600K: unfinished (120 of 401 frames)", ...]}`` for planned steps the gates read that are not done."""
    rows = ledger.phase_rows(phase, calc=False)
    out: dict[str, list[str]] = {}
    for g in gates:
        need = set(ledger.gate_steps(g, ledger.phonon_method(phase)))
        bad = [_describe(r) for r in rows
               if r["step"] in need and r["state"] not in (*ledger.GOOD_STATES, ledger.PRUNED)]
        if bad:
            out[g] = bad
    return out


def evaluate(phase_dir, operating_T: float | None = None, gates: list[str] | None = None, strict: bool = False,
             save_figure: bool = True, figure_fmt: str = "png", stage: str = "post_md") -> dict:
    """Evaluate the gates of one phase from the files on disk: ``gates``, else ``cfg.gates.enabled``, else the ones
    the phase's plan asked for (``cfg.plan.gates`` in notebook 1), else all.

    ``operating_T=None`` -> the phase's own value (``plan.json``), else ``cfg.hull.operating_T``.
    """
    phase = PhaseDir.of(phase_dir)
    enabled = ledger.resolve_gates(phase, gates)
    unknown = [g for g in enabled if g not in GATES]
    if unknown:
        raise ValueError(f"unknown gate(s) {unknown}; choose from {list(GATES)}")
    T_op = ledger.operating_T(phase) if operating_T is None else float(operating_T)
    report: dict = {"version": __version__, "stage": stage, "material": phase.material, "label": phase.label,
                    "phase_dir": phase.path, "operating_T_K": T_op, "md_temperatures_K": ledger.md_temperatures(phase),
                    "calculator": ledger.calculator_of(phase) or "?", "config": cfg.snapshot(),
                    "structural_transition": None, "mechanical": None, "vibrational": None, "thermodynamic": None,
                    "anharmonic": None, "skipped": [g for g in GATES if g not in enabled], "enabled": enabled,
                    "incomplete": {}, "flags": [], "overall": STATUS_MISSING}
    log.info("=== %s  (T = %g K, gates: %s) ===", phase.tag, T_op, ", ".join(enabled))
    if not enabled:
        report["flags"].append("[MISSING] no gate enabled")
        return _finish(report, phase, save_figure, figure_fmt)
    if not phase.has_relaxation():
        for g in enabled:
            report[g] = {"available": False, "status": STATUS_MISSING,
                         "message": "no converged relaxation (step 'relax', notebook 2)"}
        report["flags"].append("[MISSING] no converged relaxation")
        return _finish(report, phase, save_figure, figure_fmt)

    report["incomplete"] = _incomplete(phase, enabled)
    for g in GATES:
        if g not in enabled:
            continue
        if strict and g in report["incomplete"]:
            r = {"available": False, "status": STATUS_MISSING,
                 "message": "evidence incomplete: " + "; ".join(report["incomplete"][g])}
        else:
            try:
                r = _ASSESS[g](phase, **({"operating_T": T_op} if g == "thermodynamic" else {}))
            except Exception as exc:  # noqa: BLE001  one gate's failure is never another's
                log.error("%s: gate %s failed: %s", phase.tag, g, exc)
                log.debug(traceback.format_exc())
                r = {"available": False, "status": STATUS_MISSING, "message": f"evaluation failed: {exc}"}
        report[g] = r
        log.info("[%s] %s - %s", g, r.get("status", "?").upper(), r.get("message", ""))
        report["flags"] += [f"[incomplete] {g}: {x}" for x in report["incomplete"].get(g, [])]

    unplanned = sorted(set(phase.md_temperatures()) - set(report["md_temperatures_K"]))
    if unplanned:
        report["flags"].append(f"[note] MD run(s) at {unplanned} K are not in the plan and were ignored")
    report["anharmonic"] = anharmonic_summary(load_fc3(phase), T_op)        # diagnostic (fc3 step), never a gate
    if report["anharmonic"] and report["anharmonic"]["flag"]:
        report["flags"].append(report["anharmonic"]["flag"])
    return _finish(report, phase, save_figure, figure_fmt)


def _finish(report: dict, phase: PhaseDir, save_figure: bool, fmt: str) -> dict:
    for g in GATES:
        r = report.get(g)
        if r and r.get("status") in (STATUS_WARN, STATUS_FAIL, STATUS_MISSING) \
                and not any(f.endswith(r.get("message", "")) for f in report["flags"]):
            report["flags"].append(f"[{r['status'].upper()}] {g}: {r.get('message', '')}")
        if r:
            report["flags"] += [f"[note] {g}: {x}" for x in r.get("flags", []) if isinstance(x, str)]
    report["overall"] = overall_status(report)
    report["complete"] = bool(report.get("enabled")) and not report.get("incomplete") and all(
        (report.get(g) or {}).get("status", STATUS_MISSING) != STATUS_MISSING
        for g in GATES if g not in report["skipped"])
    os.makedirs(phase.stability_dir, exist_ok=True)
    slim = dict(report)
    slim["vibrational"] = _strip(report.get("vibrational"), ("vdos",))
    slim["thermodynamic"] = _strip(report.get("thermodynamic"), ("contributions",))
    write_json(os.path.join(phase.stability_dir, f"report_{report['stage']}.json"), slim)
    if save_figure:
        try:
            import matplotlib.pyplot as plt
            from heatup.analyze.report import plot_stability_report
            fig = plot_stability_report(report, os.path.join(phase.stability_dir, f"report_{report['stage']}.{fmt}"))
            plt.close(fig)
        except Exception as exc:  # noqa: BLE001
            log.warning("figure failed: %s", exc)
    log.info("=== %s: OVERALL %s ===", phase.tag, report["overall"].upper())
    return report


def _strip(d, keys):
    if not isinstance(d, dict):
        return d
    return {k: v for k, v in d.items() if k not in keys}


# ---------------------------------------------------------------------------
# many phases
# ---------------------------------------------------------------------------

def evaluate_all(database_root: str | None = None, include_competing: bool = False, **kwargs) -> list[dict]:
    """:func:`evaluate` for every *target* of *database_root* (default ``cfg.phases.database_root``), sorted
    fail -> warn -> missing -> ok.  Competing phases are evidence, not subjects: they get no verdict unless
    ``include_competing=True``.  An error in one phase never stops the others."""
    out = []
    for pd in iter_phase_dirs(database_root or cfg.phases.database_root):
        if not include_competing and (ledger.read_plan(pd) or ledger.Plan()).role == "competitor":
            continue
        try:
            out.append(evaluate(pd, **kwargs))
        except Exception as exc:  # noqa: BLE001
            log.error("%s: %s", PhaseDir.of(pd).tag, exc)
    out.sort(key=lambda r: (-_SEVERITY.get(r["overall"], 0), r["material"], r["label"]))
    return out


def summary_rows(reports: "dict[str, dict] | list[dict]") -> list[dict]:
    """Flat one-row-per-phase table (feed to ``pandas.DataFrame``)."""
    items = reports.items() if isinstance(reports, dict) else [(f"{r['material']}/{r['label']}", r) for r in reports]
    rows = []
    for tag, r in items:
        m, s, v, t = (r.get(k) or {} for k in ("mechanical", "structural_transition", "vibrational", "thermodynamic"))
        rows.append({
            "phase": tag, "overall": r.get("overall"),
            "G1 mech": m.get("status"), "B_GPa": m.get("bulk_GPa"), "G_GPa": m.get("shear_GPa"),
            "G0 struct": s.get("status"), "G2 vib": v.get("status"), "G3 hull": t.get("status"),
            "E_hull_meV": None if t.get("e_hull_at_T_eV") is None else 1e3 * t["e_hull_at_T_eV"],
        })
    return rows
