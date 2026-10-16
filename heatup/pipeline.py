"""heatup.pipeline
===================
The gate runner for one phase directory.

Gates (``cfg.gates.enabled``, default all)::

    mechanical (1)   structural_transition (0)   vibrational (2)   thermodynamic (3)

They are **independent**: whichever are enabled are evaluated, in the order of increasing cost
(relax -> elastic -> phonons -> MD -> hull), and a failing gate never stops another one - you get the
full picture of *how* a candidate fails.  Only a missing relaxation makes every gate ``missing``.
A gate whose evidence cannot be produced (a failed step, no MD yet) is ``missing`` with the reason;
a gate you disabled is listed under ``report["skipped"]`` and does not count for ``overall``.

Caching and ``force``: every step is cached by ``manifest`` (config fingerprint + input hashes).
``cfg.run.force`` (or the ``force=`` argument) lists the steps to recompute.  Gate verdicts are
*always recomputed* from the cached step outputs (cheap); v0.2 cached the final
``stability_report.json`` and returned it after the config or the structure had changed.  Reports are
now written per stage (``stability/report_<stage>.json``, with ``stage`` and ``complete`` fields)
and are never read back as a result.
"""

from __future__ import annotations

import logging
import os
import traceback

from heatup import __version__
from heatup.config import (STATUS_FAIL, STATUS_MISSING, STATUS_OK, STATUS_WARN, cfg)
from heatup.mechanical import assess_mechanical_stability
from heatup.paths import PhaseDir, iter_phase_dirs, write_json
from heatup.runtime import run_step
from heatup.structural_transition import assess_structural_transition
from heatup.vibrational import assess_vibrational_stability

log = logging.getLogger("heatup")

GATES = ("structural_transition", "mechanical", "vibrational", "thermodynamic")
_SEVERITY = {STATUS_FAIL: 3, STATUS_MISSING: 2, STATUS_WARN: 1, STATUS_OK: 0}


def overall_status(report: dict) -> str:
    skipped = set(report.get("skipped", []))
    st = [report[g].get("status", STATUS_MISSING) for g in GATES
          if g not in skipped and report.get(g) is not None]
    if STATUS_FAIL in st:
        return STATUS_FAIL
    if STATUS_WARN in st or STATUS_MISSING in st or not st:
        return STATUS_WARN
    return STATUS_OK


def _log_gate(name: str, r: dict) -> None:
    log.info("[%s] %s - %s", name, r.get("status", "?").upper(), r.get("message", ""))


def run_pipeline(phase_dir, operating_T: float | None = None, device: str | None = None,
                 md_temperatures: list[float] | None = None, force=None,
                 compute: bool = True, gates: list[str] | None = None,
                 save_figure: bool = True, figure_fmt: str = "png", mp_docs: list | None = None) -> dict:
    """Run (or just re-evaluate, ``compute=False``) the enabled gates for one phase.

    * Which gates run: ``gates`` or ``cfg.gates.enabled`` (default: all four).  They are
      **independent** - a failing gate never prevents another enabled gate from being evaluated;
      only a missing relaxation makes every gate ``missing``.  Disabled gates are listed in
      ``report["skipped"]``.
    * ``force``: ``None`` -> ``cfg.run.force``; ``True`` (all steps), a step name or a list of step
      names (``"relax" "elastic" "phonons" "qha" "md" "analysis"``) recompute those steps.
    * Steps are only run for the enabled gates (mechanical: elastic; structural_transition: md;
      vibrational: phonons + md; thermodynamic: its own competitor/phonon steps).
    """
    from heatup.config import normalize_force, override
    if force is None:
        return _run(phase_dir, operating_T, device, md_temperatures, compute, gates,
                    save_figure, figure_fmt, mp_docs)
    with override(**{"run.force": normalize_force(force)}):
        return _run(phase_dir, operating_T, device, md_temperatures, compute, gates,
                    save_figure, figure_fmt, mp_docs)


def _run(phase_dir, operating_T, device, md_temperatures, compute, gates, save_figure, figure_fmt, mp_docs) -> dict:
    from heatup.config import GATES
    from heatup.thermodynamic import assess_thermodynamic_stability

    phase = PhaseDir.of(phase_dir)
    enabled = list(cfg.gates.enabled if gates is None else gates)
    unknown = [g for g in enabled if g not in GATES]
    if unknown:
        raise ValueError(f"unknown gate(s) {unknown}; choose from {list(GATES)}")
    T_op = float(cfg.hull.operating_T if operating_T is None else operating_T)
    temps = [float(t) for t in (md_temperatures or [T_op])]
    report: dict = {"version": __version__, "stage": "post_md", "material": phase.material, "label": phase.label,
                    "phase_dir": phase.path, "operating_T_K": T_op, "md_temperatures_K": temps,
                    "calculator": cfg.calc.model_for(phase.material), "config": cfg.snapshot(),
                    "structural_transition": None, "mechanical": None, "vibrational": None,
                    "thermodynamic": None, "skipped": [g for g in GATES if g not in enabled],
                    "enabled": enabled, "flags": [], "overall": STATUS_MISSING}
    log.info("=== %s  (T = %g K, gates: %s) ===", phase.tag, T_op, ", ".join(enabled))
    if not enabled:
        report["flags"].append("[MISSING] no gate enabled")
        return _finish(report, phase, save_figure, figure_fmt)

    def step(name: str, **kw) -> bool:
        return (not compute) or run_step(name, phase, device=device, **kw)

    def gate(name: str, fn, *args, **kw) -> None:
        """Evaluate one gate; an exception inside it becomes that gate's ``missing`` - never the others'."""
        try:
            r = fn(*args, **kw)
        except Exception as exc:  # noqa: BLE001
            log.error("%s: gate %s failed: %s", phase.tag, name, exc)
            log.debug(traceback.format_exc())
            r = {"available": False, "status": STATUS_MISSING, "message": f"evaluation failed: {exc}"}
        report[name] = r
        _log_gate(name, r)

    if not step("relax") and not (not compute and phase.has_relaxation()):
        for g in enabled:
            report[g] = {"available": False, "status": STATUS_MISSING,
                         "message": "relaxation failed or not available"}
        report["flags"].append("[MISSING] relaxation failed")
        return _finish(report, phase, save_figure, figure_fmt)

    if "mechanical" in enabled:
        step("elastic")
        gate("mechanical", assess_mechanical_stability, phase)
    needs_md = "structural_transition" in enabled or "vibrational" in enabled
    if "vibrational" in enabled:
        step("phonons")
    if needs_md:
        for T in temps:
            step("md", temperature=T)
    if "structural_transition" in enabled:
        gate("structural_transition", assess_structural_transition, phase)
    if "vibrational" in enabled:
        gate("vibrational", assess_vibrational_stability, phase)
    if "thermodynamic" in enabled:
        gate("thermodynamic", assess_thermodynamic_stability, phase, operating_T=T_op, device=device,
             prepare=compute, mp_docs=mp_docs)
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
    report["complete"] = bool(report.get("enabled")) and all(
        (report.get(g) or {}).get("status", STATUS_MISSING) != STATUS_MISSING
        for g in GATES if g not in report["skipped"])
    os.makedirs(phase.stability_dir, exist_ok=True)
    slim = dict(report)
    slim["vibrational"] = _strip(report.get("vibrational"), ("vdos",))
    slim["thermodynamic"] = _strip(report.get("thermodynamic"), ("contributions",))
    write_json(os.path.join(phase.stability_dir, f"report_{report['stage']}.json"), slim)
    if save_figure:
        try:
            from heatup.plotting import plot_stability_report
            import matplotlib.pyplot as plt
            fig = plot_stability_report(report, os.path.join(phase.stability_dir,
                                                             f"report_{report['stage']}.{fmt}"))
            plt.close(fig)
        except Exception as exc:  # noqa: BLE001
            log.warning("figure failed: %s", exc)
    log.info("=== %s: OVERALL %s ===", phase.tag, report["overall"].upper())
    return report


def _strip(d, keys):
    if not isinstance(d, dict):
        return d
    return {k: v for k, v in d.items() if k not in keys}


def run_batch(database_root: str | None = None, **kwargs) -> list[dict]:
    """:func:`run_pipeline` for every phase directory under *database_root*.

    Phases that have no ``POSCAR`` are skipped; errors in one phase never stop the batch.
    Returned sorted fail -> warn -> missing -> ok.
    """
    root = database_root or cfg.phases.database_root
    out = []
    for pd in iter_phase_dirs(root):
        try:
            out.append(run_pipeline(pd, **kwargs))
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


def database_status(root: str | None = None) -> list[dict]:
    """Which calculations exist for every phase directory under *root*."""
    out = []
    for d in iter_phase_dirs(root or cfg.phases.database_root):
        p = PhaseDir.of(d)
        out.append({"phase": p.tag, "relaxed": p.has_relaxation(),
                    "elastic": os.path.exists(p.elastic_json),
                    "HA": os.path.exists(os.path.join(p.phonon_dir, "ha.json")),
                    "QHA": os.path.exists(os.path.join(p.phonon_dir, "qha.json")),
                    "MD_T_K": p.md_temperatures(),
                    "reports": sorted(f[7:-5] for f in (os.listdir(p.stability_dir) if os.path.isdir(p.stability_dir) else [])
                                          if f.startswith("report_") and f.endswith(".json"))})
    return out
