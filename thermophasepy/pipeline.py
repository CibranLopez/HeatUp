"""thermophasepy.pipeline
========================
Top-level pipeline orchestrating the three sequential stability gates.

Usage::

    from thermophasepy import run_stability_pipeline

    report = run_stability_pipeline(
        sym_dir     = "database/LiZrS2/R3m",
        operating_T = 1200.0,
        device      = "cuda",
    )
    if report["overall"] == "ok":
        # include in retraining dataset
        ...
"""

from __future__ import annotations

import json
import os
import traceback

import matplotlib.pyplot as plt

from thermophasepy import config
from thermophasepy.mechanical    import assess_mechanical_stability
from thermophasepy.vibrational   import assess_vibrational_stability
from thermophasepy.thermodynamic import assess_thermodynamic_stability
from thermophasepy.plotting      import plot_stability_report


def run_stability_pipeline(
        sym_dir: str,
        operating_T: float = 1200.0,
        candidates_root: str = config.CANDIDATES_ROOT,
        database_root: str = config.DATABASE_ROOT,
        temperatures: list[float] | None = None,
        device: str = config.DEFAULT_DEVICE,
        generate_missing_phases: bool = True,
        force_rerun: bool = False,
        save_figure: bool = True,
        figure_fmt: str = "pdf",
) -> dict:
    """Run the three-gate stability pipeline for a single material.

    Gates are evaluated in order:

        1. **Mechanical** — Born elastic stability.
           A **fail** result stops the pipeline (no point computing phonons
           for a mechanically collapsed crystal).
        2. **Vibrational** — soft-mode detection from anharmonic VDOS.
           A **fail** result stops the pipeline (thermodynamic hull requires
           meaningful free energies, which cannot be extracted from a phase
           undergoing a structural instability).
        3. **Thermodynamic** — temperature-dependent convex hull.
           Always runs if gates 1–2 are ``ok`` or ``warn``.

    A ``warn`` in any gate does not block subsequent gates, so borderline
    materials are fully characterised before a decision is made.

    Args:
        sym_dir:                 Path to ``database/<material>/<symmetry>/``.
        operating_T:             Temperature (K) for stability evaluation.
        candidates_root:         Root of the candidates POSCAR tree.
        database_root:           Root of the simulation database.
        temperatures:            Hull temperature grid (K).  Defaults to
                                 ``config.HULL_TEMPERATURES``.
        device:                  MACE compute device (``'cuda'`` or ``'cpu'``).
        generate_missing_phases: Generate secondary-phase POSCARs with PyXtal
                                 before building the hull.
        force_rerun:             Recompute even if ``stability_report.json``
                                 already exists.
        save_figure:             Save the dashboard figure to
                                 ``<sym_dir>/stability/stability_report.<fmt>``.
        figure_fmt:              Figure format (``'pdf'`` or ``'png'``).

    Returns:
        Dict with keys:

        ``'material'``, ``'symmetry'``, ``'sym_dir'``, ``'operating_T_K'``
            Identification.
        ``'mechanical'``, ``'vibrational'``, ``'thermodynamic'``
            Gate result dicts (see individual assess functions).
        ``'overall'``
            ``'ok'`` | ``'warn'`` | ``'fail'`` | ``'missing'``.
        ``'flags'``
            List of human-readable flag strings for non-OK gates.
        ``'stopped_at'``
            ``None`` if all gates ran; gate name if pipeline was stopped early.
    """
    if temperatures is None:
        temperatures = [float(t) for t in config.HULL_TEMPERATURES]

    sym_dir  = os.path.abspath(sym_dir)
    symmetry = os.path.basename(sym_dir)
    material = os.path.basename(os.path.dirname(sym_dir))
    tag      = f"{material}/{symmetry}"

    stab_dir    = os.path.join(sym_dir, "stability")
    report_path = os.path.join(stab_dir, "stability_report.json")

    if not force_rerun and os.path.exists(report_path):
        try:
            with open(report_path) as fh:
                cached = json.load(fh)
            print(f"[done] {tag} — stability_report.json exists. "
                  "Pass force_rerun=True to redo.")
            return cached
        except Exception:
            pass

    print(f"\n{'=' * 65}")
    print(f"  ThermoPhase Pipeline: {tag}")
    print(f"  Operating temperature : {operating_T:.0f} K")
    print(f"{'=' * 65}")

    report: dict = {
        "material"      : material,
        "symmetry"      : symmetry,
        "sym_dir"       : sym_dir,
        "operating_T_K" : operating_T,
        "mechanical"    : {},
        "vibrational"   : {},
        "thermodynamic" : {},
        "overall"       : config.STATUS_MISSING,
        "flags"         : [],
        "stopped_at"    : None,
    }

    # ── Gate 1: Mechanical ────────────────────────────────────────────────
    print(f"\n  [Gate 1] Mechanical stability...")
    mech = assess_mechanical_stability(sym_dir)
    report["mechanical"] = mech
    print(f"    {mech['status'].upper()} — {mech['message']}")

    if mech["status"] == config.STATUS_FAIL:
        report["stopped_at"] = "mechanical"
        report["flags"].append(f"[FAIL] Mechanical: {mech['message']}")
        _finalise(report)
        _persist(report, stab_dir, save_figure, figure_fmt)
        return report

    if mech["status"] in (config.STATUS_WARN, config.STATUS_MISSING):
        report["flags"].append(
            f"[{mech['status'].upper()}] Mechanical: {mech['message']}"
        )

    # ── Gate 2: Vibrational ───────────────────────────────────────────────
    print(f"\n  [Gate 2] Vibrational stability (anharmonic VDOS)...")
    vib = assess_vibrational_stability(sym_dir)
    report["vibrational"] = vib
    print(f"    {vib['status'].upper()} — {vib['message']}")

    if vib["status"] == config.STATUS_FAIL:
        report["stopped_at"] = "vibrational"
        report["flags"].append(f"[FAIL] Vibrational: {vib['message']}")
        _finalise(report)
        _persist(report, stab_dir, save_figure, figure_fmt)
        return report

    if vib["status"] in (config.STATUS_WARN, config.STATUS_MISSING):
        report["flags"].append(
            f"[{vib['status'].upper()}] Vibrational: {vib['message']}"
        )

    # ── Gate 3: Thermodynamic ─────────────────────────────────────────────
    print(f"\n  [Gate 3] Thermodynamic stability (convex hull with T)...")
    therm = assess_thermodynamic_stability(
        sym_dir          = sym_dir,
        operating_T      = operating_T,
        candidates_root  = candidates_root,
        database_root    = database_root,
        temperatures     = temperatures,
        device           = device,
        generate_missing = generate_missing_phases,
    )
    report["thermodynamic"] = therm
    print(f"    {therm['status'].upper()} — {therm['message']}")

    if therm["status"] in (config.STATUS_WARN, config.STATUS_FAIL,
                           config.STATUS_MISSING):
        report["flags"].append(
            f"[{therm['status'].upper()}] Thermodynamic: {therm['message']}"
        )

    _finalise(report)
    _persist(report, stab_dir, save_figure, figure_fmt)
    return report


def _finalise(report: dict) -> None:
    statuses = [
        report["mechanical"].get("status",    config.STATUS_MISSING),
        report["vibrational"].get("status",   config.STATUS_MISSING),
        report["thermodynamic"].get("status", config.STATUS_MISSING),
    ]
    if config.STATUS_FAIL in statuses:
        report["overall"] = config.STATUS_FAIL
    elif config.STATUS_WARN in statuses or config.STATUS_MISSING in statuses:
        report["overall"] = config.STATUS_WARN
    else:
        report["overall"] = config.STATUS_OK


def _persist(
        report: dict,
        stab_dir: str,
        save_figure: bool,
        figure_fmt: str,
) -> None:
    os.makedirs(stab_dir, exist_ok=True)

    # Slim down the JSON: strip large VDOS arrays (stored in anharmonic_phonons/).
    slim = {k: v for k, v in report.items() if k != "vibrational"}
    slim["vibrational"] = {
        k: v for k, v in report.get("vibrational", {}).items()
        if k not in ("omega_mev", "vdos")
    }
    with open(os.path.join(stab_dir, "stability_report.json"), "w") as fh:
        json.dump(slim, fh, indent=4)

    print(f"\n  Report → {os.path.join(stab_dir, 'stability_report.json')}")
    print(f"  Overall: {report['overall'].upper()}")

    if save_figure:
        fig_path = os.path.join(stab_dir, f"stability_report.{figure_fmt}")
        try:
            fig = plot_stability_report(report, save_path=fig_path)
            plt.close(fig)
        except Exception as exc:
            print(f"  [warn] Figure generation failed: {exc}")


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_stability_pipeline_batch(
        database_root: str = config.DATABASE_ROOT,
        candidates_root: str = config.CANDIDATES_ROOT,
        operating_T: float = 1200.0,
        temperatures: list[float] | None = None,
        device: str = config.DEFAULT_DEVICE,
        generate_missing_phases: bool = True,
        force_rerun: bool = False,
        save_figures: bool = True,
        figure_fmt: str = "pdf",
) -> list[dict]:
    """Run :func:`run_stability_pipeline` for every validated database entry.

    An entry is considered validated if its ``aimd/`` directory contains at
    least one non-empty ``output.traj`` file.

    Args:
        database_root:           Root of the simulation database.
        candidates_root:         Root of the candidate tree.
        operating_T:             Hull evaluation temperature (K).
        temperatures:            Hull temperature grid.
        device:                  MACE compute device.
        generate_missing_phases: Run PyXtal for missing polymorphs.
        force_rerun:             Recompute all reports.
        save_figures:            Write dashboard figures.
        figure_fmt:              Figure format (``'pdf'`` or ``'png'``).

    Returns:
        List of report dicts sorted by severity (fail → warn → ok) then name.
    """
    reports: list[dict] = []

    if not os.path.isdir(database_root):
        print(f"[warn] database_root not found: {database_root}")
        return reports

    for material in sorted(os.listdir(database_root)):
        mat_dir = os.path.join(database_root, material)
        if not os.path.isdir(mat_dir):
            continue
        for symmetry in sorted(os.listdir(mat_dir)):
            sym_dir  = os.path.join(mat_dir, symmetry)
            aimd_dir = os.path.join(sym_dir, "aimd")
            if not os.path.isdir(sym_dir) or not os.path.isdir(aimd_dir):
                continue

            has_traj = any(
                os.path.exists(os.path.join(aimd_dir, tf, "output.traj"))
                and os.path.getsize(
                    os.path.join(aimd_dir, tf, "output.traj")
                ) > 0
                for tf in os.listdir(aimd_dir)
                if tf.endswith("K")
            )
            if not has_traj:
                continue

            print(f"\n  Processing {material}/{symmetry}...")
            try:
                rep = run_stability_pipeline(
                    sym_dir                 = sym_dir,
                    operating_T             = operating_T,
                    candidates_root         = candidates_root,
                    database_root           = database_root,
                    temperatures            = temperatures,
                    device                  = device,
                    generate_missing_phases = generate_missing_phases,
                    force_rerun             = force_rerun,
                    save_figure             = save_figures,
                    figure_fmt              = figure_fmt,
                )
                reports.append(rep)
            except Exception as exc:
                print(f"  [error] {material}/{symmetry}: {exc}")
                traceback.print_exc()

    _order = {
        config.STATUS_FAIL   : 0,
        config.STATUS_WARN   : 1,
        config.STATUS_OK     : 2,
        config.STATUS_MISSING: 3,
    }
    reports.sort(key=lambda r: (_order.get(r["overall"], 9), r["material"]))
    return reports
