"""heatup.analyze.gates
=======================
The verdicts of the mechanical, structural-transition and vibrational checks from cached step results (the thermodynamic
one, the hull, is :mod:`heatup.analyze.hull`).  Nothing here computes anything expensive and every threshold
(``cfg.gates``) is applied on each call, so thresholds can change without recomputing.  The checks are independent and
unordered: each reads its own evidence, none reads another's verdict, any one can be asked for alone.

mechanical (Born-Huang)
    The necessary and sufficient condition for a crystal at (near) zero stress is a positive-definite 6x6
    stiffness matrix (Mouhat & Coudert, PRB 90, 224104, 2014): all eigenvalues of C > 0.  The Hill-averaged
    bulk and shear moduli are checked against soft-material thresholds.

structural transition - did the structure survive the MD run?
    v0.2 measured the plateau of ``C(tau) = <du(t) du(t+tau)>`` with ``du = r(t) - r(first frame)``: the reference
    is a *single thermal snapshot*, so even a perfectly stable crystal has ``C_inf`` = the thermal mean-square
    displacement (0.01-0.1 A^2 at 1000 K) - the threshold sat inside the noise.  Now, per MD temperature:

    1. block-mean positions of the first and last ``analysis.block_ps`` of the production run (thermal noise
       averages out), centre-of-mass drift removed: rms and max displacement of the non-mobile atoms -> FAIL
       above thresholds;
    2. cell change between the same blocks: deviatoric Green-Lagrange strain and volume change -> FAIL;
    3. symmetry breaking: every symmetry operation of the *initial* structure applied to the block-mean
       structures -> WARN when the last block is more broken than ``gates.symmetry_break_A``;
    4. diffusing species not declared in ``analysis.mobile_species`` are reported (they normally trigger 1).

vibrational - soft or imaginary modes, from the evidence chosen in ``cfg.vibrational.sources``
    ``harmonic``       imaginary modes of the 0 K phonons: share of the mode weight beyond ``harmonic.imag_tol_meV`` and the
                       imaginary BRANCHES along the band path (which branch, which q-point, how much of the path;
                       :mod:`heatup.analyze.vibrations`).
    ``quasiharmonic``  the same share at the volumes the crystal visits between 0 K and the highest temperature.
    ``anharmonic``     weight of the mass-weighted VDOS of the MD runs inside ``|w| < window``.  A solid has
                       g(w) ~ w^2 so the fraction is ~1e-4; a collapsing soft mode or a liquid-like sub-lattice puts weight
                       at w = 0.  The window is widened to 1.5x the spectral resolution of the run.  Species in
                       ``analysis.mobile_species`` are excluded.  A VDOS has no sign: it shows soft weight, not negative
                       frequencies.
    The most complete evidence that is available decides (anharmonic > quasi-harmonic > harmonic); the others are reported
    with their own status as information.  Imaginary harmonic modes at 0 K are *not* by themselves a failure at finite T
    (cubic perovskites, bcc metals are stabilised by anharmonicity), so a clean finite-temperature result outranks them.
    The figure of the selected sources is written to ``stability/vibrational.png`` (``cfg.vibrational.save_figures``).
"""

from __future__ import annotations

import logging

import numpy as np

from heatup.analyze import vibrations
from heatup.analyze.thermo import load_harmonic, load_quasiharmonic
from heatup.analyze.trajectory import load_vdos, run_mdstats, run_transition, run_vdos
from heatup.core import ledger
from heatup.core.config import STATUS_FAIL, STATUS_MISSING, STATUS_OK, STATUS_WARN, cfg
from heatup.core.layout import PhaseDir, read_json

log = logging.getLogger("heatup")
_ORDER = {STATUS_OK: 0, STATUS_WARN: 1, STATUS_FAIL: 2}


def _worst(statuses) -> str:
    return max(statuses, key=_ORDER.__getitem__)


# ---------------------------------------------------------------------------
# mechanical
# ---------------------------------------------------------------------------

def assess_mechanical_stability(phase_dir) -> dict:
    phase = PhaseDir.of(phase_dir)
    data = read_json(phase.elastic_json)
    if not data:
        return {"available": False, "born_stable": None, "status": STATUS_MISSING,
                "message": "Elastic tensor not computed (run step 'elastic')."}

    g = cfg.gates
    C = np.array(data["C_GPa"])
    mod = data["moduli"]
    B, G = mod["bulk_hill_GPa"], mod["shear_hill_GPa"]
    eig = np.linalg.eigvalsh(C)
    min_eig = float(eig.min())
    born = bool(min_eig > g.born_min_eigenvalue_GPa)

    reasons: list[str] = []
    if not born:
        reasons.append(f"Born criterion violated (min eigenvalue {min_eig:.2f} GPa)")
    if B <= g.bulk_fail_GPa:
        reasons.append(f"B = {B:.1f} GPa (<= {g.bulk_fail_GPa:g})")
    elif B < g.bulk_warn_GPa:
        reasons.append(f"B = {B:.1f} GPa (very soft)")
    if G < g.shear_warn_GPa and born:
        reasons.append(f"G = {G:.1f} GPa (low shear stiffness)")
    resid = float(np.abs(data.get("residual_stress_GPa", [0] * 6)).max())
    if resid > 0.2:
        reasons.append(f"residual stress {resid:.2f} GPa - cell not at equilibrium")

    if not born or B <= g.bulk_fail_GPa:
        status = STATUS_FAIL
    elif reasons:
        status = STATUS_WARN
    else:
        status = STATUS_OK
    msg = "; ".join(reasons) or (
        f"Born stable: min eigenvalue {min_eig:.1f} GPa, B={B:.1f}, G={G:.1f} GPa (Hill).")
    return {
        "available": True, "born_stable": born, "min_eigenvalue_GPa": min_eig,
        "eigenvalues_GPa": eig.tolist(), "C_GPa": C.tolist(),
        "bulk_GPa": B, "shear_GPa": G, "youngs_GPa": mod["youngs_hill_GPa"],
        "poisson": mod["poisson_hill"], "moduli": mod,
        "status": status, "message": msg,
    }


# ---------------------------------------------------------------------------
# structural transition
# ---------------------------------------------------------------------------

def transition_verdict(m: dict) -> tuple[str, list[str]]:
    """Structural-transition status and notes of one run's metrics (``trajectory.transition_metrics``) under ``cfg.gates``."""
    g = cfg.gates
    fails, warns = [], []
    if m["rms_A"] > g.transition_rms_A:
        fails.append(f"rms block displacement {m['rms_A']:.2f} A > {g.transition_rms_A:g}")
    if m["max_A"] > g.transition_max_A:
        fails.append(f"max atomic displacement {m['max_A']:.2f} A > {g.transition_max_A:g}")
    if m["cell_shape_strain"] > g.cell_shape_strain:
        fails.append(f"cell shape strain {m['cell_shape_strain']:.3f} > {g.cell_shape_strain:g}")
    if abs(m["cell_volume_change"]) > g.cell_volume_change:
        fails.append(f"volume change {100 * m['cell_volume_change']:+.1f}% (> {100 * g.cell_volume_change:g}%)")
    sb1 = m["symmetry_break_last_A"]
    if np.isfinite(sb1) and sb1 > g.symmetry_break_A:
        warns.append(f"symmetry of the initial structure broken by {sb1:.2f} A "
                     f"(was {m['symmetry_break_first_A']:.2f} A at the start)")
    if m["too_short"]:
        warns.append("production run shorter than two averaging blocks")
    if m["rms_A"] > 0.5 * g.transition_rms_A and not fails:
        warns.append(f"rms block displacement {m['rms_A']:.2f} A is close to the threshold")
    if fails:
        return STATUS_FAIL, fails
    return (STATUS_WARN if warns else STATUS_OK), warns


def assess_structural_transition(phase_dir, force: bool | None = None) -> dict:
    """Structural-transition check over every finished MD run (planned temperatures)."""
    phase = PhaseDir.of(phase_dir)
    temps = ledger.md_temperatures(phase)
    if not temps:
        return {"available": False, "status": STATUS_MISSING, "sim_results": [],
                "message": "No finished MD run found - run step 'md' first (notebook 2)."}
    sims = []
    for T in temps:
        try:
            m = dict(run_transition(phase, T, force=force))
            diffusion = (run_mdstats(phase, T, force=force) or {}).get("diffusion", {})
        except Exception as exc:  # noqa: BLE001
            log.warning("structural-transition check failed for %s %dK: %s", phase.tag, T, exc)
            continue
        status, notes = transition_verdict(m)
        diffusive = [sp for sp, v in diffusion.items() if v.get("mobile")]
        m.update({"status": status, "notes": notes, "diffusive_species": diffusive,
                  "undeclared_diffusive_species": [sp for sp in diffusive if sp not in cfg.analysis.mobile_species]})
        sims.append(m)
    if not sims:
        return {"available": False, "status": STATUS_MISSING, "sim_results": [],
                "message": "MD trajectories could not be analysed."}
    worst = max(sims, key=lambda r: _ORDER[r["status"]])
    failed = [f"{int(r['temperature_K'])}K" for r in sims if r["status"] == STATUS_FAIL]
    if worst["status"] == STATUS_FAIL:
        msg = f"Structural transition/instability at {', '.join(failed)}: " + "; ".join(worst["notes"])
    elif worst["status"] == STATUS_WARN:
        msg = "; ".join(worst["notes"])
    else:
        msg = (f"Structure retained at all {len(sims)} MD temperature(s) "
               f"(max rms block displacement {max(r['rms_A'] for r in sims):.2f} A).")
    slim = [{k: v for k, v in r.items() if k not in ("dacf_A2", "dacf_lag_fs")} for r in sims]
    return {"available": True, "status": worst["status"], "message": msg,
            "sim_results": slim, "max_rms_A": max(r["rms_A"] for r in sims), "failed_temperatures": failed}


# ---------------------------------------------------------------------------
# vibrational
# ---------------------------------------------------------------------------

def _status_of_fraction(f: float) -> str:
    g = cfg.gates
    return STATUS_FAIL if f >= g.imag_frac_fail else STATUS_WARN if f >= g.imag_frac_warn else STATUS_OK


def _harmonic_evidence(phase: PhaseDir) -> dict | None:
    ha = load_harmonic(phase)
    if not ha:
        return None
    rep = vibrations.harmonic_report(ha)
    rep["status"] = _status_of_fraction(rep["imag_fraction"])
    return rep


def _quasiharmonic_evidence(phase: PhaseDir) -> dict | None:
    q = load_quasiharmonic(phase)
    if not q:
        return None
    rep = vibrations.quasiharmonic_report(q)
    rep["status"] = _status_of_fraction(rep["visited_max_imag"])
    return rep


def _anharmonic_evidence(phase: PhaseDir) -> dict | None:
    """Soft-mode verdict of every MD run (VDOS)."""
    g = cfg.gates
    runs = []
    for T in ledger.md_temperatures(phase):
        try:
            run_vdos(phase, T)               # cached; recomputed when the trajectory/config changed
        except Exception as exc:  # noqa: BLE001
            log.warning("%s %dK: VDOS failed: %s", phase.tag, T, exc)
        runs.append((T, load_vdos(phase, T)))
    runs = [(T, v) for T, v in runs if v]
    if not runs:
        return None
    results, notes = [], []
    for T, v in runs:
        win = max(g.zero_window_meV, 1.5 * v["resolution_meV"])
        soft = vibrations.soft_report_md(v, win, T)
        z = soft["zero_frac"]
        st = (STATUS_FAIL if z >= g.zero_frac_fail else STATUS_WARN if z >= g.zero_frac_warn else STATUS_OK)
        q = []
        if g.min_production_ps is not None and v["production_ps"] < g.min_production_ps:
            q.append(f"short run ({v['production_ps']:.1f} ps < {g.min_production_ps:g} ps)")
        if v["alias_weight_top10pct"] > 1e-3:
            q.append("aliasing near Nyquist - lower md.nblock")
        if q and st == STATUS_OK:
            st = STATUS_WARN
        k = max(1, len(v["modes"].omega) // 500)
        results.append({
            "temperature_K": T, "zero_window_meV": win, "zero_frac": z, "per_species": soft["per_species"],
            "dominant_species": soft["dominant_species"],
            "resolution_meV": v["resolution_meV"], "production_ps": v["production_ps"],
            "status": st, "quality_notes": q, "T_kin_mean_K": v.get("T_kin_mean_K"),
            "omega_meV": v["modes"].omega[::k], "weight": v["modes"].weight[::k] * k,
        })
        notes += [f"{T} K: {x}" for x in q]
    return {"runs": results, "notes": notes, "status": _worst(r["status"] for r in results)}


def assess_vibrational_stability(phase_dir, sources: list | None = None, save_figure: bool | None = None) -> dict:
    """Vibrational check from the evidence in *sources* (default ``cfg.vibrational.sources``; see the module docstring).

    The result keeps the keys the reports read: ``status``, ``message``, ``evidence`` (the source that decided), ``ha``
    (harmonic summary), ``vdos`` (one entry per MD temperature), ``md_temperatures``; and adds ``sources``, ``harmonic``,
    ``quasiharmonic``, ``anharmonic`` (each source's own report with its status), ``missing_sources`` and ``figure``."""
    phase = PhaseDir.of(phase_dir)
    g = cfg.gates
    sources = list(cfg.vibrational.sources if sources is None else sources)
    bad = [x for x in sources if x not in vibrations.LEVEL_ORDER]
    if bad or not sources:
        raise ValueError(f"vibrational sources {sources!r}: choose from {list(vibrations.LEVEL_ORDER)}")
    sources = [x for x in vibrations.LEVEL_ORDER if x in sources]        # increasing completeness, whatever the order given

    ev = {"harmonic": _harmonic_evidence, "quasiharmonic": _quasiharmonic_evidence,
          "anharmonic": _anharmonic_evidence}
    found = {s: ev[s](phase) for s in sources}
    have = [s for s in sources if found[s]]
    missing = [s for s in sources if not found[s]]
    if not have:
        return {"available": False, "status": STATUS_MISSING, "sources": sources, "missing_sources": missing,
                "message": "No vibrational evidence found for " + ", ".join(sources) + " (run the matching steps)."}

    deciding = have[-1]                                                  # the most complete evidence that exists
    lines: list[str] = []
    if deciding == "anharmonic":
        runs = found["anharmonic"]["runs"]
        worst = max(runs, key=lambda r: r["zero_frac"])
        status = found["anharmonic"]["status"]
        if status == STATUS_FAIL:
            msg = (f"Soft-mode weight {100 * worst['zero_frac']:.1f}% within |w|<{worst['zero_window_meV']:.1f} meV at "
                   f"{worst['temperature_K']:.0f} K, mostly on {worst['dominant_species']} (fail >= {100 * g.zero_frac_fail:g}%).")
        elif status == STATUS_WARN:
            msg = (f"Possible soft modes: {100 * worst['zero_frac']:.2f}% within |w|<{worst['zero_window_meV']:.1f} meV at "
                   f"{worst['temperature_K']:.0f} K" + ("; " + "; ".join(found["anharmonic"]["notes"]) if found["anharmonic"]["notes"] else "") + ".")
        else:
            msg = (f"No soft-mode weight in the MD spectra (max {100 * worst['zero_frac']:.3f}% within "
                   f"|w|<{worst['zero_window_meV']:.1f} meV).")
        lower = [s for s in have if s != "anharmonic" and found[s]["status"] != STATUS_OK]
        for s in lower:
            r = found[s]
            frac = r["imag_fraction"] if s == "harmonic" else r["visited_max_imag"]
            lines.append(f"{s} has {100 * frac:.1f}% imaginary weight"
                         + (f" ({r['branch_summary']})" if s == "harmonic" and r["has_band"] else "") + ": "
                         + ("stabilised by anharmonicity at finite T." if status == STATUS_OK else "consistent with the MD result."))
    elif deciding == "quasiharmonic":
        r = found["quasiharmonic"]
        status = r["status"]
        msg = (f"Quasi-harmonic: {100 * r['visited_max_imag']:.2f}% imaginary weight at the volumes visited up to "
               f"{max(r['temperatures_K']):g} K.  Anharmonic stability (MD) not tested.")
        if "harmonic" in have and found["harmonic"]["status"] != STATUS_OK:
            lines.append(f"harmonic (0 K) has {100 * found['harmonic']['imag_fraction']:.1f}% imaginary weight: "
                         + found["harmonic"]["branch_summary"] + ".")
    else:
        r = found["harmonic"]
        status = r["status"]
        msg = (f"Harmonic only: {100 * r['imag_fraction']:.2f}% imaginary modes (max {r['max_imag_meV']:.1f} meV); "
               f"{r['branch_summary']}.  Finite-T stability not tested.")
    if lines:
        msg += " " + " ".join(lines)
    if missing:
        msg += f" [selected but missing: {', '.join(missing)}]"

    ha_info = found.get("harmonic")
    out = {"available": True, "status": status, "message": msg, "sources": sources, "evidence": deciding,
           "missing_sources": missing, "harmonic": ha_info, "quasiharmonic": found.get("quasiharmonic"),
           "anharmonic": found.get("anharmonic"),
           "ha": None if not ha_info else {"imag_fraction": ha_info["imag_fraction"], "max_imag_meV": ha_info["max_imag_meV"],
                                           "status": ha_info["status"]},
           "vdos": (found.get("anharmonic") or {}).get("runs", []),
           "md_temperatures": [r["temperature_K"] for r in (found.get("anharmonic") or {}).get("runs", [])]}
    if save_figure is None:
        save_figure = cfg.vibrational.save_figures
    out["figure"] = vibrations.save_vibrational_figure(phase, out, sources) if save_figure else None
    return out
