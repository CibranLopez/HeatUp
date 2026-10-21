"""heatup.analyze.gates
=======================
The verdict of Gates 0-2 from cached step results (the hull, Gate 3, is :mod:`heatup.analyze.hull`).  Nothing
here computes anything expensive and every threshold (``cfg.gates``) is applied on each call, so thresholds can
change without recomputing.

Gate 1 - mechanical (Born-Huang)
    The necessary and sufficient condition for a crystal at (near) zero stress is a positive-definite 6x6
    stiffness matrix (Mouhat & Coudert, PRB 90, 224104, 2014): all eigenvalues of C > 0.  The Hill-averaged
    bulk and shear moduli are checked against soft-material thresholds.

Gate 0 - did the structure survive the MD run?
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

Gate 2 - vibrational (dynamical) stability, two lines of evidence
    **MD / VDOS** (finite T, anharmonic): the weight of the mass-weighted VDOS inside ``|w| < window``.  A solid
    has g(w) ~ w^2 so the fraction is ~1e-4; a collapsing soft mode or a liquid-like sub-lattice puts weight at
    w = 0.  The window is widened to 1.5x the spectral resolution of the run.  Species in
    ``analysis.mobile_species`` are excluded.
    **HA** (0 K, harmonic): fraction of modes with imaginary frequency beyond ``phonon.imag_tol_meV``.  Imaginary
    harmonic modes at 0 K are *not* by themselves a failure at finite T (cubic perovskites, bcc metals are
    stabilised by anharmonicity): when an MD verdict exists it decides, the HA result is attached as information.
"""

from __future__ import annotations

import logging

import numpy as np

from heatup.analyze.thermo import load_ha
from heatup.analyze.trajectory import load_vdos, run_mdstats, run_transition, run_vdos
from heatup.core import ledger
from heatup.core.config import STATUS_FAIL, STATUS_MISSING, STATUS_OK, STATUS_WARN, cfg
from heatup.core.layout import PhaseDir, read_json

log = logging.getLogger("heatup")
_ORDER = {STATUS_OK: 0, STATUS_WARN: 1, STATUS_FAIL: 2}




# ---------------------------------------------------------------------------
# Gate 1 - mechanical
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
# Gate 0 - structural transition
# ---------------------------------------------------------------------------

def transition_verdict(m: dict) -> tuple[str, list[str]]:
    """Gate-0 status and notes of one run's metrics (``trajectory.transition_metrics``) under ``cfg.gates``."""
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
    """Gate 0 over every finished MD run (planned temperatures)."""
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
            log.warning("Gate 0 failed for %s %dK: %s", phase.tag, T, exc)
            continue
        status, notes = transition_verdict(m)
        diffusive = [sp for sp, v in diffusion.items() if v.get("mobile")]
        m.update({"status": status, "notes": notes, "diffusive_species": diffusive,
                  "undeclared_diffusive_species": [sp for sp in diffusive if sp not in cfg.analysis.mobile_species]})
        sims.append(m)
    if not sims:
        return {"available": False, "status": STATUS_MISSING, "sim_results": [],
                "message": "MD trajectories could not be analysed."}
    order = {STATUS_OK: 0, STATUS_WARN: 1, STATUS_FAIL: 2}
    worst = max(sims, key=lambda r: order[r["status"]])
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
# Gate 2 - vibrational
# ---------------------------------------------------------------------------



def _framework_zero_fraction(v: dict, window: float, mobile: list[str]) -> tuple[float, dict]:
    """Atom-weighted fraction of modes with |w| < window, over the non-mobile species."""
    num = den = 0.0
    per = {}
    for sp, p in v["partial"].items():
        om = np.asarray(p["low_omega_meV"])
        w = np.asarray(p["low_weight"])
        z = float(w[np.abs(om) < window].sum() / p["total_weight"])
        per[sp] = z
        if sp in mobile:
            continue
        num += z * p["n_atoms"]
        den += p["n_atoms"]
    if den == 0:                       # everything declared mobile -> use all
        num = sum(per[sp] * p["n_atoms"] for sp, p in v["partial"].items())
        den = sum(p["n_atoms"] for p in v["partial"].values())
    return num / den, per


def assess_vibrational_stability(phase_dir) -> dict:
    phase = PhaseDir.of(phase_dir)
    g = cfg.gates
    ha = load_ha(phase)
    temps = ledger.md_temperatures(phase)
    runs = []
    for T in temps:
        try:
            run_vdos(phase, T)               # cached; recomputed when the trajectory/config changed
        except Exception as exc:  # noqa: BLE001
            log.warning("%s %dK: VDOS failed: %s", phase.tag, T, exc)
        runs.append((T, load_vdos(phase, T)))
    runs = [(T, v) for T, v in runs if v]

    if not ha and not runs:
        return {"available": False, "status": STATUS_MISSING,
                "message": "Neither HA phonons nor MD-VDOS available."}

    notes: list[str] = []
    ha_status = None
    ha_info = None
    if ha:
        f = ha["imag_fraction"]
        ha_status = (STATUS_FAIL if f >= g.imag_frac_fail else
                     STATUS_WARN if f >= g.imag_frac_warn else STATUS_OK)
        ha_info = {"imag_fraction": f, "max_imag_meV": ha["max_imag_meV"], "status": ha_status}

    vdos_results = []
    md_status = None
    for T, v in runs:
        win = max(g.zero_window_meV, 1.5 * v["resolution_meV"])
        z, per = _framework_zero_fraction(v, win, cfg.analysis.mobile_species)
        st = (STATUS_FAIL if z >= g.zero_frac_fail else
              STATUS_WARN if z >= g.zero_frac_warn else STATUS_OK)
        q = []
        if g.min_production_ps is not None and v["production_ps"] < g.min_production_ps:
            q.append(f"short run ({v['production_ps']:.1f} ps < {g.min_production_ps:g} ps)")
        if v["alias_weight_top10pct"] > 1e-3:
            q.append("aliasing near Nyquist - lower md.nblock")
        if q and st == STATUS_OK:
            st = STATUS_WARN
        vdos_results.append({
            "temperature_K": T, "zero_window_meV": win, "zero_frac": z, "per_species": per,
            "resolution_meV": v["resolution_meV"], "production_ps": v["production_ps"],
            "status": st, "quality_notes": q, "T_kin_mean_K": v.get("T_kin_mean_K"),
            "omega_meV": v["modes"].omega[:: max(1, len(v["modes"].omega) // 500)],
            "weight": v["modes"].weight[:: max(1, len(v["modes"].omega) // 500)]
                      * max(1, len(v["modes"].omega) // 500),
        })
        notes += [f"{T} K: {x}" for x in q]
    if vdos_results:
        md_status = max((r["status"] for r in vdos_results), key=_ORDER.get)

    if md_status is not None:
        status = md_status
        worst = max(vdos_results, key=lambda r: r["zero_frac"])
        if status == STATUS_FAIL:
            msg = (f"Soft-mode weight {100 * worst['zero_frac']:.1f}% within |w|<"
                   f"{worst['zero_window_meV']:.1f} meV at {worst['temperature_K']:.0f} K "
                   f"(fail >= {100 * g.zero_frac_fail:g}%).")
        elif status == STATUS_WARN:
            msg = (f"Possible soft modes: {100 * worst['zero_frac']:.2f}% within |w|<"
                   f"{worst['zero_window_meV']:.1f} meV at {worst['temperature_K']:.0f} K"
                   + ("; " + "; ".join(notes) if notes else "") + ".")
        else:
            msg = (f"No soft-mode weight in the MD spectra (max "
                   f"{100 * worst['zero_frac']:.3f}% within |w|<{worst['zero_window_meV']:.1f} meV).")
        if ha_status in (STATUS_WARN, STATUS_FAIL):
            msg += (f" HA has {100 * ha_info['imag_fraction']:.1f}% imaginary modes "
                    f"(max {ha_info['max_imag_meV']:.1f} meV): "
                    + ("stabilised by anharmonicity at finite T." if status == STATUS_OK
                       else "consistent with the MD result."))
    else:
        status = ha_status
        msg = (f"HA only (no MD): {100 * ha_info['imag_fraction']:.2f}% imaginary modes "
               f"(max {ha_info['max_imag_meV']:.1f} meV). Finite-T stability not tested.")
    return {"available": True, "status": status, "message": msg, "ha": ha_info,
            "vdos": vdos_results, "md_temperatures": [r["temperature_K"] for r in vdos_results],
            "evidence": "MD" if md_status is not None else "HA"}
