"""heatup.vibrational
======================
Gate 2 - vibrational (dynamical) stability.

Two independent lines of evidence, combined:

**MD / VDOS (finite temperature, anharmonic)** - the weight of the mass-weighted
VDOS inside ``|w| < window`` (fraction of all modes).  A solid has
g(w) ~ w^2 so the fraction is ~1e-4; a soft mode that collapses, or a
liquid-like / diffusive sub-lattice, puts weight at w = 0.  The window is
widened to 1.5x the spectral resolution of the run (otherwise leakage of the
finite trajectory is read as "soft-mode weight"; v0.2 had no window function
and no resolution check, and normalised the VDOS to 1).  Species listed in
``gates.mobile_species`` are excluded (a superionic Li sub-lattice *should*
have weight at w = 0).

**HA (0 K, harmonic)** - fraction of modes with imaginary frequency beyond the
noise tolerance (``phonon.imag_tol_meV``).  Imaginary harmonic modes at 0 K
are *not* by themselves a failure at finite T (cubic perovskites, bcc metals
are stabilised by anharmonicity): when an MD verdict exists it decides, the
harmonic result is attached as information.  Without MD, the HA decides.
"""

from __future__ import annotations

import logging

import numpy as np

from heatup.config import (STATUS_FAIL, STATUS_MISSING, STATUS_OK, STATUS_WARN, cfg)
from heatup.paths import PhaseDir
from heatup.phonons import load_ha
from heatup.vdos import load_vdos, run_vdos

log = logging.getLogger("heatup")
_ORDER = {STATUS_OK: 0, STATUS_WARN: 1, STATUS_FAIL: 2}


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
    temps = phase.md_temperatures()
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
        z, per = _framework_zero_fraction(v, win, cfg.gates.mobile_species)
        st = (STATUS_FAIL if z >= g.zero_frac_fail else
              STATUS_WARN if z >= g.zero_frac_warn else STATUS_OK)
        q = []
        if v["production_ps"] < g.min_production_ps:
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
