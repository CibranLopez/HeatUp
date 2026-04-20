"""thermophasepy.vibrational
===========================
Gate 2: Vibrational stability from the anharmonic vibrational density of
states (VDOS).

The VDOS is extracted from ab-initio molecular dynamics (AIMD) trajectories
via the velocity autocorrelation function (VACF).  Unlike harmonic phonons
computed at 0 K from finite displacements, the AIMD-derived VDOS naturally
includes anharmonic effects and reflects the dynamical state of the crystal
at the simulation temperature.

Stability criterion
-------------------
A material is flagged as vibrationally unstable if the normalised VDOS
carries significant spectral weight within a narrow frequency window centred
on ω = 0.  Physically, this weight arises from:

* **Soft modes** — phonon branches that soften towards zero frequency,
  indicating a structural instability or imminent phase transition.
* **Pre-melting / superionic transition** — in ionic conductors, a subset
  of atoms (typically the mobile ion) can disorder well below the melting
  point, producing quasi-diffusive low-frequency spectral weight.
* **Numerical acoustic broadening** — a small, unavoidable contribution
  from the finite trajectory length that must be distinguished from the
  above physical effects using the thresholds in ``config``.

The VDOS is averaged across all AIMD temperatures available in
``aimd/<T>K/anharmonic_phonons/vdos.json``, which are expected to have
been produced by :mod:`thermophasepy.vdos` or the companion
``anharmonic_phonons.py`` module.

Cache layout::

    <sym_dir>/aimd/<T>K/anharmonic_phonons/
        vdos.json    ← {'omega_mev': [...], 'vdos': [...]}
        thermo.json  ← thermodynamic integrals at MD temperature
"""

from __future__ import annotations

import json
import os

import numpy as np

from thermophasepy import config


def assess_vibrational_stability(sym_dir: str) -> dict:
    """Check for soft / unstable modes in the anharmonic VDOS.

    Loads cached VDOS data from all available AIMD temperature sub-folders,
    averages them onto a common frequency grid, then evaluates the fraction
    of spectral weight within ``config.VIB_ZERO_WINDOW_MEV`` of ω = 0.

    Args:
        sym_dir: Path to the symmetry directory.

    Returns:
        Dict with keys:

        ``'available'`` (bool)
            True if at least one cached VDOS was found.
        ``'zero_window_frac'`` (float | None)
            Fraction of normalised VDOS within |ω| < ``VIB_ZERO_WINDOW_MEV``.
        ``'zero_window_mev'`` (float)
            The window half-width used (meV).
        ``'omega_mev'`` (list[float] | None)
            Frequency axis of the averaged VDOS (meV).
        ``'vdos'`` (list[float] | None)
            Averaged normalised VDOS g(ω), ∫g dω = 1.
        ``'n_sources'`` (int)
            Number of AIMD temperature sub-folders averaged.
        ``'sources'`` (list[str])
            Paths to the AIMD sub-folders used.
        ``'status'`` (str)
            ``'ok'`` | ``'warn'`` | ``'fail'`` | ``'missing'``.
        ``'message'`` (str)
            Human-readable explanation.
    """
    aimd_dir = os.path.join(sym_dir, "aimd")

    if not os.path.isdir(aimd_dir):
        return {
            "available"        : False,
            "zero_window_frac" : None,
            "status"           : config.STATUS_MISSING,
            "message"          : (
                "No AIMD directory found. Run MD simulation and "
                "anharmonic-phonon analysis first."
            ),
        }

    # ── Collect VDOS from all temperature sub-folders ─────────────────────
    vdos_list: list[tuple[np.ndarray, np.ndarray]] = []
    sources:   list[str] = []

    for temp_folder in sorted(os.listdir(aimd_dir)):
        if not temp_folder.endswith("K"):
            continue
        vdos_path = os.path.join(
            aimd_dir, temp_folder, "anharmonic_phonons", "vdos.json"
        )
        if not os.path.exists(vdos_path):
            continue
        try:
            with open(vdos_path) as fh:
                d = json.load(fh)
            om = np.array(d["omega_mev"], dtype=float)
            g  = np.array(d["vdos"],      dtype=float)
            norm = np.trapezoid(g, om)
            if norm > 0:
                g /= norm
            vdos_list.append((om, g))
            sources.append(os.path.join(aimd_dir, temp_folder))
        except Exception:
            continue

    if not vdos_list:
        return {
            "available"        : False,
            "zero_window_frac" : None,
            "status"           : config.STATUS_MISSING,
            "message"          : (
                "No cached anharmonic VDOS found in any AIMD sub-folder. "
                "Run anharmonic-phonon analysis first."
            ),
        }

    # ── Average VDOS onto the first grid ─────────────────────────────────
    om_ref, g_ref = vdos_list[0]
    g_avg = g_ref.copy()
    for om_i, g_i in vdos_list[1:]:
        g_avg += np.interp(om_ref, om_i, g_i, left=0.0, right=0.0)
    g_avg /= len(vdos_list)

    # Re-normalise after averaging.
    norm = np.trapezoid(g_avg, om_ref)
    if norm > 0:
        g_avg /= norm

    # ── Soft-mode fraction ────────────────────────────────────────────────
    win = config.VIB_ZERO_WINDOW_MEV
    zero_mask = np.abs(om_ref) <= win
    zero_frac = 0.0
    if zero_mask.any():
        zero_frac = float(np.trapezoid(g_avg[zero_mask], om_ref[zero_mask]))
    zero_frac = max(0.0, min(1.0, zero_frac))

    # ── Verdict ───────────────────────────────────────────────────────────
    if zero_frac >= config.VIB_ZERO_FRAC_FAIL:
        status = config.STATUS_FAIL
        message = (
            f"Soft-mode signature: {zero_frac * 100:.1f}% of VDOS weight within "
            f"|ω| < {win:.1f} meV (fail threshold {config.VIB_ZERO_FRAC_FAIL * 100:.0f}%). "
            f"Vibrationally unstable — dynamical instability persists at finite T."
        )
    elif zero_frac >= config.VIB_ZERO_FRAC_WARN:
        status = config.STATUS_WARN
        message = (
            f"Possible soft mode: {zero_frac * 100:.2f}% of VDOS weight within "
            f"|ω| < {win:.1f} meV. "
            f"Check VDOS plot — may indicate pre-melting or incipient phase transition."
        )
    else:
        status = config.STATUS_OK
        message = (
            f"No soft-mode signature detected ({zero_frac * 100:.3f}% within "
            f"|ω| < {win:.1f} meV). Vibrationally stable."
        )

    return {
        "available"        : True,
        "zero_window_frac" : zero_frac,
        "zero_window_mev"  : win,
        "omega_mev"        : om_ref.tolist(),
        "vdos"             : g_avg.tolist(),
        "n_sources"        : len(vdos_list),
        "sources"          : sources,
        "status"           : status,
        "message"          : message,
    }
