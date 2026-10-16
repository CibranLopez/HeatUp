"""heatup.vdos
===============
Vibrational density of states from MD (velocity autocorrelation).

    g(w)  ~  FFT[ W(t) * sum_i m_i <v_i(0) . v_i(t)> ]

* velocities come from the stored **momenta** (not from position differences),
* the VACF is **mass-weighted** (the unweighted one over-weights light atoms in
  a classical trajectory: <v^2> = kT/m),
* computed with FFT autocorrelation (O(T log T), v0.2 used ``np.correlate`` per
  atom and component: O(N T^2)),
* a Hann lag window removes the spectral leakage that put artificial weight
  near w = 0 (this was the quantity the v0.2 soft-mode gate measured!),
* the result is a :class:`~heatup.vibthermo.ModeSet` with 3 modes per atom,
* a Nyquist/aliasing check and the spectral resolution are reported,
* partial (per-species) spectra are stored, so mobile species can be excluded
  from the soft-mode test.

The VDOS of a *classical* trajectory equals the harmonic DOS in the harmonic
limit at any temperature; at finite T it contains the effect of anharmonic
frequency renormalisation (the "effective harmonic" spectrum).  F_vib built
from it uses the quantum oscillator formula (standard practice for
temperature-dependent effective potentials).
"""

from __future__ import annotations

import logging
import os

import numpy as np

from heatup import manifest
from heatup.config import cfg, forced
from heatup.constants import H_MEV_FS
from heatup.md_analysis import TrajData, load_trajectory
from heatup.paths import PhaseDir, read_json, write_json
from heatup.vibthermo import ModeSet

log = logging.getLogger("heatup")


def _vacf_spectrum(x: np.ndarray, dt_fs: float, max_lag: int, pad: int = 4):
    """x: (T, M) mass-weighted velocities (p/sqrt(m)). Returns (omega_meV, S) for w >= 0."""
    T = x.shape[0]
    n = 1 << (2 * T - 1).bit_length()
    X = np.fft.rfft(x, n=n, axis=0)
    ac = np.fft.irfft((X * np.conj(X)).sum(axis=1), n=n)[:max_lag]
    ac = ac / (T - np.arange(max_lag))                              # unbiased
    win = 0.5 * (1.0 + np.cos(np.pi * np.arange(max_lag) / max_lag))   # Hann (lag window)
    ac = ac * win
    # even function: S(w) = c0 + 2 sum_k c_k cos(w k dt)  =  2 Re rfft(c) - c0
    L = 1 << ((max_lag * pad) - 1).bit_length()
    S = 2.0 * np.fft.rfft(ac, n=L).real - ac[0]
    freq = np.fft.rfftfreq(L, d=dt_fs)                              # cycles/fs
    return freq * H_MEV_FS, S


LOW_REGION_MEV = 30.0


def compute_vdos(td: TrajData, max_lag_ps: float = 10.0) -> dict:
    """VDOS of the production part of *td* (already equilibration-stripped or not)."""
    p = td.production() if td.n_equil else td
    if p.mom is None:
        raise ValueError("trajectory has no momenta - rerun MD (v0.3 stores them)")
    if p.n_frames < 64:
        raise ValueError(f"only {p.n_frames} production frames")
    x = p.mom / np.sqrt(p.masses)[None, :, None]                   # (T, N, 3)
    x = x - x.mean(axis=0, keepdims=True)                          # remove DC (drift) component
    max_lag = int(min(p.n_frames // 2, max_lag_ps * 1000.0 / p.dt_fs))
    om, S = _vacf_spectrum(x.reshape(p.n_frames, -1), p.dt_fs, max_lag)

    nyq = H_MEV_FS / (2.0 * p.dt_fs)
    resolution = 2.0 * H_MEV_FS / (max_lag * p.dt_fs)               # ~ Hann main-lobe width
    keep = om <= nyq * 0.999
    om, S = om[keep], S[keep]
    # coarse regridding (keep file small, ~0.25 meV bins or the resolution)
    step = max(resolution / 4.0, 0.1)
    grid = np.arange(0.0, om.max(), step)
    g = np.interp(grid, om, S)
    ms = ModeSet.from_dos(grid, g, source="VDOS", T_md=p.T_target)

    top = (grid > 0.9 * nyq)
    alias_frac = float(ms.weight[top].sum() / ms.weight.sum())
    partial = {}
    for sp in sorted(set(p.symbols)):
        m = p.species_mask(sp)
        om_s, S_s = _vacf_spectrum(x[:, m, :].reshape(p.n_frames, -1), p.dt_fs, max_lag)
        g_s = np.interp(grid, om_s[om_s <= nyq * 0.999], S_s[om_s <= nyq * 0.999])
        pm = ModeSet.from_dos(grid, g_s, source=f"VDOS:{sp}")
        lo = grid <= LOW_REGION_MEV
        partial[sp] = {"zero_window_frac": pm.low_weight_fraction(cfg.gates.zero_window_meV),
                       "n_atoms": int(m.sum()),
                       "low_omega_meV": grid[lo], "low_weight": pm.weight[lo],   # full resolution, for Gate 2
                       "total_weight": float(pm.weight.sum()),
                       "omega_meV": grid[:: max(1, len(grid) // 400)],
                       "weight": pm.weight[:: max(1, len(grid) // 400)] * max(1, len(grid) // 400)}
    Tk = p.kinetic_temperature()
    return {
        "modes": ms, "partial": partial, "resolution_meV": float(resolution),
        "nyquist_meV": float(nyq), "alias_weight_top10pct": alias_frac,
        "n_frames": p.n_frames, "production_ps": p.duration_ps, "dt_fs": p.dt_fs,
        "T_md_K": p.T_target, "T_kin_mean_K": float(np.nanmean(Tk)) if np.isfinite(Tk).any() else None,
    }


def run_vdos(phase_dir, temperature: float, force: bool | None = None) -> dict | None:
    """Compute (or load) ``md/<T>K/vdos.json``."""
    phase = PhaseDir.of(phase_dir)
    sim = phase.md_run(temperature)
    out = os.path.join(sim, "vdos.json")
    traj = os.path.join(sim, "output.traj")
    inputs = [traj]
    force = forced("analysis", force)
    if (not force and os.path.exists(out)
            and manifest.is_fresh(sim, "analysis", phase.material, inputs)):
        return load_vdos(phase, temperature)
    td = load_trajectory(sim)
    res = compute_vdos(td)
    d = dict(res)
    d["modes"] = res["modes"].to_dict(max_points=3000)
    d["partial"] = {k: {kk: vv for kk, vv in v.items()} for k, v in res["partial"].items()}
    write_json(out, d, indent=None)
    manifest.write_manifest(sim, "analysis", phase.material, inputs)
    if res["alias_weight_top10pct"] > 1e-3:
        log.warning("%s %dK: %.2f%% of VDOS weight near the Nyquist limit (%.0f meV) - "
                    "decrease md.nblock", phase.tag, temperature,
                    100 * res["alias_weight_top10pct"], res["nyquist_meV"])
    return res


def load_vdos(phase_dir, temperature: float) -> dict | None:
    sim = PhaseDir.of(phase_dir).md_run(temperature)
    d = read_json(os.path.join(sim, "vdos.json"))
    if not d:
        return None
    d["modes"] = ModeSet.from_dict(d["modes"])
    return d
