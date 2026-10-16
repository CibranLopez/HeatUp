"""heatup.md_analysis
======================
Trajectory loading and basic observables (thermo time series, MSD / diffusion).

``TrajData`` holds the arrays every downstream analysis needs (Gate 0, VDOS,
diffusion) in one pass over ``output.traj``.

Diffusion is computed here (FFT multi-origin MSD per species, drift-corrected)
instead of ``pymatgen-analysis-diffusion``: that package is an extra heavy
dependency, assumes a fixed cell (invalid in NPT) and has changed API
repeatedly.
"""

from __future__ import annotations

import dataclasses as dc
import logging
import os

import numpy as np
from ase.io.trajectory import Trajectory

from heatup.config import cfg, forced
from heatup.constants import EV_A3_TO_GPA, KB_EV
from heatup.paths import PhaseDir, read_json, write_json

log = logging.getLogger("heatup")


@dc.dataclass
class TrajData:
    frac: np.ndarray          # (T, N, 3) wrapped fractional coordinates
    cell: np.ndarray          # (T, 3, 3) row vectors
    mom: np.ndarray | None    # (T, N, 3) ASE momenta
    masses: np.ndarray        # (N,) amu
    symbols: list
    epot: np.ndarray          # (T,) eV (nan if unavailable)
    stress: np.ndarray | None  # (T, 6) eV/A^3 (config part only)
    dt_fs: float              # time between stored frames
    T_target: float
    n_equil: int

    @property
    def n_frames(self) -> int:
        return self.frac.shape[0]

    @property
    def n_atoms(self) -> int:
        return self.frac.shape[1]

    def production(self) -> "TrajData":
        s = slice(self.n_equil, None)
        return dc.replace(
            self, frac=self.frac[s], cell=self.cell[s],
            mom=None if self.mom is None else self.mom[s], epot=self.epot[s],
            stress=None if self.stress is None else self.stress[s], n_equil=0)

    @property
    def duration_ps(self) -> float:
        return self.n_frames * self.dt_fs / 1000.0

    def species_mask(self, species) -> np.ndarray:
        sp = set([species] if isinstance(species, str) else species)
        return np.array([s in sp for s in self.symbols])

    def kinetic_temperature(self) -> np.ndarray:
        if self.mom is None:
            return np.full(self.n_frames, np.nan)
        ke = 0.5 * ((self.mom ** 2).sum(-1) / self.masses[None, :]).sum(-1)
        return 2.0 * ke / ((3 * self.n_atoms - 3) * KB_EV)


def load_trajectory(sim_dir: str) -> TrajData:
    path = os.path.join(sim_dir, "output.traj")
    meta = read_json(os.path.join(sim_dir, "md.json"))
    if not meta:
        raise FileNotFoundError(f"{sim_dir}/md.json missing")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise FileNotFoundError(path)
    stride = max(1, cfg.md.stride)
    frac, cell, mom, epot, stress = [], [], [], [], []
    masses = symbols = None
    with Trajectory(path) as tr:
        for i, a in enumerate(tr):
            if i % stride:
                continue
            if masses is None:
                masses, symbols = a.get_masses(), a.get_chemical_symbols()
            frac.append(a.get_scaled_positions(wrap=False))
            cell.append(a.cell.array.copy())
            mom.append(a.get_momenta() if a.has("momenta") else None)
            try:
                epot.append(a.get_potential_energy())
                stress.append(a.get_stress(voigt=True))
            except Exception:  # noqa: BLE001
                epot.append(np.nan)
                stress.append(None)
    has_mom = all(m is not None for m in mom)
    has_stress = all(s is not None for s in stress)
    dt = meta["frame_dt_fs"] * stride
    n_equil = int(round(cfg.md.equil_ps * 1000.0 / dt))
    return TrajData(
        frac=np.array(frac), cell=np.array(cell),
        mom=np.array(mom) if has_mom else None, masses=masses, symbols=symbols,
        epot=np.array(epot, float), stress=np.array(stress) if has_stress else None,
        dt_fs=dt, T_target=float(meta["temperature_K"]), n_equil=min(n_equil, len(frac) - 2))


def unwrap_frac(frac: np.ndarray) -> np.ndarray:
    """Cumulative minimum-image unwrapping of fractional coordinates (T,N,3)."""
    d = np.diff(frac, axis=0)
    d -= np.round(d)
    return np.concatenate([frac[:1], frac[:1] + np.cumsum(d, axis=0)], axis=0)


def autocorr_fft(x: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    """Unbiased autocorrelation along axis 0 of x (T, M) summed over M -> (max_lag,)."""
    T = x.shape[0]
    max_lag = max_lag or T
    n = 1 << (2 * T - 1).bit_length()
    X = np.fft.rfft(x, n=n, axis=0)
    ac = np.fft.irfft(X * np.conj(X), n=n, axis=0)[:max_lag].sum(axis=1)
    return ac / (T - np.arange(max_lag))


def msd_fft(r: np.ndarray) -> np.ndarray:
    """Multi-time-origin MSD(tau) of positions r (T, n, 3), averaged over atoms."""
    T, n, _ = r.shape
    flat = r.reshape(T, n * 3)
    S2 = autocorr_fft(flat) / n
    D = (r ** 2).sum(-1)                       # (T, n)
    D = np.concatenate([D, np.zeros((1, n))], axis=0)
    Q = 2.0 * D.sum()
    S1 = np.zeros(T)
    for m in range(T):
        Q -= D[m - 1].sum() + D[T - m].sum()
        S1[m] = Q / (T - m)
    return (S1 / n) - 2.0 * S2


def thermo_series(td: TrajData) -> dict:
    """T(t), P(t) (virial part), V(t), E_pot(t) summaries of the production part."""
    prod = td.production()
    t_ps = (np.arange(td.n_frames) * td.dt_fs) / 1000.0
    Tk = td.kinetic_temperature()
    V = np.abs(np.linalg.det(td.cell))
    P = (-td.stress[:, :3].sum(1) / 3.0 * EV_A3_TO_GPA) if td.stress is not None else None
    sl = slice(td.n_equil, None)

    def stat(x):
        if x is None:
            return None
        x = np.asarray(x)[sl]
        t = t_ps[sl]
        slope = np.polyfit(t, x, 1)[0] if len(x) > 3 else 0.0
        return {"mean": float(np.nanmean(x)), "std": float(np.nanstd(x)),
                "drift_per_ps": float(slope)}

    return {
        "T_K": stat(Tk), "P_GPa": stat(P), "V_A3": stat(V),
        "Epot_eV": stat(td.epot), "n_production_frames": prod.n_frames,
        "production_ps": prod.duration_ps,
        "series": {"t_ps": t_ps, "T_K": Tk, "P_GPa": P, "V_A3": V, "Epot_eV": td.epot,
                   "n_equil": td.n_equil},
    }


def diffusion_by_species(td: TrajData, min_msd_A2: float = 1.0) -> dict:
    """Tracer diffusion per species from the production part (drift-corrected)."""
    prod = td.production()
    u = unwrap_frac(prod.frac)
    h = prod.cell.mean(axis=0)
    r = u @ h                                    # (T, N, 3) Cartesian on the mean cell
    w = (prod.masses / prod.masses.sum())[None, :, None]
    r = r - (r * w).sum(axis=1, keepdims=True)   # remove centre-of-mass drift
    out: dict = {}
    T = prod.n_frames
    lag = np.arange(T) * prod.dt_fs
    lo, hi = int(0.2 * T / 2), int(0.8 * T / 2)
    for sp in sorted(set(prod.symbols)):
        m = prod.species_mask(sp)
        msd = msd_fft(r[:, m, :])[: T // 2]
        if hi - lo < 3:
            D = float("nan")
        else:
            slope = np.polyfit(lag[lo:hi], msd[lo:hi], 1)[0]       # A^2 / fs
            D = float(slope / 6.0 * 0.1)                           # -> cm^2/s
        out[sp] = {
            "D_cm2_s": D, "msd_end_A2": float(msd[-1]),
            "mobile": bool(np.isfinite(D) and D > 5e-8 and msd[-1] > min_msd_A2),
            "msd": msd[:: max(1, len(msd) // 400)], "lag_fs": lag[: T // 2][:: max(1, len(msd) // 400)],
        }
    return out


def analyse_md(phase_dir, temperature: float, force: bool | None = None) -> dict | None:
    """Thermo + diffusion summary -> ``analysis.json`` (cached)."""
    phase = PhaseDir.of(phase_dir)
    sim = phase.md_run(temperature)
    out = os.path.join(sim, "analysis.json")
    if not forced("analysis", force) and os.path.exists(out):
        return read_json(out)
    td = load_trajectory(sim)
    th = thermo_series(td)
    series = th.pop("series")
    diff = diffusion_by_species(td)
    res = {"temperature_K": temperature, "thermo": th,
           "diffusion": {k: {kk: vv for kk, vv in v.items() if kk not in ("msd", "lag_fs")}
                         for k, v in diff.items()}}
    res["series"] = {k: (v if v is None or isinstance(v, int) else np.asarray(v)[:: max(1, len(v) // 800)])
                     for k, v in series.items()}
    res["msd"] = {k: {"msd": v["msd"], "lag_fs": v["lag_fs"]} for k, v in diff.items()}
    write_json(out, res, indent=None)
    return res
