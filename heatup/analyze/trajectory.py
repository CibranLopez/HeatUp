"""heatup.analyze.trajectory
============================
Everything derived from a finished MD run (``md/<T>K/output.traj`` + ``md.json``) - calculator-free, so it also
works on trajectories computed with another code.  Three analysis steps, each with its own manifest
(:data:`heatup.core.ledger.STEPS`), none depending on the others:

``mdstats``     thermo series (T, P, V, E_pot drift) and tracer diffusion per species -> ``mdstats.json``
``vdos``        vibrational DOS from the stored momenta (below)                       -> ``vdos.json``
``transition``  Gate-0 metrics: block-mean displacement / cell change / symmetry breaking -> ``transition.json``

Only *metrics* are cached.  Verdicts (thresholds) are applied by :mod:`heatup.analyze.gates` on every call, so
changing a threshold never needs a rerun.  Reading options live in ``cfg.analysis`` (``equil_ps``, ``stride``,
``block_ps``, ``vdos_max_lag_ps``, ``mobile_species``).

Trajectory loading and observables
----------------------------------
:class:`TrajData` holds the arrays every analysis needs in one pass over ``output.traj`` (the last trajectory read
is kept in memory, so the three steps read the file once).  Diffusion is computed here (FFT multi-origin MSD per
species, drift-corrected) instead of ``pymatgen-analysis-diffusion``: that package is a heavy dependency, assumes
a fixed cell (invalid in NPT) and has changed API repeatedly.

VDOS
----
    g(w)  ~  FFT[ W(t) * sum_i m_i <v_i(0) . v_i(t)> ]

* velocities come from the stored **momenta** (not from position differences),
* the VACF is **mass-weighted** (the unweighted one over-weights light atoms: <v^2> = kT/m),
* FFT autocorrelation (O(T log T)),
* a Hann lag window removes the spectral leakage that put artificial weight near w = 0 (the quantity the
  soft-mode gate measures),
* the result is a :class:`~heatup.analyze.thermo.ModeSet` with 3 modes per atom; Nyquist/aliasing and the spectral
  resolution are reported; partial (per-species) spectra are stored so mobile species can be excluded.

The VDOS of a *classical* trajectory equals the harmonic DOS in the harmonic limit at any temperature; at finite T
it contains the anharmonic frequency renormalisation (the "effective harmonic" spectrum).  F_vib built from it uses
the quantum oscillator formula (standard practice for temperature-dependent effective potentials).
"""

from __future__ import annotations

import dataclasses as dc
import logging
import os

import numpy as np
from ase.io.trajectory import Trajectory

from heatup.analyze.thermo import ModeSet
from heatup.core import ledger
from heatup.core.config import EV_A3_TO_GPA, H_MEV_FS, KB_EV, cfg, forced
from heatup.core.layout import PhaseDir, read_json, write_json

log = logging.getLogger("heatup")


# ---------------------------------------------------------------------------
# trajectory loading, MSD, thermo series
# ---------------------------------------------------------------------------

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


_TRAJ_CACHE: dict = {}


def load_trajectory(sim_dir: str) -> TrajData:
    """Read ``<sim_dir>/output.traj`` (+ ``md.json``).  Only finished runs have an ``md.json``."""
    path = os.path.join(sim_dir, "output.traj")
    mdjson = os.path.join(sim_dir, "md.json")
    meta = read_json(mdjson)
    if not meta:
        raise FileNotFoundError(f"{mdjson} missing - the MD run is not finished")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise FileNotFoundError(path)
    key = (path, os.stat(path).st_mtime_ns, os.path.getsize(path), os.stat(mdjson).st_mtime_ns,
           cfg.analysis.stride, cfg.analysis.equil_ps)
    if _TRAJ_CACHE.get("key") != key:
        _TRAJ_CACHE.clear()
        _TRAJ_CACHE.update(key=key, td=_read_trajectory(path, meta))
    return _TRAJ_CACHE["td"]


def _read_trajectory(path: str, meta: dict) -> TrajData:
    stride = max(1, cfg.analysis.stride)
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
    n_equil = int(round(cfg.analysis.equil_ps * 1000.0 / dt))
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


def diffusion_by_species(td: TrajData, min_msd_A2: float | None = None) -> dict:
    """Tracer diffusion per species from the production part (drift-corrected)."""
    min_msd_A2 = cfg.analysis.msd_mobile_A2 if min_msd_A2 is None else min_msd_A2
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


# ---------------------------------------------------------------------------
# VDOS
# ---------------------------------------------------------------------------

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


def compute_vdos(td: TrajData, max_lag_ps: float | None = None) -> dict:
    """VDOS of the production part of *td* (already equilibration-stripped or not)."""
    max_lag_ps = cfg.analysis.vdos_max_lag_ps if max_lag_ps is None else max_lag_ps
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
        partial[sp] = {"n_atoms": int(m.sum()),
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




# ---------------------------------------------------------------------------
# Gate-0 metrics
# ---------------------------------------------------------------------------

log = logging.getLogger("heatup")


def symmetry_break(f_mean: np.ndarray, f_ref: np.ndarray, cell: np.ndarray,
                   numbers: np.ndarray, max_ops: int = 200, symprec: float = 1e-3):
    """rms mismatch (A) between a structure and its images under the symmetry
    operations of the ideal reference structure ``f_ref`` (worst operation).

    Returns ``(amplitude_A, n_ops)``; (nan, 0) if spglib is unavailable.
    """
    try:
        import spglib
        sym = spglib.get_symmetry((cell, f_ref % 1.0, numbers), symprec=symprec)
        rots, trans = sym["rotations"], sym["translations"]
    except Exception:  # noqa: BLE001
        return float("nan"), 0
    n_ops = len(rots)
    idx = np.arange(n_ops)
    if n_ops > max_ops:
        idx = np.unique(np.linspace(0, n_ops - 1, max_ops).astype(int))
    same = numbers[:, None] == numbers[None, :]
    worst = 0.0
    for k in idx:
        fp = f_mean @ rots[k].T + trans[k]
        d = fp[:, None, :] - f_mean[None, :, :]
        d -= np.round(d)
        dist = np.linalg.norm(d @ cell, axis=-1)
        dist = np.where(same, dist, np.inf)
        worst = max(worst, float(np.sqrt((dist.min(axis=1) ** 2).mean())))
    return worst, int(n_ops)


def transition_metrics(td: TrajData, mobile: list[str] | None = None) -> dict:
    """Block-mean displacement / cell / symmetry metrics for one trajectory."""
    a = cfg.analysis
    mobile = list(mobile if mobile is not None else a.mobile_species)
    p = td.production()
    nb = int(round(a.block_ps * 1000.0 / p.dt_fs))
    nb = max(1, min(nb, p.n_frames // 2))
    short = p.n_frames < 2 * int(round(a.block_ps * 1000.0 / p.dt_fs))

    u = unwrap_frac(p.frac)                                         # (T,N,3)
    f0, f1 = u[:nb].mean(0), u[-nb:].mean(0)
    h0, h1 = p.cell[:nb].mean(0), p.cell[-nb:].mean(0)
    hm = p.cell.mean(0)
    d_frac = f1 - f0
    fixed = ~p.species_mask(mobile) if mobile else np.ones(p.n_atoms, bool)
    if fixed.sum() == 0:
        fixed[:] = True
    # remove rigid drift of the framework (mass-weighted mean over non-mobile atoms)
    wf = (p.masses[fixed] / p.masses[fixed].sum())[:, None]
    d_frac = d_frac - (d_frac[fixed] * wf).sum(0, keepdims=True)
    d_cart = d_frac @ hm
    dist = np.linalg.norm(d_cart, axis=1)

    per_species = {sp: {"rms_A": float(np.sqrt((dist[p.species_mask(sp)] ** 2).mean())),
                        "max_A": float(dist[p.species_mask(sp)].max())}
                   for sp in sorted(set(p.symbols))}
    rms = float(np.sqrt((dist[fixed] ** 2).mean()))
    mx = float(dist[fixed].max())

    # cell change: x1 = x0 @ M, M = h0^-1 h1  ->  eps = (M M^T - I)/2
    M = np.linalg.solve(h0, h1)
    eps = 0.5 * (M @ M.T - np.eye(3))
    dev = eps - np.trace(eps) / 3.0 * np.eye(3)
    shape_strain = float(np.abs(np.linalg.eigvalsh(dev)).max())
    dV = float(np.linalg.det(h1) / np.linalg.det(h0) - 1.0)

    from ase.data import atomic_numbers
    numbers = np.array([atomic_numbers[x] for x in p.symbols])
    ideal = td.frac[0]                       # frame 0 of the run = relaxed, symmetric supercell
    sb0, n_ops = symmetry_break(f0, ideal, hm, numbers)
    sb1, _ = symmetry_break(f1, ideal, hm, numbers)

    # DACF relative to the first-block mean (diagnostic)
    uc = (u - f0[None]) @ hm
    uc = uc[:, fixed, :].reshape(p.n_frames, -1)
    max_lag = p.n_frames // 2
    dacf = autocorr_fft(uc, max_lag) / uc.shape[1]
    sub = max(1, max_lag // 600)

    return {
        "rms_A": rms, "max_A": mx, "per_species": per_species,
        "cell_shape_strain": shape_strain, "cell_volume_change": dV,
        "symmetry_break_first_A": sb0, "symmetry_break_last_A": sb1, "n_symmetry_ops": n_ops,
        "dacf_A2": dacf[::sub], "dacf_lag_fs": (np.arange(max_lag) * p.dt_fs)[::sub],
        "dacf_plateau_A2": float(dacf[int(0.8 * max_lag):].mean()),
        "block_frames": nb, "production_ps": p.duration_ps, "too_short": bool(short),
        "mobile_species": mobile,
    }


# ---------------------------------------------------------------------------
# the three analysis steps (cached per MD temperature, one manifest each)
# ---------------------------------------------------------------------------

def _cached(phase: PhaseDir, step: str, T: float, force: bool | None, compute, load):
    """Return the result of *step* at *T*: loaded when its manifest is fresh, else computed (run marker + manifest)."""
    spec = ledger.STEPS[step]
    out = os.path.join(spec.where(phase, T), spec.outputs[0])
    if not forced(step, force) and os.path.exists(out) and ledger.is_fresh(phase, step, T, calc=False):
        return load()
    with ledger.tracking(phase, step, T):
        res = compute()
        ledger.record(phase, step, T)
    return res


def run_mdstats(phase_dir, temperature: float, force: bool | None = None) -> dict | None:
    """Thermo + diffusion summary -> ``md/<T>K/mdstats.json`` (not ``analysis.json``: other software keeps its own there)."""
    phase = PhaseDir.of(phase_dir)
    sim = phase.md_run(temperature)
    out = os.path.join(sim, "mdstats.json")

    def compute():
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

    return _cached(phase, "mdstats", temperature, force, compute, lambda: read_json(out))


def run_vdos(phase_dir, temperature: float, force: bool | None = None) -> dict | None:
    """Compute (or load) ``md/<T>K/vdos.json``; the result holds ``modes`` as a ModeSet."""
    phase = PhaseDir.of(phase_dir)
    out = os.path.join(phase.md_run(temperature), "vdos.json")

    def compute():
        res = compute_vdos(load_trajectory(phase.md_run(temperature)))
        d = dict(res)
        d["modes"] = res["modes"].to_dict(max_points=3000)
        write_json(out, d, indent=None)
        if res["alias_weight_top10pct"] > 1e-3:
            log.warning("%s %dK: %.2f%% of VDOS weight near the Nyquist limit (%.0f meV) - decrease md.nblock",
                        phase.tag, temperature, 100 * res["alias_weight_top10pct"], res["nyquist_meV"])
        return res

    return _cached(phase, "vdos", temperature, force, compute, lambda: load_vdos(phase, temperature))


def load_vdos(phase_dir, temperature: float) -> dict | None:
    sim = PhaseDir.of(phase_dir).md_run(temperature)
    d = read_json(os.path.join(sim, "vdos.json"))
    if not d:
        return None
    d["modes"] = ModeSet.from_dict(d["modes"])
    return d


def run_transition(phase_dir, temperature: float, force: bool | None = None) -> dict | None:
    """Gate-0 metrics -> ``md/<T>K/transition.json`` (no verdict inside: see ``gates.transition_verdict``)."""
    phase = PhaseDir.of(phase_dir)
    out = os.path.join(phase.md_run(temperature), "transition.json")

    def compute():
        m = transition_metrics(load_trajectory(phase.md_run(temperature)))
        m["temperature_K"] = temperature
        write_json(out, m, indent=None)
        return m

    return _cached(phase, "transition", temperature, force, compute, lambda: read_json(out))
