"""heatup.analyze.thermo
========================
Free energies per atom - everything between a set of vibrational modes and ``G(T)``.

1. :class:`ModeSet` - every vibrational result (harmonic phonon mesh, quasi-harmonic volumes, anharmonic = VDOS from MD) reduced to
   frequencies ``omega`` (meV, negative = imaginary) with weights ``w`` (modes **per atom**, ``sum(w) == 3``
   for a crystal).  All F_vib then follow from one function, so the vibrational levels are comparable by construction.
2. loaders of the cached results (``load_harmonic``, ``load_quasiharmonic``, ``load_perturbative``) and the summary of the
   perturbative (fc3) result (``perturbative_summary``: a diagnostic, never a gate).
3. ``G(T) = H0 + sum of the selected terms``  [eV/atom] as independent contributions (:class:`GibbsAssembler`).  ``H0`` is
   the relaxed enthalpy (== energy at P = 0) from ``relaxation/energy.json``.  Which terms and models enter is declared
   ONCE, in :mod:`heatup.core.contributions` (``cfg.terms``); :func:`free_energy` assembles from that selection.  Every
   contribution is a function ``fn(phase, T) -> array`` that returns **zeros when its optional data are absent**, and
   raises :class:`MissingData` when the data it *needs* are not there.

Vibrational models (``cfg.terms.vibrational``)
----------------------------------------------
``harmonic``       ``harmonic/harmonic.json`` (3 modes/atom, imaginary modes skipped and flagged)
``ZPE``            only the zero-point energy of the harmonic modes
``quasiharmonic``  ``quasiharmonic/quasiharmonic.json``: G(T) at V_eq(T) (E(V) + F_vib(V,T) + PV), replaces H0 + F_vib
``anharmonic``     MD spectra at the available MD temperatures, interpolated in T (temperature-renormalised
                   effective-harmonic F_vib); experimental

Corrections of v0.2: DOS integrates to 3 modes per atom (was 1: every F_vib 3x too small); imaginary modes are
kept in the ModeSet and excluded from F_vib only at evaluation time, so the amount of unreliable spectrum is always
known; F_el conserves the electron number *relative to the DOS window* and subtracts the T = 0 reference; F_mag is
the standard Dinsdale (1991) Inden-Hillert-Jarl expression, only the *thermal* part is added; finite pressure is
handled by relaxing under pressure (enthalpy), not by a separate EOS term; optional per-element energy shifts
(``hull.reference_energy_shift``).
"""

from __future__ import annotations

import dataclasses as dc
import logging
import os
from typing import Callable, Sequence

import numpy as np

from heatup.core import ledger
from heatup.core.config import KB_EV, KB_MEV, MODES_PER_ATOM, OMEGA_MIN_MEV, cfg
from heatup.core.layout import PhaseDir, composition, read_json, read_structure

log = logging.getLogger("heatup")
Contribution = Callable[[PhaseDir, np.ndarray], np.ndarray]


class MissingData(Exception):
    """Data a free-energy contribution needs are absent or do not cover the requested range."""


# ---------------------------------------------------------------------------
# ModeSet and the quantum-harmonic kernels
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Per-mode kernels (omega, T -> meV or meV/K); omega > 0 required
# ---------------------------------------------------------------------------

def _x(omega: np.ndarray, T: float) -> np.ndarray:
    return np.clip(omega / (KB_MEV * T), 1e-10, 700.0)


def helmholtz_kernel(omega: np.ndarray, T: float) -> np.ndarray:
    """1/2 w + kT ln(1 - e^{-w/kT})   [meV]"""
    if T <= 0.0:
        return 0.5 * omega
    x = _x(omega, T)
    return 0.5 * omega + KB_MEV * T * np.log1p(-np.exp(-x))


def energy_kernel(omega: np.ndarray, T: float) -> np.ndarray:
    """w (1/2 + n)   [meV]"""
    if T <= 0.0:
        return 0.5 * omega
    x = _x(omega, T)
    return omega * (0.5 + 1.0 / np.expm1(x))


def entropy_kernel(omega: np.ndarray, T: float) -> np.ndarray:
    """kB [x/(e^x-1) - ln(1-e^-x)]   [meV/K]"""
    if T <= 0.0:
        return np.zeros_like(omega)
    x = _x(omega, T)
    return KB_MEV * (x / np.expm1(x) - np.log1p(-np.exp(-x)))


def cv_kernel(omega: np.ndarray, T: float) -> np.ndarray:
    """kB x^2 e^x/(e^x-1)^2   [meV/K]"""
    if T <= 0.0:
        return np.zeros_like(omega)
    x = _x(omega, T)
    return KB_MEV * x**2 * np.exp(x) / np.expm1(x) ** 2


# ---------------------------------------------------------------------------
# ModeSet
# ---------------------------------------------------------------------------

@dc.dataclass
class ModeSet:
    """Frequencies (meV, negative = imaginary) with weights (modes per atom)."""

    omega: np.ndarray
    weight: np.ndarray
    source: str = "?"
    meta: dict = dc.field(default_factory=dict)

    def __post_init__(self) -> None:
        self.omega = np.asarray(self.omega, dtype=float).ravel()
        self.weight = np.asarray(self.weight, dtype=float).ravel()
        if self.omega.shape != self.weight.shape:
            raise ValueError("omega and weight must have the same shape")

    # -- constructors -------------------------------------------------------
    @classmethod
    def from_frequencies(cls, omega_meV, q_weights=None, n_atoms: int | None = None,
                         source: str = "harmonic", **meta) -> "ModeSet":
        """From a (n_q, n_branches) array of frequencies and q-point weights."""
        om = np.asarray(omega_meV, dtype=float)
        if om.ndim == 1:
            om = om[None, :]
        nq, nb = om.shape
        qw = np.ones(nq) if q_weights is None else np.asarray(q_weights, float)
        qw = qw / qw.sum()
        w = np.repeat(qw[:, None], nb, axis=1)
        n_at = n_atoms if n_atoms is not None else nb / 3.0
        w = w / n_at              # modes per atom: sum = nb / n_atoms (= 3 for a crystal)
        return cls(om.ravel(), w.ravel(), source=source, meta=meta)

    @classmethod
    def from_dos(cls, omega_meV, g, n_modes_per_atom: float = MODES_PER_ATOM,
                 source: str = "anharmonic", **meta) -> "ModeSet":
        """From a DOS g(omega) on a grid; rescaled so that sum(w) = n_modes_per_atom."""
        om = np.asarray(omega_meV, dtype=float)
        g = np.clip(np.asarray(g, dtype=float), 0.0, None)
        d = np.gradient(om)
        w = g * d
        tot = w.sum()
        if not np.isfinite(tot) or tot <= 0:
            raise ValueError("DOS has no positive weight")
        return cls(om, w * (n_modes_per_atom / tot), source=source, meta=meta)

    # -- properties ----------------------------------------------------------
    @property
    def total_weight(self) -> float:
        return float(self.weight.sum())

    def imag_fraction(self, tol_meV: float = 0.5) -> float:
        """Fraction of the mode weight with omega < -tol (genuinely imaginary)."""
        return float(self.weight[self.omega < -abs(tol_meV)].sum() / self.weight.sum())

    def max_imag_meV(self) -> float:
        neg = self.omega[self.omega < 0]
        return float(-neg.min()) if neg.size else 0.0

    def physical(self, omega_min: float = OMEGA_MIN_MEV):
        m = self.omega > omega_min
        return self.omega[m], self.weight[m]

    def mean_frequency_meV(self) -> float:
        o, w = self.physical()
        return float((o * w).sum() / w.sum())

    def low_weight_fraction(self, window_meV: float) -> float:
        """Fraction of *all* modes with |omega| < window."""
        m = np.abs(self.omega) < window_meV
        return float(self.weight[m].sum() / self.weight.sum())

    # -- thermodynamics (per atom, eV) ---------------------------------------
    def _integrate(self, kernel, T: float) -> float:
        o, w = self.physical()
        return float((w * kernel(o, T)).sum())

    def f_vib(self, T) -> np.ndarray | float:
        """F_vib(T) in eV/atom (imaginary / zero modes skipped)."""
        return self._scalar_or_array(T, helmholtz_kernel, 1e-3)

    def e_vib(self, T):
        return self._scalar_or_array(T, energy_kernel, 1e-3)

    def s_vib(self, T):
        """eV/K/atom"""
        return self._scalar_or_array(T, entropy_kernel, 1e-3)

    def cv(self, T):
        """eV/K/atom"""
        return self._scalar_or_array(T, cv_kernel, 1e-3)

    def zpe(self) -> float:
        return float(self.f_vib(0.0))

    def _scalar_or_array(self, T, kernel, scale: float):
        arr = np.atleast_1d(np.asarray(T, dtype=float))
        out = np.array([self._integrate(kernel, float(t)) * scale for t in arr])
        return float(out[0]) if np.ndim(T) == 0 else out

    # -- (de)serialisation -----------------------------------------------------
    def to_dict(self, max_points: int | None = 4000) -> dict:
        om, w = self.omega, self.weight
        if max_points is not None and om.size > max_points:
            # re-bin to a histogram (keeps imaginary part)
            edges = np.linspace(om.min(), om.max() + 1e-9, max_points + 1)
            idx = np.digitize(om, edges) - 1
            wb = np.bincount(idx, weights=w, minlength=max_points)[:max_points]
            ob = np.bincount(idx, weights=w * om, minlength=max_points)[:max_points]
            keep = wb > 0
            om, w = ob[keep] / wb[keep], wb[keep]
        return {"omega_meV": om.tolist(), "weight": w.tolist(),
                "source": self.source, "meta": self.meta,
                "modes_per_atom": self.total_weight}

    @classmethod
    def from_dict(cls, d: dict) -> "ModeSet":
        return cls(np.array(d["omega_meV"]), np.array(d["weight"]),
                   source=d.get("source", "?"), meta=d.get("meta", {}))

    def smooth_dos(self, sigma_meV: float = 0.5, n: int = 600):
        """Gaussian-broadened DOS (states per atom per meV) for plotting."""
        lo, hi = float(self.omega.min()) - 3 * sigma_meV, float(self.omega.max()) + 3 * sigma_meV
        grid = np.linspace(lo, hi, n)
        dos = np.zeros_like(grid)
        for o, w in zip(self.omega, self.weight):
            dos += w * np.exp(-0.5 * ((grid - o) / sigma_meV) ** 2)
        dos /= sigma_meV * np.sqrt(2 * np.pi)
        return grid, dos


def f_vib_curve(modes: ModeSet, temperatures: Sequence[float]) -> np.ndarray:
    return np.asarray(modes.f_vib(np.asarray(temperatures, float)), dtype=float)


def interpolate_f_vib(
        modesets: Sequence[tuple[float, ModeSet]],
        temperatures: Sequence[float],
) -> np.ndarray:
    """Temperature-renormalised effective-harmonic F_vib(T).

    ``modesets`` is a list of ``(T_md, ModeSet)`` measured at those MD
    temperatures.  At each hull temperature T we use the spectrum measured at
    the nearest MD temperatures (linear interpolation *of F_vib(T; g_i)*
    between the two neighbours, constant extrapolation outside).  v0.2 averaged
    all spectra and used that one average for every T, which is neither the
    spectrum at T nor a controlled approximation.
    """
    ms = sorted(modesets, key=lambda p: p[0])
    Ts = np.asarray(temperatures, float)
    if len(ms) == 1:
        return f_vib_curve(ms[0][1], Ts)
    t_md = np.array([m[0] for m in ms])
    curves = np.array([f_vib_curve(m[1], Ts) for m in ms])   # (n_md, n_T)
    out = np.empty_like(Ts)
    for j, T in enumerate(Ts):
        out[j] = np.interp(T, t_md, curves[:, j])
    return out


def spread_warning(curves: np.ndarray, threshold: float) -> tuple[bool, float]:
    """Relative spread of F_vib between different MD spectra (consistency check)."""
    if curves.shape[0] < 2:
        return False, 0.0
    mean = curves.mean(axis=0)
    ok = np.abs(mean) > 1e-6
    if not ok.any():
        return False, 0.0
    rel = (curves.max(axis=0) - curves.min(axis=0))[ok] / np.abs(mean[ok])
    worst = float(rel.max())
    return worst > threshold, worst

# ---------------------------------------------------------------------------
# cached phonon results
# ---------------------------------------------------------------------------

def load_harmonic(phase_dir) -> dict | None:
    """Cached harmonic result with ``modes`` converted to a ModeSet."""
    phase = PhaseDir.of(phase_dir)
    d = read_json(os.path.join(phase.harmonic_dir, "harmonic.json"))
    if not d:
        return None
    d["modes"] = ModeSet.from_dict(d["modes"])
    return d


def load_quasiharmonic(phase_dir) -> dict | None:
    return read_json(os.path.join(PhaseDir.of(phase_dir).quasiharmonic_dir, "quasiharmonic.json"))


def load_perturbative(phase_dir) -> dict | None:
    return read_json(os.path.join(PhaseDir.of(phase_dir).perturbative_dir, "perturbative.json"))


load_ha, load_qha, load_fc3 = load_harmonic, load_quasiharmonic, load_perturbative       # names of earlier versions


def perturbative_summary(res: dict | None, T: float, ratio_warn: float | None = None) -> dict | None:
    """Values of a cached perturbative (fc3) result at the temperature closest to *T*, plus a ``flag`` when the linewidths
    say the harmonic picture is strained there.  Pure function (unit-tested)."""
    if not res or not res.get("temperatures_K"):
        return None
    ratio_warn = cfg.perturbative.ratio_warn if ratio_warn is None else ratio_warn
    temps = np.asarray(res["temperatures_K"], float)
    i = int(np.argmin(np.abs(temps - T)))
    ratio = res["gamma_over_nu_mean"][i]
    out = {"T_K": float(temps[i]), "kappa_W_mK": res["kappa_W_mK"][i], "gamma_over_nu_mean": ratio,
           "frac_ioffe_regel": res["frac_ioffe_regel"][i], "tau_median_ps": res["tau_median_ps"][i],
           "ratio_warn": ratio_warn, "flag": None}
    if ratio is not None and ratio > ratio_warn:
        out["flag"] = (f"[note] perturbative: <gamma/nu> = {ratio:.3f} > {ratio_warn} at {temps[i]:g} K "
                       f"({100 * (out['frac_ioffe_regel'] or 0):.1f}% of the weight beyond the Ioffe-Regel limit): "
                       "harmonic F_vib is less reliable here, compare with the anharmonic (MD) overlay")
    return out


anharmonic_summary = perturbative_summary                                          # name of earlier versions


# ---------------------------------------------------------------------------
# G(T) of one phase
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# reference energy
# ---------------------------------------------------------------------------

def reference_energy(phase: PhaseDir) -> float | None:
    """H0 per atom (eV), including optional per-element shifts."""
    e = phase.energy()
    if not e or not e.get("converged", False):
        return None
    h0 = float(e.get("enthalpy_eV_per_atom", e["energy_eV_per_atom"]))
    shifts = cfg.hull.reference_energy_shift
    if shifts:
        comp = composition(read_structure(phase.contcar))
        h0 += sum(x * shifts.get(el, 0.0) for el, x in comp.items())
    return h0


# ---------------------------------------------------------------------------
# vibrational contributions
# ---------------------------------------------------------------------------

def f_vib_zpe(phase: PhaseDir, T: np.ndarray) -> np.ndarray:
    """Zero-point energy of the harmonic modes (eV/atom), the same at every temperature."""
    ha = load_harmonic(phase)
    if not ha:
        raise MissingData(f"{phase.tag}: harmonic/harmonic.json missing")
    return np.full(np.shape(T), float(ha["modes"].zpe()))


def f_vib_harmonic(phase: PhaseDir, T: np.ndarray) -> np.ndarray:
    ha = load_harmonic(phase)
    if not ha:
        raise MissingData(f"{phase.tag}: harmonic/harmonic.json missing")
    return np.asarray(ha["modes"].f_vib(T), float)


def f_vib_anharmonic(phase: PhaseDir, T: np.ndarray) -> np.ndarray:
    """Effective-harmonic F_vib from the MD spectra (phonons from MD), interpolated between the MD temperatures."""
    from heatup.analyze.trajectory import load_vdos
    sets = []
    for t in ledger.md_temperatures(phase):
        v = load_vdos(phase, t)
        if v:
            sets.append((float(t), v["modes"]))
    if not sets:
        raise MissingData(f"{phase.tag}: no MD spectra (run the analysis of the MD runs first)")
    return interpolate_f_vib(sets, T)


def anharmonic_consistency(phase: PhaseDir, T: np.ndarray) -> tuple[bool, float]:
    from heatup.analyze.trajectory import load_vdos
    cur = []
    for t in ledger.md_temperatures(phase):
        v = load_vdos(phase, t)
        if v:
            cur.append(np.asarray(v["modes"].f_vib(T)))
    if len(cur) < 2:
        return False, 0.0
    return spread_warning(np.array(cur), cfg.gates.fe_consistency)


def f_vib_quasiharmonic(phase: PhaseDir, T: np.ndarray) -> np.ndarray:
    """G_quasiharmonic(T) - H0 (so that H0 + contribution = G_quasiharmonic)."""
    q = load_quasiharmonic(phase)
    if not q:
        raise MissingData(f"{phase.tag}: quasiharmonic/quasiharmonic.json missing")
    qt = np.asarray(q["temperatures"], float)
    if T.min() < qt.min() - 1e-9 or T.max() > qt.max() + 1e-9:          # np.interp would clamp silently
        raise MissingData(f"{phase.tag}: quasi-harmonic computed for {qt.min():g}-{qt.max():g} K but the hull needs "
                          f"{T.min():g}-{T.max():g} K (cfg.quasiharmonic.temperatures, notebook 1)")
    g = np.interp(T, qt, q["G_eV_per_atom"])
    # G = E(V) + F_vib + PV is already an enthalpy-type quantity: H0 + contribution = G.  H0 carries the
    # optional per-element shift, so the contribution is taken relative to the *unshifted* relaxed enthalpy
    # (v0.3 subtracted E and the shift, which double-counted PV at P != 0 and dropped the shift for these phases).
    return g - float((phase.energy() or {})["enthalpy_eV_per_atom"])


# ---------------------------------------------------------------------------
# electronic
# ---------------------------------------------------------------------------

def _fermi(eps: np.ndarray, mu: float, T: float) -> np.ndarray:
    if T <= 0:
        return (eps <= mu).astype(float)
    return 1.0 / (1.0 + np.exp(np.clip((eps - mu) / (KB_EV * T), -500, 500)))


def f_el(phase: PhaseDir, T: np.ndarray) -> np.ndarray:
    """F_el(T) - F_el(0) from ``electronic/edos.json`` (energies relative to E_F, states/eV/atom)."""
    d = read_json(phase.join("electronic", "edos.json"))
    if not d:
        return np.zeros_like(T)
    eps = np.array(d["energies_eV"], float)
    g = np.array(d["dos_per_eV"], float)
    n0 = float(np.trapezoid(g * _fermi(eps, 0.0, 0.0), eps))     # electrons in the window at T=0
    u0 = float(np.trapezoid(eps * g * _fermi(eps, 0.0, 0.0), eps))
    out = np.zeros_like(T)
    for i, t in enumerate(T):
        if t <= 0:
            continue
        lo, hi = eps.min(), eps.max()
        for _ in range(80):                       # bisection for mu(T) conserving n0
            mid = 0.5 * (lo + hi)
            if np.trapezoid(g * _fermi(eps, mid, t), eps) < n0:
                lo = mid
            else:
                hi = mid
        mu = 0.5 * (lo + hi)
        f = np.clip(_fermi(eps, mu, t), 1e-300, 1 - 1e-16)
        s = -KB_EV * np.trapezoid(g * (f * np.log(f) + (1 - f) * np.log(1 - f)), eps)
        u = np.trapezoid(eps * g * f, eps) - u0
        out[i] = u - t * s
    return out


# ---------------------------------------------------------------------------
# magnetic (Hillert-Jarl-Inden, Dinsdale 1991)
# ---------------------------------------------------------------------------

def _hjg_A(p: float) -> float:
    return 518.0 / 1125.0 + 11692.0 / 15975.0 * (1.0 / p - 1.0)


def _hjg_f(tau: float, p: float) -> float:
    A = _hjg_A(p)
    if tau < 1:
        return 1.0 - (79.0 / (140.0 * p * tau)
                      + 474.0 / 497.0 * (1.0 / p - 1.0) * (tau ** 3 / 6 + tau ** 9 / 135 + tau ** 15 / 600)) / A
    return -(tau ** -5 / 10 + tau ** -15 / 315 + tau ** -25 / 1500) / A


def f_mag(phase: PhaseDir, T: np.ndarray) -> np.ndarray:
    """Thermal magnetic free energy from ``magnetic/moments.json``::

        {"curie_temperature_K": 1043, "beta_muB": 2.2, "structure_factor_p": 0.4}
    (p = 0.40 bcc, 0.28 fcc/hcp/other - Hillert & Jarl 1978, Dinsdale 1991).  Returns the thermal part with S(0) = 0:
    G_mag(T) - G_mag(0) - kB T ln(beta+1)   (see ``heatup.analyze.terms``: CALPHAD books the spin-disorder entropy
    in the paramagnetic lattice part, which phonon free energies of the ordered state do not contain)."""
    d = read_json(phase.join("magnetic", "moments.json"))
    if not d:
        return np.zeros_like(T)
    from heatup.analyze.terms import magnetic_free_energy
    return magnetic_free_energy(T, float(d["curie_temperature_K"]), float(d["beta_muB"]),
                                float(d.get("structure_factor_p", 0.4)), third_law=True)


# ---------------------------------------------------------------------------
# configurational (ideal mixing on partially occupied sites)
# ---------------------------------------------------------------------------

def f_conf(phase: PhaseDir, T: np.ndarray) -> np.ndarray:
    """-T S_conf per atom from ``disorder/site_occupancies.json``::

        {"sites": [{"species": {"Li": 0.5, "Vac": 0.5}}, ...], "sro_factor": 1.0}
    Every site counts as one "atom position" (vacancies included in the average).
    """
    d = read_json(phase.join("disorder", "site_occupancies.json"))
    if not d or not d.get("sites"):
        return np.zeros_like(T)
    s = 0.0
    for site in d["sites"]:
        for x in site.get("species", {}).values():
            if 0.0 < x < 1.0:
                s -= KB_EV * x * np.log(x)
    s = s / len(d["sites"]) * float(d.get("sro_factor", 1.0))
    return -T * s


# ---------------------------------------------------------------------------
# assembler
# ---------------------------------------------------------------------------

class GibbsAssembler:
    """``G = H0 + sum_i contribution_i``; add your own with :meth:`register`."""

    def __init__(self) -> None:
        self._contrib: list[tuple[str, Contribution]] = []

    def register(self, name: str, fn: Contribution) -> "GibbsAssembler":
        self._contrib.append((name, fn))
        return self

    def compute(self, phase_dir, temperatures) -> dict:
        phase = PhaseDir.of(phase_dir)
        T = np.asarray(temperatures, float)
        h0 = reference_energy(phase)
        if h0 is None:
            return {"available": False, "message": "no converged relaxation"}
        G = np.full_like(T, h0)
        contrib: dict[str, np.ndarray] = {}
        problems: list[str] = []
        for name, fn in self._contrib:
            try:
                v = np.asarray(fn(phase, T), float)
            except (FileNotFoundError, MissingData) as exc:
                problems.append(f"{name}: {exc}")
                return {"available": False, "message": "; ".join(problems), "problems": problems}
            contrib[name] = v
            G = G + v
        return {"available": True, "temperatures": T, "H0_eV_per_atom": h0,
                "G_eV_per_atom": G, "contributions": contrib, "problems": problems}


def free_energy(phase_dir, temperatures, model: str | None = None) -> dict:
    """G(T) of one phase assembled from the selected terms (``cfg.terms`` / as prepared; ``heatup.core.contributions``),
    with quality flags.  ``model`` overrides the vibrational model for this call (e.g. the anharmonic overlay)."""
    from heatup.core import contributions
    phase = PhaseDir.of(phase_dir)
    sel = contributions.resolve(phase, vibrational=model)
    res = contributions.assemble(sel, phase).compute(phase, temperatures)
    vib = sel["vibrational"]
    res["method"] = vib
    flags: list[str] = []
    if res.get("available"):
        if vib in ("harmonic", "quasiharmonic", "ZPE"):
            ha = load_harmonic(phase)
            if ha and ha["imag_fraction"] >= cfg.gates.imag_frac_fail:
                flags.append(f"harmonic imaginary weight {100 * ha['imag_fraction']:.1f}% - F_vib unreliable")
        if vib == "quasiharmonic":
            q = load_quasiharmonic(phase) or {}
            T_arr = np.asarray(q.get("temperatures", []), float)
            bad = T_arr[~np.asarray(q.get("valid", []), bool)] if len(T_arr) else []
            if len(bad):
                flags.append(f"quasi-harmonic equilibrium volume leaves the sampled range above {bad.min():g} K")
        if vib == "anharmonic":
            warn, spread = anharmonic_consistency(phase, np.asarray(temperatures, float))
            if warn:
                flags.append(f"F_vib differs by {100 * spread:.1f}% between MD temperatures")
    res["flags"] = flags
    return res
