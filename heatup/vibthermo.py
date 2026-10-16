"""heatup.vibthermo
====================
Quantum-harmonic thermodynamics of a *set of modes*.

Every vibrational result in HeatUp - harmonic phonons (phonopy mesh), QHA
volumes, VDOS from MD - is reduced to a :class:`ModeSet`: frequencies
``omega`` (meV) with weights ``w`` (modes **per atom**; ``sum(w) == 3`` for a
crystal).  All free energies then follow from one function, so HA, QHA and
VDOS are comparable by construction.

Fixes relative to v0.2
----------------------
* The DOS was normalised to **1** state per atom; physically it must integrate
  to **3**, so every F_vib was 3x too small (and the 3x cancelled nowhere,
  because elements/compounds have different contributions to the hull).
* Imaginary modes were silently removed *before* normalisation; they are now
  kept in the ModeSet, reported (``imag_fraction``) and excluded from F_vib
  only at evaluation time, so the caller always knows how much of the
  spectrum is unreliable.
"""

from __future__ import annotations

import dataclasses as dc
from typing import Sequence

import numpy as np

from heatup.constants import KB_MEV, MODES_PER_ATOM, OMEGA_MIN_MEV


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
                         source: str = "HA", **meta) -> "ModeSet":
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
                 source: str = "VDOS", **meta) -> "ModeSet":
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
