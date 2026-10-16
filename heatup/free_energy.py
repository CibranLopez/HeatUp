"""heatup.free_energy
======================
Gibbs/Helmholtz free energy per atom as a sum of independent contributions::

    G(T) = H0 + F_vib(T) + F_el(T) + F_mag(T) + F_conf(T)       [eV/atom]

``H0`` is the relaxed enthalpy (== energy at P = 0) from ``relaxation/energy.json``.
Every contribution is a function ``fn(phase, T) -> array`` that returns **zeros
when its data are absent**, so the model degrades gracefully.

Vibrational methods (``method=``)
---------------------------------
``HA``   harmonic, phonons/ha.json (3 modes/atom, imaginary modes skipped & flagged)
``QHA``  phonons/qha.json: G(T) at V_eq(T) (E(V) + F_vib(V,T) + PV), replaces H0 + F_vib
``VDOS`` MD spectra at the available MD temperatures, interpolated in T
         (temperature-renormalised effective-harmonic F_vib)

Corrected vs v0.2
-----------------
* DOS normalisation (3 modes per atom, not 1);
* F_el: electron number conserved *relative to the DOS window* (v0.2 solved for
  the nominal electron count, which is wrong whenever the tabulated window
  does not contain all valence states) and the T = 0 reference is subtracted;
* F_mag: the Inden-Hillert-Jarl function was wrong (missing the structure-factor
  terms); now the standard Dinsdale (1991) expression using T_C, beta, p, and only
  the *thermal* part G_mag(T) - G_mag(0) is added (the ordered ground state is
  already in the spin-polarised DFT/MACE energy);
* PV: no separate "EOS" term - finite pressure is handled by relaxing under
  pressure (enthalpy), which is the only consistent way;
* optional per-element energy shifts (``hull.reference_energy_shift``).
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from heatup.config import cfg
from heatup.constants import KB_EV
from heatup.paths import PhaseDir, read_json
from heatup.structure import composition, read_structure
from heatup.vibthermo import interpolate_f_vib, spread_warning

log = logging.getLogger("heatup")
Contribution = Callable[[PhaseDir, np.ndarray], np.ndarray]


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

def f_vib_ha(phase: PhaseDir, T: np.ndarray) -> np.ndarray:
    from heatup.phonons import load_ha
    ha = load_ha(phase)
    if not ha:
        raise FileNotFoundError(f"{phase.tag}: phonons/ha.json missing")
    return np.asarray(ha["modes"].f_vib(T), float)


def f_vib_vdos(phase: PhaseDir, T: np.ndarray) -> np.ndarray:
    from heatup.vdos import load_vdos
    sets = []
    for t in phase.md_temperatures():
        v = load_vdos(phase, t)
        if v:
            sets.append((float(t), v["modes"]))
    if not sets:
        raise FileNotFoundError(f"{phase.tag}: no MD spectra")
    return interpolate_f_vib(sets, T)


def vdos_consistency(phase: PhaseDir, T: np.ndarray) -> tuple[bool, float]:
    from heatup.vdos import load_vdos
    cur = []
    for t in phase.md_temperatures():
        v = load_vdos(phase, t)
        if v:
            cur.append(np.asarray(v["modes"].f_vib(T)))
    if len(cur) < 2:
        return False, 0.0
    return spread_warning(np.array(cur), cfg.gates.fe_consistency)


def f_vib_qha(phase: PhaseDir, T: np.ndarray) -> np.ndarray:
    """G_QHA(T) - H0 (so that H0 + contribution = G_QHA)."""
    from heatup.phonons import load_qha
    q = load_qha(phase)
    if not q:
        raise FileNotFoundError(f"{phase.tag}: phonons/qha.json missing")
    h0 = reference_energy(phase)
    g = np.interp(T, q["temperatures"], q["G_eV_per_atom"])
    e = phase.energy()
    # qha G is built from the (fixed-cell) E(V) curve - express relative to H0
    return g - float(e["energy_eV_per_atom"]) - (h0 - float(e["enthalpy_eV_per_atom"]))


VIB_METHODS = {"HA": f_vib_ha, "QHA": f_vib_qha, "VDOS": f_vib_vdos}


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
    (p = 0.28 bcc, 0.40 otherwise).  Returns G_mag(T) - G_mag(0) per atom.
    """
    d = read_json(phase.join("magnetic", "moments.json"))
    if not d:
        return np.zeros_like(T)
    Tc, beta, p = float(d["curie_temperature_K"]), float(d["beta_muB"]), float(d.get("structure_factor_p", 0.4))
    if Tc <= 0 or beta <= 0:
        return np.zeros_like(T)
    lnb = np.log(beta + 1.0)
    g0 = -KB_EV * lnb * 79.0 * Tc / (140.0 * p * _hjg_A(p))     # lim T->0 of kB T ln(b+1) f(T/Tc)
    out = np.zeros_like(T)
    for i, t in enumerate(T):
        out[i] = 0.0 if t <= 0 else KB_EV * t * lnb * _hjg_f(t / Tc, p) - g0
    return out


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
            except FileNotFoundError as exc:
                problems.append(f"{name}: {exc}")
                return {"available": False, "message": "; ".join(problems), "problems": problems}
            contrib[name] = v
            G = G + v
        return {"available": True, "temperatures": T, "H0_eV_per_atom": h0,
                "G_eV_per_atom": G, "contributions": contrib, "problems": problems}


def build_assembler(method: str | None = None) -> GibbsAssembler:
    method = (method or cfg.hull.phonon_method).upper()
    if method not in VIB_METHODS:
        raise ValueError(f"phonon method must be one of {sorted(VIB_METHODS)}, got {method!r}")
    asm = GibbsAssembler().register(f"F_vib[{method}]", VIB_METHODS[method])
    if cfg.hull.include_electronic:
        asm.register("F_el", f_el)
    if cfg.hull.include_magnetic:
        asm.register("F_mag", f_mag)
    if cfg.hull.include_configurational:
        asm.register("F_conf", f_conf)
    return asm


def free_energy(phase_dir, temperatures, method: str | None = None) -> dict:
    """Convenience: assemble G(T) with the given vibrational method and attach quality flags."""
    phase = PhaseDir.of(phase_dir)
    method = (method or cfg.hull.phonon_method).upper()
    res = build_assembler(method).compute(phase, temperatures)
    res["method"] = method
    flags: list[str] = []
    if res.get("available"):
        if method in ("HA", "QHA"):
            from heatup.phonons import load_ha
            ha = load_ha(phase)
            if ha and ha["imag_fraction"] >= cfg.gates.imag_frac_fail:
                flags.append(f"HA imaginary weight {100 * ha['imag_fraction']:.1f}% - F_vib unreliable")
        if method == "VDOS":
            warn, spread = vdos_consistency(phase, np.asarray(temperatures, float))
            if warn:
                flags.append(f"F_vib differs by {100 * spread:.1f}% between MD temperatures")
    res["flags"] = flags
    return res
