"""heatup.analyze.terms  --  DRAFT: free-energy terms beyond HA / QHA
======================================================================
Pure functions (numpy only, units in the docstrings) plus thin adapters for ``GibbsAssembler``.  Nothing here
reads a calculator.  Every equation carries its reference; "[verified]" = checked against the paper text during
drafting, "[recall]" = from memory, check before citing.  Tests: ``test_terms.py``.

FRAMEWORK - one nuclear term that REPLACES HA/QHA, not an extra term on top
----------------------------------------------------------------------------
    G(T) = U0(T) + P<V>/N  +  F_vib^Q[ w_eff(T) ]  +  dF_AH(T)      (nuclear; ONE method)
         + F_el(T) + F_mag(T)                                        (orthogonal, per phase)
         + F_conf+orient(T) + F_def(T)                               (ensemble over discrete states)
    and gas references enter the HULL as pseudo-phases with mu(T, p).

    HA   : U0 = E0(V0),               w_eff = w_HA(V0),    dF_AH = 0
    QHA  : U0 = E0(V),                w_eff = w_HA(V),     dF_AH = 0   (V minimises F + PV)
    EHA  : U0 = <U_MD> - n_dof kT/2,  w_eff = w_VDOS(T),   dF_AH = 0   (TDEP prescription; this module)
    TI   : same as EHA plus dF_AH = int_0^1 <U_MD - U_eff>_lambda dlambda  (Hellman 2013, Eq. 23)
    Each line is a different approximation of the SAME quantity: never add two of them.

References (full list at the end of the file).
"""
from __future__ import annotations

import logging
import os

import numpy as np

log = logging.getLogger("heatup")

# ---- constants (CODATA 2018) --------------------------------------------------------------------------------
KB_EV = 8.617333262e-5          # eV/K
KB_MEV = 8.617333262e-2         # meV/K
KB_J = 1.380649e-23
H_J = 6.62607015e-34
HBAR_JS = 1.054571817e-34
AMU_KG = 1.66053906660e-27
EV_J = 1.602176634e-19
GPA_TO_EV_A3 = 1.0 / 160.21766208
N_A = 6.02214076e23


# =================================================================================================================
# 1. NUCLEAR: effective-harmonic free energy with a temperature-dependent reference energy U0  (EHA / "TDEP-lite")
# =================================================================================================================
# F_TDEP = U0 + F_vib                                             Hellman et al. PRB 87, 104111 (2013) Eq. 22 [verified]
# F_vib  = int g(w) [ kT ln(1 - e^{-hw/kT}) + hw/2 ] dw           Eq. 18 [verified]  (quantum formula, ZPE included)
# U0     = < U_MD(t) - 1/2 sum Phi u u >                          Eq. 21 [verified]  (classical MD average)
# F      = U0 + F_vib + int_0^1 <U_MD - U_TDEP>_lambda dlambda    Eq. 23 [verified]  (exact; dF_AH small if model good)
#
# Equipartition shortcut (my derivation, consistent with Eq. 21): for the least-squares force-constant fit, the
# normal equations give <u.(F_MD + Phi u)> = 0, and generalised equipartition gives <u.grad U> = n_dof kT in the
# classical canonical ensemble; hence <1/2 u Phi u> = n_dof kT / 2 and
#       U0 = <U_MD> - n_dof k T / 2 ,   n_dof = 3N - 3        (centre-of-mass motion removed, as in md.py)
# Holds exactly for an unconstrained fit; with symmetry constraints / NPT cell fluctuations it is O(1/N) accurate.
# Check: harmonic crystal -> U0 = E0 and F = HA exactly (test_terms.py::test_u0_equipartition_harmonic).

def u0_from_md(epot_mean_eV: float, n_atoms: int, T: float, n_dof: int | None = None) -> float:
    """U0 per atom (eV) from the mean MD potential energy of a cell of *n_atoms* atoms at temperature T (K)."""
    n_dof = 3 * n_atoms - 3 if n_dof is None else n_dof
    return (float(epot_mean_eV) - 0.5 * n_dof * KB_EV * T) / n_atoms


def f_vib_quantum(omega_meV, weight, T) -> np.ndarray:
    """Quantum harmonic F_vib per atom (eV).  ``omega_meV`` > 0, ``weight`` = modes per atom.  Eq. 18 above."""
    om, w = np.asarray(omega_meV, float), np.asarray(weight, float)
    T = np.atleast_1d(np.asarray(T, float))
    out = np.empty(T.shape)
    for i, t in enumerate(T):
        if t <= 0:
            out[i] = 0.5 * (w * om).sum() * 1e-3
        else:
            x = om / (KB_MEV * t)
            out[i] = (w * (0.5 * om + KB_MEV * t * np.log1p(-np.exp(-np.clip(x, 1e-10, 700))))).sum() * 1e-3
    return out


def f_vib_classical(omega_meV, weight, T) -> np.ndarray:
    """Classical harmonic F per atom (eV): sum_i kT ln(hw_i/kT).  (classical limit of Eq. 18)"""
    om, w = np.asarray(omega_meV, float), np.asarray(weight, float)
    T = np.atleast_1d(np.asarray(T, float))
    return np.array([(w * KB_MEV * t * np.log(om / (KB_MEV * t))).sum() * 1e-3 for t in T])


def quantum_correction(omega_meV, weight, T) -> np.ndarray:
    """F_Q - F_cl of the harmonic model (eV/atom): converts a CLASSICAL free energy (TI, Gibbs-Helmholtz) to a
    quantum one at harmonic level.  -> 0 for T >> w/kB.  Standard (Born & Huang; Wallace, Thermodynamics of
    Crystals 1972) [recall]; the two kernels above are Eq. 18 of Hellman 2013 and its classical limit."""
    return f_vib_quantum(omega_meV, weight, T) - f_vib_classical(omega_meV, weight, T)


def delta_f_ah(lambdas, dU_mean_eV_per_atom) -> float:
    """dF_AH = int_0^1 <U_MD - U_eff>_lambda dlambda  (Hellman 2013 Eq. 23 [verified]); Simpson/trapezoid on the
    lambda grid of the switching runs (Hamiltonian H_lambda = (1-lambda) U_eff + lambda U_MD, NVT)."""
    lam, du = np.asarray(lambdas, float), np.asarray(dU_mean_eV_per_atom, float)
    if lam[0] != 0.0 or lam[-1] != 1.0:
        raise ValueError("lambda grid must span [0, 1]")
    return float(np.trapezoid(du, lam))


def fit_force_constants(u, f, enforce_asr: bool = True) -> np.ndarray:
    """EXPERIMENTAL - DO NOT USE FOR FREE ENERGIES.  Dense, symmetry-free least-squares force constants (eV/A^2),
    TDEP Eq. 4: Phi = F U^+ (sign: f = -Phi u).

    VALIDATION RESULT: exact on noise-free harmonic data (test_terms.py) but ILL-CONDITIONED on real MD: for 64
    atoms and 1200 frames (EMT Cu) it returned frequencies ~9% too high (F error 12-35 meV/atom), because the
    sample covariance of 192 coordinates from 1200 frames is badly conditioned.  Use TDEP / hiphive with lattice
    symmetry (Hellman 2013 Eqs. 7-13) instead.

    ``u``: displacements from the MEAN positions (A), ``f``: MD forces (eV/A); shapes (Nt, N, 3).  Small cells
    only (3N x 3N matrix; Nt*3N >= ~(3N)^2/2 equations needed).  A real TDEP code uses lattice symmetry
    (Eqs. 7-13) - use it for production.  Acoustic sum rule enforced on the 3x3 blocks."""
    u = np.asarray(u, float)
    f = np.asarray(f, float)
    nt, n = u.shape[0], u.shape[1]
    U, F = u.reshape(nt, 3 * n), f.reshape(nt, 3 * n)
    phi_t = -np.linalg.lstsq(U, F, rcond=None)[0]          # F = -U Phi^T
    phi = 0.5 * (phi_t.T + phi_t)
    if enforce_asr:
        blocks = phi.reshape(n, 3, n, 3).copy()
        for i in range(n):
            blocks[i, :, i, :] = 0.0
            blocks[i, :, i, :] = -blocks[i, :, :, :].sum(axis=1)
        phi = blocks.reshape(3 * n, 3 * n)
        phi = 0.5 * (phi + phi.T)
    return phi


def modes_from_force_constants(phi, masses_amu, drop_translations: bool = True):
    """(omega_meV, weight_per_atom) of a dense supercell force-constant matrix; imaginary -> negative omega."""
    n = len(masses_amu)
    m = np.repeat(np.asarray(masses_amu, float), 3)
    d = phi / np.sqrt(np.outer(m, m))                      # eV / (A^2 amu)
    lam = np.linalg.eigvalsh(0.5 * (d + d.T))
    conv = EV_J / (1e-20 * AMU_KG)                         # eV/(A^2 amu) -> s^-2
    om = np.sign(lam) * np.sqrt(np.abs(lam) * conv) * HBAR_JS / EV_J * 1e3      # meV
    order = np.argsort(np.abs(om))
    if drop_translations:
        om = om[order][3:]
    return np.sort(om), np.full(om.shape, 1.0 / n)


def u0_tdep(epot_eV, u, phi) -> float:
    """TDEP Eq. 21: U0 = < U_MD - 1/2 u.Phi.u >  (eV per cell).  Cross-check of :func:`u0_from_md`."""
    u = np.asarray(u, float)
    nt = u.shape[0]
    flat = u.reshape(nt, -1)
    return float(np.mean(np.asarray(epot_eV, float) - 0.5 * np.einsum("ti,ij,tj->t", flat, phi, flat)))


def nuclear_contribution_eha(phase, T, pressure_GPa: float = 0.0):
    """EXPERIMENTAL - NOT VALIDATED.  Contribution (eV/atom) so that ``H0 + contribution = U0(T) + P<V> + F_vib^Q[w_VDOS(T)]``.

    VALIDATION RESULT (EMT Cu, 64 atoms): disagrees with QHA by +4/+12/+20 meV/atom at 300/600/900 K although the
    exact anharmonic correction at fixed volume is only +1.3 (600 K) / +2.6 (900 K).  Known biases: (a) VACF
    spectrum is ~0.016 too high in <ln w> even in the harmonic limit (+2.3..2.6 meV/atom at 600 K), (b) a 64-atom
    cell samples a coarse q-mesh (+1.4/+2.9/+4.3 meV/atom at 300/600/900 K), (c) NPT cell degrees of freedom
    leave ~3 kT per cell inside <U> that ``u0_from_md`` does not subtract.  Prefer QHA + ``anharmonic.f_anh``.

    Replaces ``thermo.f_vib_vdos`` (which was ``F_vib`` only: the static cost of the NPT-expanded cell was
    missing).  Reads ``md/<T>K/analysis.json`` (mean Epot, mean V) and ``vdos.json``.  U0 - H0 is interpolated in
    T with PCHIP through the anchor (0 K, 0); above the highest MD temperature it is extended linearly (flagged).
    """
    from scipy.interpolate import PchipInterpolator
    from heatup.analyze.thermo import MissingData, interpolate_f_vib
    from heatup.analyze.trajectory import load_vdos, run_mdstats
    from heatup.core import ledger
    from heatup.core.layout import read_json

    T = np.asarray(T, float)
    e = phase.energy() or {}
    h0_raw = float(e.get("enthalpy_eV_per_atom", e.get("energy_eV_per_atom", np.nan)))
    pts, sets, flags = [(0.0, 0.0)], [], []
    for t in ledger.md_temperatures(phase):
        v, st = load_vdos(phase, t), run_mdstats(phase, t)
        if not v or not st:
            continue
        th = st["thermo"]
        n = int(read_json(os.path.join(phase.md_run(t), "md.json"))["n_atoms"])
        ep = th["Epot_eV"]["mean"]
        if ep is None or not np.isfinite(ep):
            raise MissingData(f"{phase.tag} {t}K: no potential energies in the trajectory (needed for U0)")
        drift = abs(th["Epot_eV"]["drift_per_ps"]) * th["production_ps"] / n
        if drift > 5e-4:
            flags.append(f"{t}K: Epot drifts {1e3 * drift:.2f} meV/atom over the production run (U0 not converged)")
        Tk = v.get("T_kin_mean_K")
        if Tk and abs(Tk - t) / t > 0.02:
            flags.append(f"{t}K: kinetic T = {Tk:.0f} K (U0 uses the target T)")
        u0 = u0_from_md(ep, n, t)
        pv = pressure_GPa * GPA_TO_EV_A3 * th["V_A3"]["mean"] / n
        pts.append((float(t), u0 + pv - h0_raw))
        sets.append((float(t), v["modes"]))
    if not sets:
        raise MissingData(f"{phase.tag}: no MD spectra / statistics (run_analysis first)")
    for f in flags:
        log.warning("%s %s", phase.tag, f)
    tt, cc = (np.array(a) for a in zip(*sorted(pts)))
    if len(tt) == 2:
        c = np.interp(T, tt, cc)
    else:
        c = PchipInterpolator(tt, cc, extrapolate=False)(T)
        hi = T > tt[-1]
        slope = (cc[-1] - cc[-2]) / (tt[-1] - tt[-2])
        c = np.where(hi, cc[-1] + slope * (T - tt[-1]), c)
    return c + interpolate_f_vib(sets, T)


# =================================================================================================================
# 2. ELECTRONIC  (Mermin finite-T functional at fixed ionic positions / volume)
# =================================================================================================================
# F_el(T) = U_el(T) - T S_el(T) - [U_el(0)],   S_el = -kB int g [ f ln f + (1-f) ln(1-f) ] de,  f = Fermi(e; mu(T), T)
# mu(T) from electron-number conservation.  Mermin, Phys. Rev. 137, A1441 (1965) [recall]; decomposition
# F = (U_tot - T S_el) + F_vib: Hellman 2013 Eq. 17 [verified].  Sommerfeld limit: F_el - F_el(0) = -(pi^2/6) g(EF) (kT)^2.
# NOT for gapped materials (exponentially small).  Use the MD-snapshot-averaged DOS for metals (electron-phonon).

def electronic_free_energy(eps_eV, dos_per_eV, T) -> np.ndarray:
    """F_el(T) - F_el(0), eV/atom.  ``eps_eV`` relative to E_F; ``dos_per_eV`` states/eV/atom (spin included)."""
    eps, g = np.asarray(eps_eV, float), np.asarray(dos_per_eV, float)
    T = np.atleast_1d(np.asarray(T, float))
    occ0 = (eps <= 0).astype(float)
    n0 = np.trapezoid(g * occ0, eps)
    u0 = np.trapezoid(eps * g * occ0, eps)
    out = np.zeros(T.shape)

    def fermi(mu, t):
        return 1.0 / (1.0 + np.exp(np.clip((eps - mu) / (KB_EV * t), -500, 500)))

    for i, t in enumerate(T):
        if t <= 0:
            continue
        lo, hi = eps.min(), eps.max()
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if np.trapezoid(g * fermi(mid, t), eps) < n0:
                lo = mid
            else:
                hi = mid
        f = np.clip(fermi(0.5 * (lo + hi), t), 1e-300, 1 - 1e-16)
        s = -KB_EV * np.trapezoid(g * (f * np.log(f) + (1 - f) * np.log(1 - f)), eps)
        u = np.trapezoid(eps * g * f, eps) - u0
        out[i] = u - t * s
    return out


# =================================================================================================================
# 3. MAGNETIC  (Inden-Hillert-Jarl; thermal part, third-law consistent)
# =================================================================================================================
# G_mag(T) = kT ln(beta+1) g(tau),  tau = T/Tc,  beta = mean moment (mu_B)      Hillert & Jarl, Calphad 2, 227 (1978);
#                                                                                Dinsdale, Calphad 15, 317 (1991)
# g(tau<1) = 1 - [ 79/(140 p tau) + 474/497 (1/p - 1)(tau^3/6 + tau^9/135 + tau^15/600) ] / A
# g(tau>1) = - ( tau^-5/10 + tau^-15/315 + tau^-25/1500 ) / A
# A = 518/1125 + 11692/15975 (1/p - 1);  p = 0.40 (bcc), 0.28 (fcc, hcp, other)   [verified: Khina 2018; AGH 2015;
# the previous docstring of f_mag had bcc/other swapped].  AFM: Tc -> Tneel and beta -> beta/(-1 bcc | -3 other) [recall].
#
# THIRD-LAW FIX.  Writing G_mag(T->0) = kT ln(beta+1) + g0 with g0 = -k ln(beta+1) 79 Tc/(140 p A) shows the CALPHAD
# G_mag carries S_mag(0) = -k ln(beta+1): the spin-disorder entropy is booked in the paramagnetic *lattice* part of a
# CALPHAD assessment.  Phonon/DFT free energies of an ordered state do not contain it, so it must be added back:
#       F_mag^thermal(T) = G_mag(T) - g0 - kT ln(beta+1)       ->  S(0) = 0,  S(inf) = k ln(beta+1)
# (the version in thermo.f_mag omits the last term: its S(0+) = -k ln(beta+1) < 0; test_terms.py shows it.)

def hjl_structure_factor(lattice: str) -> float:
    return 0.40 if lattice.lower() == "bcc" else 0.28


def _g_tau(tau: float, p: float) -> float:
    A = 518.0 / 1125.0 + 11692.0 / 15975.0 * (1.0 / p - 1.0)
    if tau < 1.0:
        return 1.0 - (79.0 / (140.0 * p * tau) + 474.0 / 497.0 * (1.0 / p - 1.0)
                      * (tau ** 3 / 6 + tau ** 9 / 135 + tau ** 15 / 600)) / A
    return -(tau ** -5 / 10 + tau ** -15 / 315 + tau ** -25 / 1500) / A


def magnetic_free_energy(T, Tc: float, beta: float, p: float = 0.40, third_law: bool = True) -> np.ndarray:
    """Thermal magnetic free energy per atom (eV).  ``third_law=False`` reproduces the current ``thermo.f_mag``."""
    T = np.atleast_1d(np.asarray(T, float))
    if Tc <= 0 or beta <= 0:
        return np.zeros(T.shape)
    A = 518.0 / 1125.0 + 11692.0 / 15975.0 * (1.0 / p - 1.0)
    lnb = np.log(beta + 1.0)
    g0 = -KB_EV * lnb * 79.0 * Tc / (140.0 * p * A)
    out = np.zeros(T.shape)
    for i, t in enumerate(T):
        if t > 0:
            out[i] = KB_EV * t * lnb * _g_tau(t / Tc, p) - g0 - (KB_EV * t * lnb if third_law else 0.0)
    return out


# =================================================================================================================
# 4. CONFIGURATIONAL  (+ orientational, if you enumerate orientations as configurations)
# =================================================================================================================
# (a) ideal mixing:   S = -kB sum_sites sum_species x ln x                          upper bound [verified: 2608.10465]
# (b) enumerated configurations with their own vibrational free energy (coarse graining):
#       F(T) = -kT ln sum_s  g_s exp( -[E_s + F_vib,s(T)] / kT )                    van de Walle & Ceder, RMP 74, 11
#       (2002) [recall];  g_s = degeneracy.  Finite-cell bias: only the cell's own configurations are sampled.
# (c) lattice-model Monte Carlo (cluster expansion):  d(beta F)/d beta = <E>  =>
#       beta F(beta) = -S_inf/kB + int_0^beta <E>_beta' d beta'                     van de Walle & Asta, MSMSE 10, 521
#       (2002) [recall; cited in the ATAT-based phase-diagram literature]; S_inf/kB = ln(#states) at beta = 0.
#       Same identity (Gibbs-Helmholtz) as temperature integration of MD enthalpies.  Packages: icet, smol, CASM, ATAT.

def ideal_mixing_free_energy(site_occupancies, T) -> np.ndarray:
    """-T S_id per atom position.  ``site_occupancies``: list of {species: fraction} (vacancies as a species)."""
    s = 0.0
    for site in site_occupancies:
        for x in site.values():
            if 0.0 < x < 1.0:
                s -= KB_EV * x * np.log(x)
    return -np.atleast_1d(np.asarray(T, float)) * s / len(site_occupancies)


def coarse_grained_free_energy(E_cell_eV, Fvib_cell_eV, T, n_atoms: int, degeneracy=None) -> np.ndarray:
    """Boltzmann ensemble over enumerated configurations (per CELL energies, eV) -> F per atom (eV).
    ``Fvib_cell_eV``: (n_config, n_T) vibrational free energy of each configuration (same method for all)."""
    E = np.asarray(E_cell_eV, float)[:, None]
    Fv = np.asarray(Fvib_cell_eV, float)
    g = np.ones(len(E)) if degeneracy is None else np.asarray(degeneracy, float)
    T = np.asarray(T, float)
    a = -(E + Fv) / (KB_EV * T[None, :]) + np.log(g)[:, None]
    amax = a.max(axis=0)
    return -KB_EV * T * (amax + np.log(np.exp(a - amax).sum(axis=0))) / n_atoms


def free_energy_by_beta_integration(T, E_mean_eV, E_inf_eV, s_inf_over_kB):
    """Free energy per atom (eV) from MC averages <E>(T) (eV/atom) at the temperatures *T* (K):
    beta F = -s_inf + int_0^beta <E> d beta'.  ``E_inf`` = <E> at beta = 0 (random state);
    ``s_inf_over_kB`` = entropy per atom at beta = 0 in units of kB (ideal: -sum x ln x)."""
    T = np.asarray(T, float)
    beta = 1.0 / (KB_EV * T)
    order = np.argsort(beta)
    b = np.concatenate([[0.0], beta[order]])
    e = np.concatenate([[E_inf_eV], np.asarray(E_mean_eV, float)[order]])
    integ = np.concatenate([[0.0], np.cumsum(0.5 * (e[1:] + e[:-1]) * np.diff(b))])
    bf = -s_inf_over_kB + integ
    out = np.empty_like(bf[1:])
    out[order] = bf[1:] / beta[order]
    return out


# =================================================================================================================
# 5. ROTORS (hindered rotation / orientational entropy of molecular ions)
# =================================================================================================================
# 1D hindered rotor:  H = -B d^2/dphi^2 + (V0/2)(1 - cos n phi),  B = hbar^2/(2 I),  barrier V0, n equivalent minima.
# Exact quantum partition function by diagonalisation in the plane-wave basis exp(i m phi) (Pitzer & Gwinn, J. Chem.
# Phys. 10, 428 (1942) [recall: solved this problem tabulated]).  Z_H = sum exp(-E/kT) counts every one of the n
# minima; the physical partition function is  Z = Z_H / sigma,  with sigma = ``symmetry_number`` = number of the n
# orientations that are the SAME microstate because they differ by a permutation of identical atoms (CH3 about its
# C3 axis: sigma = 3; a cation whose n orientations are distinguishable relative to the host: sigma = 1).
# Limits (tested): V0 -> 0 free rotor F = -kT ln( sqrt(pi kT / B) / sigma );  V0 large: harmonic with
# hw = n sqrt(B V0) and an extra -kT ln(n / sigma).  The orientational entropy k ln(n/sigma) is INSIDE this term ->
# do not add it again in F_conf, and drop the librational mode(s) from the phonon sum.  Simpler alternative for isolated low modes: quasi-RRHO interpolation, Grimme, Chem. Eur. J.
# 18, 9955 (2012) [verified through the ORCA / Q-Chem manuals].  Rotor + phonon coupling is neglected.

def rotational_constant_meV(inertia_amu_A2: float) -> float:
    """B = hbar^2 / (2 I) in meV for I in amu A^2  (about 2.0905 meV / I)."""
    return HBAR_JS ** 2 / (2.0 * inertia_amu_A2 * AMU_KG * 1e-20) / EV_J * 1e3


def hindered_rotor_levels(B_meV: float, V0_meV: float, n_fold: int, m_max: int = 60) -> np.ndarray:
    m = np.arange(-m_max, m_max + 1)
    H = np.diag(B_meV * m.astype(float) ** 2 + 0.5 * V0_meV)
    k = len(m)
    for i in range(k - n_fold):
        H[i, i + n_fold] = H[i + n_fold, i] = -0.25 * V0_meV
    return np.linalg.eigvalsh(H)


def hindered_rotor_free_energy(T, B_meV: float, V0_meV: float, n_fold: int, m_max: int = 60,
                               symmetry_number: int = 1) -> np.ndarray:
    """Free energy per rotor (eV): -kT ln( Z_H / sigma ), Z_H over all n minima, sigma = ``symmetry_number``."""
    lev = hindered_rotor_levels(B_meV, V0_meV, n_fold, m_max)
    T = np.atleast_1d(np.asarray(T, float))
    out = np.empty(T.shape)
    for i, t in enumerate(T):
        a = -(lev - lev[0]) / (KB_MEV * t)
        out[i] = (lev[0] - KB_MEV * t * (np.log(np.exp(a).sum()) - np.log(symmetry_number))) * 1e-3
    return out


# =================================================================================================================
# 6. GAS REFERENCES (ideal gas; O2, N2, ...)
# =================================================================================================================
# mu_O(T,p) = 1/2 mu_O2 = mu_O(T,p0) + 1/2 kT ln(p/p0)                                 Reuter & Scheffler, PRB 65,
# mu_O(T,p0) = 1/2 [ H(T) - H(0) ] - 1/2 T [ S(T) - S(0) ]   (zero: 1/2 E_O2^total)    035406 (2001), Eqs. 21-22 [verified]
# Here H(T)-H(0) and S(T) from the ideal-gas partition functions (McQuarrie, Statistical Mechanics) [recall]:
#   S_trans/k = ln[ (2 pi m k T/h^2)^(3/2) kT/p ] + 5/2        H_trans = 5/2 kT
#   S_rot/k   = ln[T/(sigma Theta_rot)] + 1 (linear)           H_rot   = kT      (valid T >> Theta_rot; NOT H2)
#   S_vib/k   = sum x/(e^x-1) - ln(1-e^-x), x = Theta_vib/T    H_vib-ZPE = k sum Theta/(e^x-1)
#   S_el/k    = ln g_el  (O2: 3)
# Total G of the molecule = E_el + ZPE + [H(T)-H(0)] - T S.  Use E_el corrected so that oxide formation energies at
# 0 K are right (DFT/MLIP O2 over-binding): apply ONCE, via hull.reference_energy_shift or the argument ``E_el``.
# Spectroscopic constants: NIST Chemistry WebBook (check; values below are typical).

GAS = {
    "O2": dict(mass=31.998, theta_rot=2.069, sigma=2, theta_vib=(2273.6,), g_el=3, linear=True, atoms=2),
    "N2": dict(mass=28.0134, theta_rot=2.875, sigma=2, theta_vib=(3393.5,), g_el=1, linear=True, atoms=2),
}


def ideal_gas_thermal(T, p_bar, mass, theta_rot, sigma, theta_vib=(), g_el=1, linear=True, atoms=2):
    """(H(T)-H(0), S(T)) per molecule in eV and eV/K for T > 0 (arrays)."""
    T = np.asarray(T, float)
    m = mass * AMU_KG
    s_tr = np.log((2 * np.pi * m * KB_J * T / H_J ** 2) ** 1.5 * KB_J * T / (p_bar * 1e5)) + 2.5
    h = 2.5 + 1.0
    s_rot = np.log(T / (sigma * theta_rot)) + 1.0
    s_vib = 0.0
    for th in theta_vib:
        x = th / T
        h = h + x / np.expm1(x)
        s_vib = s_vib + x / np.expm1(x) - np.log1p(-np.exp(-x))
    s = s_tr + s_rot + s_vib + np.log(g_el)
    return h * KB_EV * T, s * KB_EV


def gas_free_energy_per_atom(species: str, T, p_bar: float, E_el_eV: float) -> np.ndarray:
    """G per ATOM (eV) of an ideal-gas reference for the hull (pure-element pseudo-phase): ``E_el_eV`` = corrected
    electronic energy of the molecule.  Add to ``comps`` as ``({"O": 1.0}, G)``."""
    c = GAS[species]
    T = np.asarray(T, float)
    zpe = 0.5 * KB_EV * sum(c["theta_vib"])
    hh, ss = ideal_gas_thermal(np.maximum(T, 1e-3), p_bar, c["mass"], c["theta_rot"], c["sigma"], c["theta_vib"],
                               c["g_el"], c["linear"], c["atoms"])
    g = E_el_eV + zpe + hh - T * ss
    g = np.where(T > 0, g, E_el_eV + zpe)
    return g / c["atoms"]


# =================================================================================================================
# 7. POINT DEFECTS (dilute, independent)
# =================================================================================================================
# Per site with formation FREE energy g_f(T) = E_f + F_vib,f - T S_other (vibrational part from the same method):
#   Z = 1 + d exp(-g_f/kT)  ->  F = -kT ln Z ;  dilute limit: c = d exp(-g_f/kT), F = -kT c.   [textbook;
#   Freysoldt et al., RMP 86, 253 (2014) for g_f [recall]].  Use g_f computed with TI/MLIP for anharmonic cases
#   (Hai et al. 2025, materialsmodeling.org/assets/publications/HaiBerBer25.pdf [found in search]).
# Double counting: if vacancies/antisites are already in ``site_occupancies`` (F_conf) do NOT also add this.

def defect_free_energy_per_site(g_f_eV, T, degeneracy: float = 1.0) -> np.ndarray:
    T = np.atleast_1d(np.asarray(T, float))
    gf = np.broadcast_to(np.asarray(g_f_eV, float), T.shape)
    a = -gf / (KB_EV * np.maximum(T, 1e-9))
    return -KB_EV * T * np.logaddexp(0.0, a + np.log(degeneracy))


# =================================================================================================================
# 8. ASSEMBLER ADAPTERS (see INTEGRATION notes in the reply)
# =================================================================================================================

def register_terms(asm, cfg) -> None:
    """Example: in ``thermo.build_assembler`` replace the F_vib[VDOS] registration and the ad-hoc F_mag/F_el by

        VIB_METHODS["VDOS"] = lambda ph, T: nuclear_contribution_eha(ph, T, cfg.relax.pressure_GPa)
    """
    raise NotImplementedError("see reply: wire-up snippets")


# =================================================================================================================
# REFERENCES
# =================================================================================================================
# [1] O. Hellman, P. Steneteg, I. A. Abrikosov, S. I. Simak, Phys. Rev. B 87, 104111 (2013); arXiv:1303.1145.
#     Eqs. 1, 17, 18, 20, 21, 22, 23 used here. [verified]
# [2] K. Reuter, M. Scheffler, Phys. Rev. B 65, 035406 (2001); arXiv:cond-mat/0107229. Eqs. 14-17, 21-22, A1-A9. [verified]
# [3] M. Hillert, M. Jarl, Calphad 2, 227 (1978); A. T. Dinsdale, Calphad 15, 317 (1991) (SGTE). Coefficients and
#     p = 0.40 bcc / 0.28 other cross-checked in Khina, Adv. Mater. Technol. 2018, 041; Comput. Methods Mater. Sci.
#     15, 179 (2015). [verified]
# [4] A. van de Walle, Calphad 33, 266 (2009); arXiv:0906.1608 (ATAT: configurational + vibrational + electronic
#     entropy). A. van de Walle, G. Ceder, Rev. Mod. Phys. 74, 11 (2002). [recall]
# [5] S. Grimme, Chem. Eur. J. 18, 9955 (2012) (quasi-RRHO); K. S. Pitzer, W. D. Gwinn, J. Chem. Phys. 10, 428 (1942). [recall]
# [6] D. Frenkel, A. J. C. Ladd, J. Chem. Phys. 81, 3188 (1984); R. Freitas, M. Asta, M. de Koning, Comput. Mater. Sci.
#     112, 333 (2016) (TI for solids; cited in the calorine tutorial). [recall]
# [7] N. D. Mermin, Phys. Rev. 137, A1441 (1965). [recall]
