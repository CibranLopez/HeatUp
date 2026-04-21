"""heatup.free_energy
============================
Generalised Gibbs free energy G(T, P) assembler.

The total Gibbs free energy per atom is decomposed as:

    G(T, P) = E₀  +  F_vib(T)  +  F_el(T)  +  F_mag(T)
                   +  F_conf(T) +  PV(T, P)  +  [extensible]

Each contribution is implemented as an independent, optional module that
reads data from a standard file in the material's symmetry directory and
returns a contribution as a function of temperature.  If the required data
file is absent, the contribution returns zero silently (with a debug message),
so the framework degrades gracefully as data become available.

This design allows the framework to be used at different levels of theory:

    Level 0  (always):     E₀  (MLIP ground-state energy)
    Level 1  (+ phonons):  + F_vib   (harmonic or anharmonic)
    Level 2  (+ e-DOS):    + F_el    (electronic free energy)
    Level 3  (+ moments):  + F_mag   (magnetic free energy)
    Level 4  (+ disorder): + F_conf  (configurational entropy)
    Level 5  (+ P):        + PV      (pressure–volume work)

Adding a new contribution requires only:
    1. Writing a function with signature
           contribution(sym_dir, T_array) → np.ndarray  [eV/atom]
    2. Registering it in FREE_ENERGY_CONTRIBUTIONS.

Physical conventions
--------------------
- Energies in **eV per atom** throughout.
- Temperature in **K**.
- Pressure in **GPa** (converted internally to eV/Å³).

File conventions
----------------
Each contribution reads a JSON file from the symmetry directory:

    relaxation/energy.json          → {"energy_eV_per_atom": float}
    phonons/dos.json                → {"energies_eV": [...], "weights": [...]}
    aimd/<T>K/anharmonic_phonons/   → computed by heatup.vibrational
    electronic/edos.json            → {"energies_eV": [...], "dos_per_eV": [...],
                                        "fermi_energy_eV": float,
                                        "n_electrons_per_atom": float}
    magnetic/moments.json           → {"mean_moment_muB": float,
                                        "exchange_J_meV": float,
                                        "spin": float}
    disorder/site_occupancies.json  → {"sites": [{"species": {"A": x, "B": 1-x}, ...}]}
    equation_of_state/eos.json      → {"volumes_A3": [...], "energies_eV": [...]}
"""

from __future__ import annotations

import json
import os
import warnings
from typing import Callable, Sequence

import numpy as np

from heatup import config

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
# A contribution is any callable (sym_dir, T_array) → np.ndarray [eV/atom].
Contribution = Callable[[str, np.ndarray], np.ndarray]


# =============================================================================
# Ground-state energy
# =============================================================================

def e0(sym_dir: str) -> float | None:
    """Return the MLIP ground-state energy per atom (eV), or None."""
    path = os.path.join(sym_dir, "relaxation", "energy.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return float(json.load(fh)["energy_eV_per_atom"])


# =============================================================================
# Vibrational contribution  F_vib(T)
# =============================================================================

def _bose_einstein(omega_eV: np.ndarray, T: float) -> np.ndarray:
    """Bose–Einstein occupation: n(ω, T) = 1/(exp(ω/kT) − 1)."""
    if T <= 0:
        return np.zeros_like(omega_eV)
    x = omega_eV / (config.KB_EV * T)
    return np.where(x > 500, 0.0, 1.0 / np.expm1(np.clip(x, 1e-12, 500.0)))


def _helmholtz_integrand(omega_eV: np.ndarray, T: float) -> np.ndarray:
    """F_vib kernel: ½ω + kT ln(1 − e^{−ω/kT})."""
    if T <= 0:
        return 0.5 * omega_eV
    x = np.clip(omega_eV / (config.KB_EV * T), 1e-12, 500.0)
    return 0.5 * omega_eV + config.KB_EV * T * np.log1p(-np.exp(-x))


def harmonic_f_vib(sym_dir: str, T_array: np.ndarray) -> np.ndarray:
    """Harmonic vibrational free energy per atom from phonons/dos.json.

    Uses the quantum harmonic oscillator expression:
        F_vib(T) = ∫ g(ω) [½ω + kT ln(1−e^{−ω/kT})] dω

    Args:
        sym_dir: Symmetry directory path.
        T_array: Temperature array (K), shape (N,).

    Returns:
        F_vib contribution array, shape (N,), in eV/atom.
        Returns zeros if dos.json is absent.
    """
    path = os.path.join(sym_dir, "phonons", "dos.json")
    if not os.path.exists(path):
        return np.zeros_like(T_array)

    with open(path) as fh:
        d = json.load(fh)
    en = np.array(d["energies_eV"], dtype=float)
    wt = np.array(d["weights"],     dtype=float)

    # Strip acoustic / numerical noise.
    mask = en > config.OMEGA_MIN_MEV * config.MEV_TO_EV
    en, wt = en[mask], wt[mask]
    norm = np.trapezoid(wt, en)
    if norm <= 0:
        return np.zeros_like(T_array)
    wt /= norm

    result = np.empty_like(T_array, dtype=float)
    for i, T in enumerate(T_array):
        result[i] = float(np.trapezoid(wt * _helmholtz_integrand(en, T), en))
    return result


def anharmonic_f_vib(
        sym_dir: str,
        T_array: np.ndarray,
        device: str = config.DEFAULT_DEVICE,
) -> np.ndarray:
    """Anharmonic vibrational free energy per atom from AIMD VACF/VDOS.

    Reads cached ``aimd/<T>K/anharmonic_phonons/vdos.json`` files (produced
    by :mod:`heatup.vibrational`) and averages the VDOS across all
    available MD temperatures.  Falls back to harmonic if no AIMD data exist.

    Args:
        sym_dir: Symmetry directory path.
        T_array: Temperature array (K), shape (N,).
        device:  Compute device if a new AIMD simulation must be triggered.

    Returns:
        F_vib contribution array, shape (N,), in eV/atom.
    """
    aimd_dir = os.path.join(sym_dir, "aimd")
    vdos_list: list[tuple[np.ndarray, np.ndarray]] = []

    if os.path.isdir(aimd_dir):
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
                    vd = json.load(fh)
                # Check trajectory quality flag written by VDOS computation.
                n_frames = int(vd.get("n_frames", config.VIB_MIN_FRAMES))
                if n_frames < config.VIB_MIN_FRAMES:
                    import warnings as _warn
                    _warn.warn(
                        f"anharmonic_f_vib: only {n_frames} production frames in "
                        f"{vdos_path} (minimum recommended: {config.VIB_MIN_FRAMES}). "
                        f"VDOS may have artefactual broadening near ω = 0.",
                        stacklevel=3,
                    )
                om_mev = np.array(vd["omega_mev"], dtype=float)
                g      = np.array(vd["vdos"],      dtype=float)
                # Convert meV → eV.
                om_eV = om_mev * config.MEV_TO_EV
                mask  = om_eV > config.OMEGA_MIN_MEV * config.MEV_TO_EV
                om_eV, g = om_eV[mask], g[mask]
                norm = np.trapezoid(g, om_eV)
                if norm > 0:
                    g /= norm
                vdos_list.append((om_eV, g))
            except Exception:
                continue

    if not vdos_list:
        # Fallback to harmonic.
        return harmonic_f_vib(sym_dir, T_array)

    # Average VDOS across MD temperatures.
    om_ref, g_ref = vdos_list[0]
    g_avg = g_ref.copy()
    for om_i, g_i in vdos_list[1:]:
        g_avg += np.interp(om_ref, om_i, g_i, left=0.0, right=0.0)
    g_avg /= len(vdos_list)
    norm = np.trapezoid(g_avg, om_ref)
    if norm > 0:
        g_avg /= norm

    result = np.empty_like(T_array, dtype=float)
    for i, T in enumerate(T_array):
        result[i] = float(np.trapezoid(g_avg * _helmholtz_integrand(om_ref, T), om_ref))
    return result


# =============================================================================
# Electronic contribution  F_el(T)
# =============================================================================

def electronic_f_el(sym_dir: str, T_array: np.ndarray) -> np.ndarray:
    """Electronic free energy per atom from the electronic DOS.

    Reads ``electronic/edos.json`` with keys:
        ``energies_eV``         Energy grid relative to E_F (eV).
        ``dos_per_eV``          DOS in states / (eV · atom).
        ``fermi_energy_eV``     Fermi energy (eV) — used only for reference.
        ``n_electrons_per_atom``Number of valence electrons per atom.

    The electronic free energy is computed as:
        F_el(T) = U_el(T) − T·S_el(T)
    where
        f(ε, T)  = 1/(1 + exp(ε/kT))              [Fermi–Dirac at μ(T)]
        U_el(T)  = ∫ ε · g(ε) · f(ε,T) dε − E_band,0
        S_el(T)  = −k_B ∫ g(ε)[f ln f + (1−f) ln(1−f)] dε

    The chemical potential μ(T) is solved self-consistently at each T to
    conserve the electron number.  For metals this is the dominant correction
    to E₀ at elevated temperatures; for insulators/semiconductors with a
    band gap > ~1 eV this contribution is negligible at T < 2000 K.

    Args:
        sym_dir: Symmetry directory path.
        T_array: Temperature array (K), shape (N,).

    Returns:
        F_el contribution array, shape (N,), in eV/atom.
        Returns zeros if edos.json is absent.
    """
    path = os.path.join(sym_dir, "electronic", "edos.json")
    if not os.path.exists(path):
        return np.zeros_like(T_array)

    with open(path) as fh:
        d = json.load(fh)

    eps  = np.array(d["energies_eV"],     dtype=float)   # relative to E_F
    g    = np.array(d["dos_per_eV"],      dtype=float)   # states/(eV·atom)
    n_el = float(d["n_electrons_per_atom"])

    # Reference electron energy at T=0 (μ = 0 by construction).
    n_ref = float(np.trapezoid(g * _fermi_dirac(eps, 0.0, 0.0), eps))
    if abs(n_ref - n_el) > 0.5:
        # DOS does not integrate to n_el at T=0 — data may be unnormalised.
        # Rescale g to correct electron number.
        if n_ref > 0:
            g *= n_el / n_ref

    # Ground-state band energy (T=0 reference, μ=0).
    f0       = _fermi_dirac(eps, 0.0, 0.0)
    e_band_0 = float(np.trapezoid(eps * g * f0, eps))

    result = np.empty_like(T_array, dtype=float)
    for i, T in enumerate(T_array):
        if T <= 0.0:
            result[i] = 0.0
            continue
        mu  = _solve_mu(eps, g, n_el, T)
        f   = _fermi_dirac(eps, mu, T)
        # Clip f away from 0 and 1 to avoid log(0).
        fc  = np.clip(f,  1e-300, 1.0 - 1e-300)
        fc2 = np.clip(1 - f, 1e-300, 1.0 - 1e-300)
        U_el = float(np.trapezoid(eps * g * f, eps)) - e_band_0
        S_el = -config.KB_EV * float(
            np.trapezoid(g * (fc * np.log(fc) + fc2 * np.log(fc2)), eps)
        )
        result[i] = U_el - T * S_el

    return result


def _fermi_dirac(eps: np.ndarray, mu: float, T: float) -> np.ndarray:
    """Fermi–Dirac distribution f(ε; μ, T)."""
    if T <= 0.0:
        return np.where(eps <= mu, 1.0, 0.0).astype(float)
    x = (eps - mu) / (config.KB_EV * T)
    return 1.0 / (1.0 + np.exp(np.clip(x, -500, 500)))


def _solve_mu(
        eps: np.ndarray,
        g: np.ndarray,
        n_el: float,
        T: float,
        tol: float = 1e-6,
        max_iter: int = 100,
) -> float:
    """Bisection solver for the chemical potential μ(T) at electron count n_el."""
    mu_lo, mu_hi = float(eps.min()), float(eps.max())
    for _ in range(max_iter):
        mu_mid = 0.5 * (mu_lo + mu_hi)
        n_mid  = float(np.trapezoid(g * _fermi_dirac(eps, mu_mid, T), eps))
        if n_mid < n_el:
            mu_lo = mu_mid
        else:
            mu_hi = mu_mid
        if mu_hi - mu_lo < tol:
            break
    return 0.5 * (mu_lo + mu_hi)


# =============================================================================
# Magnetic contribution  F_mag(T)
# =============================================================================

def magnetic_f_mag(sym_dir: str, T_array: np.ndarray) -> np.ndarray:
    """Magnetic free energy per atom in the Weiss mean-field approximation.

    Reads ``magnetic/moments.json`` with keys:
        ``mean_moment_muB``  Mean magnetic moment per atom in μ_B.
        ``exchange_J_meV``   Effective exchange constant J in meV
                             (from Curie temperature: T_C = 2JS(S+1)/3k_B).
        ``spin``             Effective spin quantum number S.

    The Weiss model gives the magnetic Helmholtz free energy per site as:
        F_mag(T) ≈ −k_B T ln(2S + 1)    for T ≫ T_C  (paramagnetic limit)
        F_mag(T) ≈ −J·S²                for T ≪ T_C  (fully ordered)

    We use the interpolation of Inden and Hillert [Inden 1976; Dinsdale 1991]:

        F_mag(T) = −R·T_C · f(τ) · ln(β + 1)

    where τ = T/T_C, β = mean_moment_muB, and f(τ) is the polynomial
    from the CALPHAD literature.  This is the standard approach used in
    CALPHAD modelling and is accurate for 3d transition metals and their
    alloys.

    For systems with negligible magnetism (mean_moment_muB < 0.1 μ_B),
    this contribution is set to zero.

    Args:
        sym_dir: Symmetry directory path.
        T_array: Temperature array (K), shape (N,).

    Returns:
        F_mag contribution array, shape (N,), in eV/atom.
        Returns zeros if moments.json is absent.
    """
    path = os.path.join(sym_dir, "magnetic", "moments.json")
    if not os.path.exists(path):
        return np.zeros_like(T_array)

    with open(path) as fh:
        d = json.load(fh)

    mu_B = float(d["mean_moment_muB"])
    J    = float(d["exchange_J_meV"]) * config.MEV_TO_EV   # → eV
    S    = float(d.get("spin", 0.5 * mu_B))

    if mu_B < 0.1:
        return np.zeros_like(T_array)

    # Curie temperature from mean-field: T_C = 2JS(S+1)/(3k_B)
    T_C = 2.0 * J * S * (S + 1.0) / (3.0 * config.KB_EV)
    if T_C <= 0:
        return np.zeros_like(T_array)

    # Reference: Inden–Hillert polynomial (CALPHAD form).
    # Dinsdale, A. T. CALPHAD 15, 317 (1991), Eq. 5.
    def _f_ind(tau: float) -> float:
        """Inden–Hillert f(τ) function (purely dimensionless polynomial).
        
        Reference: Dinsdale, CALPHAD 15, 317 (1991), Eq. 5.
        The polynomial coefficients are dimensionless; no kB factor here.
        """
        if tau <= 1.0:
            return 1.0 - (
                tau**3 / 6.0 + tau**9 / 135.0 + tau**15 / 600.0
            ) / _D_below
        else:
            return -(tau**-5 / 10.0 + tau**-15 / 315.0 + tau**-25 / 1500.0) / _D_above

    # Normalisation constants from Inden (1976).
    _D_below = 0.6549348
    _D_above = 0.6509262

    # Magnetic free energy per atom.
    ln_beta = np.log(mu_B + 1.0)   # Dinsdale convention: β = Bohr magneton / atom
    result = np.empty_like(T_array, dtype=float)
    for i, T in enumerate(T_array):
        tau = T / T_C if T_C > 0 else 1e6
        result[i] = -config.KB_EV * T * _f_ind(float(tau)) * ln_beta

    return result


# =============================================================================
# Configurational contribution  F_conf(T)   [mixing entropy]
# =============================================================================

def configurational_f_conf(sym_dir: str, T_array: np.ndarray) -> np.ndarray:
    """Configurational (mixing) free energy from site occupancies.

    Reads ``disorder/site_occupancies.json`` with structure::

        {
            "sites": [
                {"species": {"A": 0.5, "B": 0.5}},
                {"species": {"A": 1.0}},
                ...
            ]
        }

    The ideal configurational entropy per atom is:
        S_conf = −k_B / N_atoms × Σ_sites Σ_species x_ij · ln(x_ij)

    where x_ij is the fractional occupancy of species j on site i.
    Ordered sites (single species, x = 1) contribute zero.

    This contribution is relevant for:
        - Rocksalt/spinel solid solutions
        - Off-stoichiometric compounds
        - High-entropy materials
        - Partially disordered superionic conductors (e.g., Li₇La₃Zr₂O₁₂
          where Li occupies multiple partially occupied sites)

    Args:
        sym_dir: Symmetry directory path.
        T_array: Temperature array (K), shape (N,).

    Returns:
        F_conf = −T · S_conf contribution array, shape (N,), in eV/atom.
        Returns zeros if site_occupancies.json is absent.
    """
    path = os.path.join(sym_dir, "disorder", "site_occupancies.json")
    if not os.path.exists(path):
        return np.zeros_like(T_array)

    with open(path) as fh:
        d = json.load(fh)

    sites = d.get("sites", [])
    if not sites:
        return np.zeros_like(T_array)

    # Compute S_conf per atom.
    s_conf = 0.0
    n_sites = len(sites)
    for site in sites:
        occupancies = site.get("species", {})
        for occ in occupancies.values():
            if 0.0 < occ < 1.0:
                s_conf -= config.KB_EV * occ * np.log(occ)

    # Normalise per atom (divide by number of sites, which ≈ atoms/formula unit).
    s_conf /= n_sites

    # Cowley short-range order correction (optional).
    # If disorder/sro_parameters.json exists, apply the Warren–Cowley correction:
    # S_conf^corr = S_conf^ideal × (1 - Σ_shells α_shell × z_shell / N_bonds)
    # where α_shell are Cowley SRO parameters from the RDF.
    sro_path = os.path.join(sym_dir, "disorder", "sro_parameters.json")
    if os.path.exists(sro_path):
        try:
            with open(sro_path) as _fh:
                sro = json.load(_fh)
            # sro_factor ∈ [0, 1]: 1 = ideal mixing, 0 = fully ordered
            sro_factor = float(sro.get("sro_factor", 1.0))
            s_conf *= sro_factor
        except Exception:
            pass  # fall back to ideal mixing silently

    return np.array([-T * s_conf for T in T_array], dtype=float)


# =============================================================================
# PV contribution  (pressure–volume work)
# =============================================================================

def pv_contribution(
        sym_dir: str,
        T_array: np.ndarray,
        pressure_GPa: float = 0.0,
) -> np.ndarray:
    """PV term for finite-pressure calculations.

    Reads the equation of state from ``equation_of_state/eos.json``::

        {
            "volumes_A3":  [V₁, V₂, ...],     # per atom
            "energies_eV": [E₁, E₂, ...]      # per atom
        }

    and fits a Birch–Murnaghan EOS to extract the equilibrium volume V(T)
    at the given pressure.  For zero pressure (default), V(T) ≈ V₀ and
    the PV term is negligible.

    Args:
        sym_dir:       Symmetry directory path.
        T_array:       Temperature array (K), shape (N,).
        pressure_GPa:  External pressure in GPa.

    Returns:
        PV contribution array, shape (N,), in eV/atom.
        Returns zeros if eos.json is absent or pressure is zero.
    """
    if abs(pressure_GPa) < 1e-6:
        return np.zeros_like(T_array)

    path = os.path.join(sym_dir, "equation_of_state", "eos.json")
    if not os.path.exists(path):
        return np.zeros_like(T_array)

    with open(path) as fh:
        d = json.load(fh)

    volumes  = np.array(d["volumes_A3"],  dtype=float)   # per atom
    energies = np.array(d["energies_eV"], dtype=float)   # per atom

    # Fit Birch–Murnaghan EOS to get V₀, B₀.
    try:
        V0, E0_eos, B0, B0p = _fit_bm_eos(volumes, energies)
    except Exception:
        return np.zeros_like(T_array)

    # Convert pressure: GPa → eV/Å³  (1 GPa = 0.006242 eV/Å³)
    P_eV_A3 = pressure_GPa * 0.006241509125883258

    # PV work at equilibrium volume (temperature-independent at this level).
    pv = P_eV_A3 * V0
    return np.full_like(T_array, pv, dtype=float)


def _fit_bm_eos(
        volumes: np.ndarray,
        energies: np.ndarray,
) -> tuple[float, float, float, float]:
    """Fit third-order Birch–Murnaghan EOS, return (V0, E0, B0, B0p)."""
    from scipy.optimize import curve_fit

    def bm3(V, V0, E0, B0, B0p):
        eta = (V0 / V) ** (2.0 / 3.0)
        return (E0
                + 9.0 * V0 * B0 / 16.0
                * ((eta - 1.0)**3 * B0p
                   + (eta - 1.0)**2 * (6.0 - 4.0 * eta)))

    V_mid = float(volumes[np.argmin(energies)])
    p0 = [V_mid, float(energies.min()), 1.0, 4.0]
    popt, _ = curve_fit(bm3, volumes, energies, p0=p0, maxfev=5000)
    return tuple(popt)


# =============================================================================
# Free-energy registry — add new contributions here
# =============================================================================

class GibbsAssembler:
    """Assembles the total Gibbs free energy from registered contributions.

    Usage::

        from heatup.free_energy import GibbsAssembler, anharmonic_f_vib

        asm = GibbsAssembler()
        asm.register("vib_anh", anharmonic_f_vib, weight=1.0)
        # asm.register("el",     electronic_f_el,  weight=1.0)

        result = asm.compute(sym_dir, temperatures)
        # result["G_eV_per_atom"] → array [eV/atom]

    Each registered contribution must have signature:
        fn(sym_dir: str, T_array: np.ndarray) → np.ndarray  [eV/atom]

    The ``weight`` parameter allows partial inclusion (e.g. weight=0.5 for
    a contribution that is double-counted elsewhere).
    """

    def __init__(self) -> None:
        self._contributions: list[tuple[str, Contribution, float]] = []

    def register(
            self,
            name: str,
            fn: Contribution,
            weight: float = 1.0,
    ) -> "GibbsAssembler":
        """Register a free-energy contribution.

        Args:
            name:   Human-readable label (used in output dict).
            fn:     Callable (sym_dir, T_array) → np.ndarray [eV/atom].
            weight: Linear weight applied to the contribution (default 1).

        Returns:
            self (for chaining).
        """
        self._contributions.append((name, fn, weight))
        return self

    def compute(
            self,
            sym_dir: str,
            temperatures: Sequence[float],
            pressure_GPa: float = 0.0,
    ) -> dict:
        """Compute G(T) = E₀ + Σ weight_i · contribution_i(T).

        Args:
            sym_dir:       Symmetry directory path.
            temperatures:  Temperature grid (K).
            pressure_GPa:  External pressure in GPa (passed to PV contribution).

        Returns:
            Dict with keys:
                ``'E0_eV_per_atom'``      Ground-state energy per atom.
                ``'temperatures'``         Temperature grid.
                ``'G_eV_per_atom'``        Total Gibbs free energy per atom.
                ``'F_eV_per_atom'``        Alias for G (for hull compatibility).
                ``'contributions'``        Dict name → array of contribution values.
                ``'available_contributions'``  List of names with non-zero data.
        """
        T_array = np.array(temperatures, dtype=float)

        e0_val = e0(sym_dir)
        if e0_val is None:
            return {
                "E0_eV_per_atom"         : None,
                "temperatures"           : list(temperatures),
                "G_eV_per_atom"          : None,
                "F_eV_per_atom"          : None,
                "contributions"          : {},
                "available_contributions": [],
            }

        G = np.full_like(T_array, e0_val)
        contrib_values: dict[str, list[float]] = {}
        available: list[str] = []

        for name, fn, weight in self._contributions:
            try:
                vals = weight * fn(sym_dir, T_array)
                contrib_values[name] = vals.tolist()
                if np.any(np.abs(vals) > 1e-10):
                    available.append(name)
                G += vals
            except Exception as exc:
                import warnings as _warnings
                _warnings.warn(
                    f"GibbsAssembler: contribution '{name}' failed for "
                    f"{sym_dir}: {exc}",
                    stacklevel=2,
                )
                contrib_values[name] = np.zeros_like(T_array).tolist()

        return {
            "E0_eV_per_atom"         : e0_val,
            "temperatures"           : T_array.tolist(),
            "G_eV_per_atom"          : G.tolist(),
            "F_eV_per_atom"          : G.tolist(),    # hull-compatible alias
            "contributions"          : contrib_values,
            "available_contributions": available,
        }


# =============================================================================
# Default assembler — used by the pipeline unless overridden
# =============================================================================

def build_default_assembler(
        phonon_mode: str = "anharmonic",
        include_electronic: bool = True,
        include_magnetic: bool = True,
        include_configurational: bool = True,
        include_pv: bool = False,
        device: str = config.DEFAULT_DEVICE,
) -> GibbsAssembler:
    """Build the default GibbsAssembler with all available contributions.

    The assembler is configured by the flags below.  Contributions for which
    data are absent are registered but return zeros silently.

    Args:
        phonon_mode:              ``'anharmonic'`` (AIMD VACF/VDOS, default)
                                  or ``'harmonic'`` (finite-displacement DOS).
        include_electronic:       Include F_el(T) from electronic DOS.
        include_magnetic:         Include F_mag(T) from magnetic moments.
        include_configurational:  Include F_conf(T) from site occupancies.
        include_pv:               Include PV term (requires eos.json).
        device:                   Compute device for AIMD triggering.

    Returns:
        Configured :class:`GibbsAssembler`.
    """
    import functools

    asm = GibbsAssembler()

    # Vibrational.
    if phonon_mode == "anharmonic":
        vib_fn = functools.partial(anharmonic_f_vib, device=device)
        asm.register("F_vib_anharmonic", vib_fn)
    else:
        asm.register("F_vib_harmonic", harmonic_f_vib)

    # Electronic.
    if include_electronic:
        asm.register("F_el", electronic_f_el)

    # Magnetic.
    if include_magnetic:
        asm.register("F_mag", magnetic_f_mag)

    # Configurational.
    if include_configurational:
        asm.register("F_conf", configurational_f_conf)

    # PV.
    if include_pv:
        pv_fn = functools.partial(pv_contribution, pressure_GPa=config.HULL_PRESSURE_GPa)
        asm.register("PV", pv_fn)

    return asm
