"""Anharmonic phonon analysis from AIMD trajectories.

Extracts the vibrational density of states (VDOS) from AIMD trajectories via
the velocity autocorrelation function (VACF), then computes temperature-dependent
thermodynamic quantities in the quasi-harmonic approximation.

This module is the anharmonic counterpart of the harmonic phonon DOS stored in
``phonons/dos.json``.  Results are written to::

    database/<material>/<symmetry>/aimd/<T>K/anharmonic_phonons/
        vdos.json          ← VDOS (omega in meV, g(omega) normalised)
        thermo.json        ← E_vib, F_vib, Cv, S_vib, nu_mean at the MD temperature
        free_energy.json   ← F(T) curve over a requested temperature grid (eV/atom)

so they are computed once and reused in subsequent stability analyses.

Physics
-------
Velocities are estimated from finite differences of fractional coordinates
(PBC-aware, minimum-image convention).  The VACF is computed via
``numpy.correlate(..., 'full')``, normalised by the mean kinetic energy,
then Fourier-transformed to give the VDOS.  Thermodynamic integrals follow
the quantum harmonic oscillator::

    E_vib(T) = ∫ g(ω) ω [½ + n(ω,T)] dω          [meV/atom]
    F_vib(T) = ∫ g(ω) [½ω + kT ln(1-e^{-ω/kT})] dω  [meV/atom → converted to eV]
    Cv(T)    = ∫ g(ω) kB (ω/kT)² e^{ω/kT}/(e^{ω/kT}-1)² dω
    S_vib(T) = ∫ g(ω) [ω/(2T tanh(ω/2kT)) - kB ln(2 sinh(ω/2kT))] dω

Units throughout this module
-----------------------------
- omega (frequency axis): **meV**  (matches the user's provided snippet)
- E_vib, F_vib: **meV/atom** internally, converted to **eV/atom** before
  writing ``free_energy.json`` for use in the convex-hull code.
- Cv, S_vib: **meV/(K·atom)**
- nu_mean: **meV** (mean vibrational frequency)

Integration with thermodynamic_stability
-----------------------------------------
Use ``get_anharmonic_free_energy(sym_dir, temperatures)`` as a drop-in
replacement for ``_compute_free_energies`` when ``phonon_source='anharmonic'``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import warnings
from typing import Sequence

import numpy as np
from scipy.fft import fft, fftfreq


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

#: Boltzmann constant in meV/K.
_KB_MEV: float = 8.617333262145e-2

#: Boltzmann constant in eV/K (for consistency with thermodynamic_stability).
_KB_EV: float = 8.617333262e-5

#: Conversion: meV → eV.
_MEV_TO_EV: float = 1e-3

#: Minimum physical frequency in meV.  Modes below this (acoustic/numerical
#: noise at Γ) are excluded from thermodynamic integrals.
OMEGA_MIN_MEV: float = 0.1   # ~0.1 meV ≈ 24 GHz

#: Fractional free-energy difference above which a warning is raised when
#: averaging over multiple MD temperatures (e.g. 0.05 = 5 %).
FREE_ENERGY_CONSISTENCY_THRESHOLD: float = 0.05


# ---------------------------------------------------------------------------
# Core thermodynamic kernels  (all omega in meV, T in K)
# ---------------------------------------------------------------------------

def _norm_trapz(x: np.ndarray, fx: np.ndarray, rho: np.ndarray) -> float:
    """Weighted average: ∫ fx·rho dx / ∫ rho dx  (trapezoidal)."""
    denom = np.trapezoid(rho, x)
    if denom == 0.0:
        return 0.0
    return float(np.trapezoid(fx * rho, x) / denom)


def _phonon_energy(omega: np.ndarray, T: float) -> np.ndarray:
    """Mean phonon energy ω·[½ + n(ω,T)] in meV."""
    if T <= 0.0:
        return 0.5 * omega
    x = omega / (_KB_MEV * T)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        n = np.where(x > 500, 0.0, 1.0 / (np.expm1(np.clip(x, 0, 500))))
    return omega * (0.5 + n)


def _heat_capacity(omega: np.ndarray, T: float) -> np.ndarray:
    """Cv kernel: kB (ω/kT)² e^x / (e^x-1)² in meV/(K·atom)."""
    if T <= 0.0:
        return np.zeros_like(omega)
    x = np.clip(omega / (_KB_MEV * T), 1e-12, 500.0)
    ex = np.exp(x)
    return _KB_MEV * x**2 * ex / (ex - 1.0)**2


def _helmholtz(omega: np.ndarray, T: float) -> np.ndarray:
    """F_vib kernel: ½ω + kT ln(1-e^{-ω/kT}) in meV."""
    if T <= 0.0:
        return 0.5 * omega
    x = np.clip(omega / (_KB_MEV * T), 1e-12, 500.0)
    return 0.5 * omega + _KB_MEV * T * np.log1p(-np.exp(-x))


def _entropy(omega: np.ndarray, T: float) -> np.ndarray:
    """S_vib kernel: ω/(2T tanh(ω/2kT)) - kB ln(2 sinh(ω/2kT)) in meV/(K·atom)."""
    if T <= 0.0:
        return np.zeros_like(omega)
    x = np.clip(omega / (_KB_MEV * T), 1e-12, 500.0)
    return omega / (2.0 * T * np.tanh(0.5 * x)) - _KB_MEV * np.log(2.0 * np.sinh(0.5 * x))


# ---------------------------------------------------------------------------
# VACF → VDOS
# ---------------------------------------------------------------------------

def compute_vdos_from_traj(
        traj: list,
        traj_timestep_fs: float,
        temperature_K: float,
) -> dict:
    """Compute VDOS and vibrational thermodynamics from an ASE trajectory.

    Velocities are estimated from PBC-aware finite differences of fractional
    coordinates (minimum-image convention), then converted to Cartesian Å/fs.

    The VACF is computed via ``numpy.correlate(..., 'full')``, normalised by
    the total kinetic energy proxy (sum of squared velocities), and
    Fourier-transformed to give the one-sided VDOS.

    Args:
        traj:             List of ASE Atoms objects (production frames only —
                          equilibration must be stripped by the caller).
        traj_timestep_fs: Effective time between consecutive frames in fs.
                          Equal to ``TIMESTEP_FS * NBLOCK * STEP_SKIP``.
        temperature_K:    MD temperature used for thermodynamic integrals.

    Returns:
        Dict with keys:

        ``'omega_mev'``   (list[float]) — frequency axis in meV, positive half only.
        ``'vdos'``        (list[float]) — normalised VDOS g(ω), ∫g dω = 1.
        ``'nu_mean_mev'`` (float)       — mean vibrational frequency (meV).
        ``'E_vib_mev_atom'`` (float)    — mean phonon energy at T (meV/atom).
        ``'F_vib_mev_atom'`` (float)    — Helmholtz free energy at T (meV/atom).
        ``'Cv_mev_Katom'``  (float)     — heat capacity at T (meV/(K·atom)).
        ``'S_vib_mev_Katom'``(float)    — vibrational entropy at T (meV/(K·atom)).
        ``'n_frames'``    (int)         — number of trajectory frames used.
        ``'n_atoms'``     (int)         — number of atoms per frame.
        ``'temperature_K'`` (float)     — MD temperature (recorded for reference).
    """
    if len(traj) < 4:
        raise ValueError(
            f'Trajectory has only {len(traj)} frames — need at least 4 for VACF.'
        )

    # ── Velocities from PBC-aware finite differences ──────────────────────
    # pos: (n_frames, n_atoms, 3) in fractional coordinates
    pos  = np.array([atoms.get_scaled_positions() for atoms in traj])   # (T, N, 3)
    dpos = np.diff(pos, axis=0)                                          # (T-1, N, 3)
    dpos -= np.round(dpos)                                               # minimum image

    # Convert to Cartesian Å using the cell of the first frame
    cell = traj[0].get_cell()[:]   # (3, 3)  row vectors
    # dpos @ cell: each displacement vector (frac) → Cartesian Å
    dpos_cart = dpos @ cell                                              # (T-1, N, 3)

    # Duplicate the last frame to keep shape (T, N, 3) — consistent with the
    # user's original snippet (appends last velocity twice).
    vel = np.concatenate(
        [dpos_cart / traj_timestep_fs,
         dpos_cart[-1:] / traj_timestep_fs],
        axis=0,
    )                                                                    # (T, N, 3)

    n_frames, n_atoms, _ = vel.shape

    # ── VACF via np.correlate 'full' ──────────────────────────────────────
    # Frequency axis for the full (2T-1)-point correlation output.
    # fftfreq gives cycles/fs; multiply by 1e3 * h/eV to get meV.
    # h = 4.135667696e-15 eV·s = 4.135667696e-3 eV·fs → in meV: 4.135667696
    omega_full = fftfreq(2 * n_frames - 1, traj_timestep_fs) * 1e3 * 4.135667696
    omega_pos  = omega_full[:n_frames]   # positive-frequency half

    corr = np.zeros(2 * n_frames - 1)
    for i in range(n_atoms):
        for j in range(3):
            corr += np.correlate(vel[:, i, j], vel[:, i, j], 'full')

    # Normalise by total kinetic energy proxy (sum of v²)
    norm_factor = np.sum(vel ** 2)
    if norm_factor == 0.0:
        raise ValueError('All velocities are zero — trajectory may be corrupt.')
    VACF = corr / norm_factor

    # ── FFT → VDOS ────────────────────────────────────────────────────────
    pdos  = np.abs(fft(VACF - np.mean(VACF)))
    vdos_full = pdos[:n_frames]

    # ── Strip non-physical modes ──────────────────────────────────────────
    mask  = omega_pos >= OMEGA_MIN_MEV
    om    = omega_pos[mask]
    g     = vdos_full[mask]

    if len(om) < 2:
        raise ValueError(
            'No physical modes found above OMEGA_MIN_MEV '
            f'({OMEGA_MIN_MEV} meV). Check trajectory and timestep.'
        )

    # Normalise so ∫g dω = 1
    norm = np.trapezoid(g, om)
    if norm <= 0.0:
        raise ValueError('VDOS integral is zero or negative — cannot normalise.')
    g = g / norm

    # ── Thermodynamic integrals ───────────────────────────────────────────
    nu_mean      = _norm_trapz(om, om,                              g)
    E_vib        = _norm_trapz(om, _phonon_energy(om, temperature_K), g)
    F_vib        = _norm_trapz(om, _helmholtz    (om, temperature_K), g)
    Cv           = _norm_trapz(om, _heat_capacity(om, temperature_K), g)
    S_vib        = _norm_trapz(om, _entropy      (om, temperature_K), g)

    return {
        'omega_mev'       : om.tolist(),
        'vdos'            : g.tolist(),
        'nu_mean_mev'     : float(nu_mean),
        'E_vib_mev_atom'  : float(E_vib),
        'F_vib_mev_atom'  : float(F_vib),
        'Cv_mev_Katom'    : float(Cv),
        'S_vib_mev_Katom' : float(S_vib),
        'n_frames'        : n_frames,
        'n_atoms'         : n_atoms,
        'temperature_K'   : float(temperature_K),
    }


# ---------------------------------------------------------------------------
# Free-energy curve over a temperature grid
# ---------------------------------------------------------------------------

def _f_vib_curve_from_vdos(
        omega_mev: np.ndarray,
        vdos: np.ndarray,
        temperatures: Sequence[float],
        e0_eV_per_atom: float,
) -> dict:
    """Compute F(T) = E0 + F_vib(T) over a temperature grid.

    F_vib is integrated from the anharmonic VDOS and returned in eV/atom
    so it is directly compatible with ``thermodynamic_stability._compute_free_energies``.

    Args:
        omega_mev:       Frequency axis in meV (physical modes only, normalised).
        vdos:            Normalised VDOS g(ω), ∫g dω = 1.
        temperatures:    Temperature grid in K.
        e0_eV_per_atom:  MACE ground-state energy per atom in eV.

    Returns:
        Dict with ``'temperatures'``, ``'F_eV_per_atom'``, ``'E0_eV_per_atom'``.
    """
    Fs = []
    for T in temperatures:
        if T <= 0.0:
            # Zero-point energy: ½ ∫ g(ω) ω dω  [meV/atom] → eV/atom
            zpe_mev = float(np.trapezoid(0.5 * vdos * omega_mev, omega_mev))
            Fs.append(e0_eV_per_atom + zpe_mev * _MEV_TO_EV)
        else:
            f_mev = _norm_trapz(omega_mev, _helmholtz(omega_mev, T), vdos)
            Fs.append(e0_eV_per_atom + f_mev * _MEV_TO_EV)
    return {
        'E0_eV_per_atom': e0_eV_per_atom,
        'temperatures'  : list(temperatures),
        'F_eV_per_atom' : Fs,
    }


# ---------------------------------------------------------------------------
# Cache I/O — per sim_dir
# ---------------------------------------------------------------------------

def _anharmonic_dir(sim_dir: str) -> str:
    """Return path to the anharmonic_phonons cache folder inside sim_dir."""
    return os.path.join(sim_dir, 'anharmonic_phonons')


def _load_cached_vdos(sim_dir: str) -> dict | None:
    """Load cached VDOS from sim_dir/anharmonic_phonons/vdos.json, or None."""
    path = os.path.join(_anharmonic_dir(sim_dir), 'vdos.json')
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def _load_cached_free_energy(sim_dir: str) -> dict | None:
    """Load cached F(T) curve from anharmonic_phonons/free_energy.json, or None."""
    path = os.path.join(_anharmonic_dir(sim_dir), 'free_energy.json')
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def _save_anharmonic_cache(
        sim_dir: str,
        vdos_result: dict,
        free_energy: dict,
) -> None:
    """Write VDOS and free-energy results to the anharmonic_phonons cache."""
    anh_dir = _anharmonic_dir(sim_dir)
    os.makedirs(anh_dir, exist_ok=True)

    # VDOS + thermo at MD temperature
    thermo = {
        'nu_mean_mev'    : vdos_result['nu_mean_mev'],
        'E_vib_mev_atom' : vdos_result['E_vib_mev_atom'],
        'F_vib_mev_atom' : vdos_result['F_vib_mev_atom'],
        'Cv_mev_Katom'   : vdos_result['Cv_mev_Katom'],
        'S_vib_mev_Katom': vdos_result['S_vib_mev_Katom'],
        'n_frames'       : vdos_result['n_frames'],
        'n_atoms'        : vdos_result['n_atoms'],
        'temperature_K'  : vdos_result['temperature_K'],
    }

    vdos_out = {
        'omega_mev': vdos_result['omega_mev'],
        'vdos'     : vdos_result['vdos'],
    }

    with open(os.path.join(anh_dir, 'vdos.json'),        'w') as fh:
        json.dump(vdos_out,    fh, indent=4)
    with open(os.path.join(anh_dir, 'thermo.json'),      'w') as fh:
        json.dump(thermo,      fh, indent=4)
    with open(os.path.join(anh_dir, 'free_energy.json'), 'w') as fh:
        json.dump(free_energy, fh, indent=4)


# ---------------------------------------------------------------------------
# Per-simulation VDOS computation with caching
# ---------------------------------------------------------------------------

def compute_anharmonic_phonons_for_sim(
        sim_dir: str,
        temperatures: Sequence[float],
        force_recompute: bool = False,
) -> dict | None:
    """Compute (or load cached) anharmonic VDOS and F(T) for one sim_dir.

    Reads the trajectory from ``sim_dir/output.traj`` and
    ``sim_dir/simulation-input.json``, strips equilibration frames, then
    calls :func:`compute_vdos_from_traj`.  Results are cached in
    ``sim_dir/anharmonic_phonons/``.

    Args:
        sim_dir:         Path to a temperature sub-folder,
                         e.g. ``database/AgI/P63mc/aimd/1200K``.
        temperatures:    Temperature grid for the F(T) curve (K).
        force_recompute: Ignore cached results and recompute.

    Returns:
        Free-energy dict ``{'E0_eV_per_atom', 'temperatures', 'F_eV_per_atom'}``
        compatible with :func:`thermodynamic_stability._compute_free_energies`,
        or ``None`` if the trajectory is unavailable or analysis fails.
    """
    # ── Check for valid cached result ─────────────────────────────────────
    if not force_recompute:
        cached_fe = _load_cached_free_energy(sim_dir)
        if cached_fe is not None:
            # Check the cached temperature grid covers what we need.
            cached_Ts = set(cached_fe.get('temperatures', []))
            requested_Ts = set(float(t) for t in temperatures)
            if requested_Ts.issubset(cached_Ts):
                return cached_fe
            # Grid mismatch — recompute free-energy curve from cached VDOS.
            cached_vdos = _load_cached_vdos(sim_dir)
            if cached_vdos is not None:
                e0 = float(cached_fe['E0_eV_per_atom'])
                om = np.array(cached_vdos['omega_mev'])
                g  = np.array(cached_vdos['vdos'])
                fe = _f_vib_curve_from_vdos(om, g, temperatures, e0)
                # Update cached free_energy.json with the new grid
                anh_dir = _anharmonic_dir(sim_dir)
                os.makedirs(anh_dir, exist_ok=True)
                with open(os.path.join(anh_dir, 'free_energy.json'), 'w') as fh:
                    json.dump(fe, fh, indent=4)
                return fe

    # ── Load trajectory ───────────────────────────────────────────────────
    input_path = os.path.join(sim_dir, 'simulation-input.json')
    traj_path  = os.path.join(sim_dir, 'output.traj')

    if not os.path.exists(traj_path):
        print(f'  [anharmonic] {sim_dir}: output.traj not found — skipping.')
        return None
    if os.path.getsize(traj_path) == 0:
        print(f'  [anharmonic] {sim_dir}: output.traj is empty — skipping.')
        return None
    if not os.path.exists(input_path):
        print(f'  [anharmonic] {sim_dir}: simulation-input.json not found — skipping.')
        return None

    with open(input_path) as fh:
        params = json.load(fh)

    timestep_fs      = float(params['timestep_fs'])
    nblock           = int(params['nblock'])
    step_skip        = int(params.get('step_skip', 1))
    step_equiv       = int(params.get('step_equiv', 100))
    temperature_K    = float(params['temperature'])
    traj_timestep_fs = timestep_fs * nblock * step_skip

    # ── Load E0 from relaxation ───────────────────────────────────────────
    # Walk up from sim_dir (aimd/<T>K) → symmetry_dir → relaxation/energy.json
    sym_dir      = os.path.dirname(os.path.dirname(sim_dir))  # up two levels
    energy_path  = os.path.join(sym_dir, 'relaxation', 'energy.json')
    if not os.path.exists(energy_path):
        print(f'  [anharmonic] {sim_dir}: relaxation/energy.json not found — skipping.')
        return None
    with open(energy_path) as fh:
        e0_eV_per_atom = float(json.load(fh)['energy_eV_per_atom'])

    # ── Read & strip equilibration ────────────────────────────────────────
    try:
        from ase.io import read as ase_read
        full_traj = ase_read(traj_path, index=f'::{step_skip}')
    except Exception as exc:
        print(f'  [anharmonic] {sim_dir}: could not read trajectory: {exc}')
        return None

    prod_traj = full_traj[step_equiv:]
    if len(prod_traj) < 4:
        print(f'  [anharmonic] {sim_dir}: only {len(prod_traj)} production frames — '
              f'need at least 4.')
        return None

    # ── Compute VDOS ─────────────────────────────────────────────────────
    print(f'  [anharmonic] Computing VDOS from {len(prod_traj)} frames '
          f'({sim_dir}) ...')
    try:
        vdos_result = compute_vdos_from_traj(prod_traj, traj_timestep_fs, temperature_K)
    except Exception as exc:
        print(f'  [anharmonic] {sim_dir}: VDOS computation failed: {exc}')
        return None

    # ── Build F(T) curve ─────────────────────────────────────────────────
    om = np.array(vdos_result['omega_mev'])
    g  = np.array(vdos_result['vdos'])
    fe = _f_vib_curve_from_vdos(om, g, temperatures, e0_eV_per_atom)

    # ── Cache results ─────────────────────────────────────────────────────
    _save_anharmonic_cache(sim_dir, vdos_result, fe)
    print(f'    ✓ Cached → {_anharmonic_dir(sim_dir)}')

    return fe


# ---------------------------------------------------------------------------
# Multi-temperature averaging — main entry point for stability pipeline
# ---------------------------------------------------------------------------

def get_anharmonic_free_energy(
        sym_dir: str,
        temperatures: Sequence[float],
        consistency_threshold: float = FREE_ENERGY_CONSISTENCY_THRESHOLD,
        force_recompute: bool = False,
        device: str = 'cuda',
        run_aimd_if_missing: bool = True,
        aimd_temperature: float | None = None,
) -> dict | None:
    """Compute anharmonic free energies for a symmetry directory.

    This is the top-level entry point for the stability pipeline.  It:

    1. Scans ``sym_dir/aimd/`` for completed temperature sub-folders.
    2. For each found simulation, computes or loads the anharmonic VDOS and
       F(T) curve (cached in ``aimd/<T>K/anharmonic_phonons/``).
    3. **Averages** F(T) across all available MD temperatures at each
       requested temperature (phonons are expected to be similar across T).
    4. **Warns** when the spread across MD temperatures exceeds
       ``consistency_threshold`` at any point on the grid.
    5. If no AIMD simulation exists, optionally triggers a new one at
       ``aimd_temperature`` (defaults to the median of ``temperatures``)
       using ``run_single_md.py`` in a subprocess.

    Args:
        sym_dir:              Path to ``database/<material>/<symmetry>/``.
        temperatures:         Temperature grid for F(T) in K.
        consistency_threshold: Fractional F_vib spread that triggers a warning.
        force_recompute:      Ignore caches and recompute all VDOS.
        device:               Compute device for MACE if AIMD must be triggered.
        run_aimd_if_missing:  If True and no AIMD exists, run a new simulation.
        aimd_temperature:     MD temperature to use when no AIMD exists.  Defaults
                              to the median of the non-zero requested temperatures.

    Returns:
        Dict with ``'E0_eV_per_atom'``, ``'temperatures'``, ``'F_eV_per_atom'``
        (averaged over all available MD simulations), or ``None`` on failure.
        Also contains ``'anharmonic_sources'`` (list of sim_dirs used) and
        ``'consistency_warning'`` (bool).
    """
    temperatures = list(temperatures)
    aimd_dir     = os.path.join(sym_dir, 'aimd')

    # ── Discover completed sim_dirs ───────────────────────────────────────
    sim_dirs = _find_completed_sim_dirs(aimd_dir)

    # ── No AIMD available — optionally trigger one ────────────────────────
    if not sim_dirs:
        if not run_aimd_if_missing:
            print(f'  [anharmonic] No AIMD found for {sym_dir} and '
                  f'run_aimd_if_missing=False — cannot compute anharmonic phonons.')
            return None

        T_md = aimd_temperature
        if T_md is None:
            nonzero = [t for t in temperatures if t > 0]
            T_md = float(np.median(nonzero)) if nonzero else 300.0

        print(f'  [anharmonic] No AIMD found for {sym_dir}.')
        print(f'  [anharmonic] Triggering MD at {T_md:.0f} K ...')
        ok = _trigger_aimd(sym_dir, T_md, device=device)
        if not ok:
            print(f'  [anharmonic] MD failed — cannot compute anharmonic phonons.')
            return None

        sim_dirs = _find_completed_sim_dirs(aimd_dir)
        if not sim_dirs:
            print(f'  [anharmonic] MD completed but no valid sim_dir found.')
            return None

    # ── Compute VDOS + F(T) per sim_dir ──────────────────────────────────
    fe_per_sim: list[dict] = []
    sources: list[str] = []

    for sd in sim_dirs:
        fe = compute_anharmonic_phonons_for_sim(
            sd, temperatures, force_recompute=force_recompute,
        )
        if fe is not None:
            fe_per_sim.append(fe)
            sources.append(sd)

    if not fe_per_sim:
        print(f'  [anharmonic] No valid VDOS could be computed for {sym_dir}.')
        return None

    # ── Average F(T) across simulations ──────────────────────────────────
    # All fe dicts share the same E0 (same relaxed structure), so use the
    # mean E0 across the set (they should be identical, but guard anyway).
    e0_values  = [fe['E0_eV_per_atom'] for fe in fe_per_sim]
    e0_mean    = float(np.mean(e0_values))

    # Interpolate each simulation's F curve onto the requested temperature grid.
    F_matrix = np.zeros((len(fe_per_sim), len(temperatures)))
    for i, fe in enumerate(fe_per_sim):
        Ts_i = np.array(fe['temperatures'])
        Fs_i = np.array(fe['F_eV_per_atom'])
        for j, T in enumerate(temperatures):
            F_matrix[i, j] = float(np.interp(T, Ts_i, Fs_i))

    F_mean = F_matrix.mean(axis=0)
    F_std  = F_matrix.std(axis=0)

    # ── Consistency check ─────────────────────────────────────────────────
    consistency_warning = False
    if len(fe_per_sim) > 1:
        # Fractional spread relative to |F_mean| at each T (skip T=0 / F≈0)
        nonzero_mask = np.abs(F_mean) > 1e-10
        if nonzero_mask.any():
            rel_spread = F_std[nonzero_mask] / np.abs(F_mean[nonzero_mask])
            max_spread = float(rel_spread.max())
            worst_T    = float(np.array(temperatures)[nonzero_mask][np.argmax(rel_spread)])
            if max_spread > consistency_threshold:
                consistency_warning = True
                print(
                    f'\n  ⚠ [anharmonic] FREE ENERGY CONSISTENCY WARNING for {sym_dir}:\n'
                    f'    Max fractional spread across {len(fe_per_sim)} MD temperatures '
                    f'= {max_spread * 100:.1f}% (at T={worst_T:.0f} K).\n'
                    f'    Threshold = {consistency_threshold * 100:.0f}%.\n'
                    f'    Sources: {[os.path.basename(s) for s in sources]}\n'
                    f'    Review the MD trajectories for structural phase transitions.\n'
                )

    if len(fe_per_sim) > 1:
        print(f'  [anharmonic] Averaged F(T) over {len(fe_per_sim)} MD simulation(s): '
              f'{[os.path.basename(s) for s in sources]}')

    return {
        'E0_eV_per_atom'      : e0_mean,
        'temperatures'        : temperatures,
        'F_eV_per_atom'       : F_mean.tolist(),
        'anharmonic_sources'  : sources,
        'n_sources'           : len(sources),
        'consistency_warning' : consistency_warning,
        'F_std_eV_per_atom'   : F_std.tolist(),
    }


# ---------------------------------------------------------------------------
# Helper: find completed sim_dirs
# ---------------------------------------------------------------------------

def _find_completed_sim_dirs(aimd_dir: str) -> list[str]:
    """Return all sim_dirs under aimd_dir that have a non-empty output.traj."""
    results = []
    if not os.path.isdir(aimd_dir):
        return results
    for temp_folder in sorted(os.listdir(aimd_dir)):
        if not temp_folder.endswith('K'):
            continue
        sd        = os.path.join(aimd_dir, temp_folder)
        traj_path = os.path.join(sd, 'output.traj')
        if (os.path.isdir(sd)
                and os.path.exists(traj_path)
                and os.path.getsize(traj_path) > 0):
            results.append(sd)
    return results


# ---------------------------------------------------------------------------
# Helper: trigger a new AIMD simulation via run_single_md.py subprocess
# ---------------------------------------------------------------------------

def _trigger_aimd(sym_dir: str, temperature: float, device: str = 'cuda') -> bool:
    """Run an NPT MD simulation in a subprocess for CUDA isolation."""
    from heatup.structure_utils import run_md_subprocess
    return run_md_subprocess(sym_dir, temperature, device=device)
