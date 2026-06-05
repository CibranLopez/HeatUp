"""heatup.structural_transition
================================
Gate 0: Structural transition detection from AIMD trajectories.

Detects whether an MD simulation has undergone a structural phase transition
by computing the displacement autocorrelation function (DACF)::

    C(τ) = (1 / (N·3)) · Σᵢ Σ_α ⟨Δuᵢₐ(t) · Δuᵢₐ(t + τ)⟩_t   [Å²]

where Δuᵢₐ(t) = rᵢₐ(t) − rᵢₐ(0) is the PBC-aware cumulative displacement
of atom i in direction α from its initial (first production frame) position.

For a thermally vibrating solid, atoms remain near their equilibrium sites:
Δu(t) is bounded and oscillatory → C(τ) → 0 as τ → ∞.

After a structural phase transition, atoms settle at new equilibrium
positions: Δu(t) → constant ≠ 0 → C(τ) plateaus at C_∞ > 0.

The long-time value C_∞ is estimated as the mean of C(τ) over the last
``config.DACF_LONG_TIME_FRAC`` fraction of the computed lag range (0 … T//2).
A structural transition is declared if C_∞ > ``config.DACF_THRESHOLD_ANG2``.

Results are cached in::

    <sim_dir>/anharmonic_phonons/dacf.json

so they are computed once and reused in subsequent pipeline runs.
"""

from __future__ import annotations

import json
import os

import numpy as np

from heatup import config


# ---------------------------------------------------------------------------
# Core DACF computation
# ---------------------------------------------------------------------------

def compute_dacf_from_traj(
        traj: list,
        traj_timestep_fs: float,
) -> dict:
    """Compute the displacement autocorrelation function from a trajectory.

    Velocities are *not* used here — only positions.  PBC-aware cumulative
    displacements from the first frame are computed via the minimum-image
    convention on finite differences of fractional coordinates.

    The DACF is evaluated via FFT-based linear autocorrelation (zero-padded
    to avoid circular aliasing) with the unbiased denominator (n_frames − τ).
    Lags are limited to T//2 so that the unbiased estimator remains reliable.

    Args:
        traj:             List of ASE Atoms objects (production frames only —
                          equilibration must be stripped by the caller).
        traj_timestep_fs: Effective time between consecutive frames in fs
                          (= ``timestep_fs × nblock × step_skip``).

    Returns:
        Dict with keys:

        ``'dacf'`` (list[float])
            C(τ) in Å², length T//2, from τ=0 to τ=(T//2−1)·dt.
        ``'lag_fs'`` (list[float])
            Lag times in femtoseconds, one value per DACF entry.
        ``'c_inf_ang2'`` (float)
            Long-time plateau C_∞ (Å²), averaged over the last
            ``config.DACF_LONG_TIME_FRAC`` of the lag range.
        ``'transitioned'`` (bool)
            True if C_∞ > ``config.DACF_THRESHOLD_ANG2``.
        ``'n_frames'`` (int)
            Number of trajectory frames used.
        ``'n_atoms'`` (int)
            Number of atoms per frame.
    """
    if len(traj) < 4:
        raise ValueError(
            f'Trajectory has only {len(traj)} frames — need at least 4 for DACF.'
        )

    n_frames = len(traj)
    n_atoms  = len(traj[0])

    # ── PBC-aware cumulative displacements (fractional → Cartesian Å) ──────
    pos  = np.array([a.get_scaled_positions() for a in traj])  # (T, N, 3)
    dpos = np.diff(pos, axis=0)                                 # (T-1, N, 3)
    dpos -= np.round(dpos)                                      # minimum image
    u_frac = np.concatenate(
        [np.zeros((1, n_atoms, 3)), np.cumsum(dpos, axis=0)],
        axis=0,
    )  # (T, N, 3) — cumulative displacement from frame 0 in fractional coords

    cell   = traj[0].get_cell()[:]  # (3, 3) row vectors
    u_cart = u_frac @ cell          # (T, N, 3)  in Å

    # ── FFT-based autocorrelation with unbiased estimator ──────────────────
    # Zero-padding to 2·T ensures the circular FFT gives the linear
    # autocorrelation: IFFT(|FFT(x, n=2T)|²)[τ] = Σ_{t=0}^{T-τ-1} x[t]·x[t+τ].
    # Lags are limited to T//2 for reliable unbiased division.
    max_lag  = n_frames // 2
    u_flat   = u_cart.reshape(n_frames, -1)  # (T, N*3)
    n_fft    = 2 * n_frames                  # zero-pad length
    dacf_sum = np.zeros(max_lag)

    for k in range(u_flat.shape[1]):
        X        = np.fft.rfft(u_flat[:, k], n=n_fft)
        autocorr = np.fft.irfft(X * np.conj(X), n=n_fft)[:max_lag]
        dacf_sum += np.real(autocorr)

    # Unbiased denominator: at lag τ, there are (n_frames − τ) pairs.
    counts = np.arange(n_frames, n_frames - max_lag, -1, dtype=float)
    dacf   = dacf_sum / counts / u_flat.shape[1]  # average over N*3 DOF

    # ── Long-time plateau ───────────────────────────────────────────────────
    long_start   = max(1, int((1.0 - config.DACF_LONG_TIME_FRAC) * max_lag))
    c_inf        = float(np.mean(dacf[long_start:]))
    transitioned = c_inf > config.DACF_THRESHOLD_ANG2

    lag_fs = np.arange(max_lag, dtype=float) * traj_timestep_fs

    return {
        'dacf'        : dacf.tolist(),
        'lag_fs'      : lag_fs.tolist(),
        'c_inf_ang2'  : c_inf,
        'transitioned': transitioned,
        'n_frames'    : n_frames,
        'n_atoms'     : n_atoms,
    }


# ---------------------------------------------------------------------------
# Per-simulation assessment with caching
# ---------------------------------------------------------------------------

def assess_structural_transition_for_sim(
        sim_dir: str,
        force_recompute: bool = False,
) -> dict | None:
    """Compute or load cached DACF for a single AIMD temperature sim_dir.

    The result is cached in ``sim_dir/anharmonic_phonons/dacf.json``.

    Args:
        sim_dir:         Path to a temperature sub-folder
                         (e.g. ``database/AgI/P63mc/aimd/1200K``).
        force_recompute: Ignore cached result and recompute.

    Returns:
        DACF result dict (keys: ``'dacf'``, ``'lag_fs'``, ``'c_inf_ang2'``,
        ``'transitioned'``, ``'n_frames'``, ``'n_atoms'``, ``'sim_dir'``,
        ``'temperature_K'``), or ``None`` if the trajectory is missing or
        analysis fails.
    """
    anh_dir    = os.path.join(sim_dir, 'anharmonic_phonons')
    cache_path = os.path.join(anh_dir, 'dacf.json')

    if not force_recompute and os.path.exists(cache_path):
        try:
            with open(cache_path) as fh:
                return json.load(fh)
        except Exception:
            pass  # cache corrupt — fall through to recompute

    traj_path  = os.path.join(sim_dir, 'output.traj')
    input_path = os.path.join(sim_dir, 'simulation-input.json')

    if not os.path.exists(traj_path) or os.path.getsize(traj_path) == 0:
        return None
    if not os.path.exists(input_path):
        return None

    with open(input_path) as fh:
        params = json.load(fh)

    timestep_fs      = float(params['timestep_fs'])
    nblock           = int(params['nblock'])
    step_skip        = int(params.get('step_skip', 1))
    step_equiv       = int(params.get('step_equiv', 100))
    # Backward/forward compatible key: older runs used `temperature`,
    # current inputs use `temperature_K`.
    temperature_K    = float(params.get('temperature_K', params.get('temperature')))
    traj_timestep_fs = timestep_fs * nblock * step_skip

    try:
        from ase.io import read as ase_read
        full_traj = ase_read(traj_path, index=f'::{step_skip}')
    except Exception as exc:
        print(f'  [dacf] {sim_dir}: could not read trajectory: {exc}')
        return None

    prod_traj = full_traj[step_equiv:]
    if len(prod_traj) < 4:
        print(f'  [dacf] {sim_dir}: only {len(prod_traj)} production frames — skipping.')
        return None

    print(f'  [dacf] Computing DACF from {len(prod_traj)} frames ({sim_dir}) ...')
    try:
        result = compute_dacf_from_traj(prod_traj, traj_timestep_fs)
    except Exception as exc:
        print(f'  [dacf] {sim_dir}: DACF computation failed: {exc}')
        return None

    result['sim_dir']      = sim_dir
    result['temperature_K'] = temperature_K

    os.makedirs(anh_dir, exist_ok=True)
    with open(cache_path, 'w') as fh:
        json.dump(result, fh, indent=4)

    flag = '✗ TRANSITION' if result['transitioned'] else '✓ stable'
    print(f'    {flag}  C_∞ = {result["c_inf_ang2"]:.4f} Å²'
          f'  (threshold {config.DACF_THRESHOLD_ANG2:.3f} Å²)')

    return result


# ---------------------------------------------------------------------------
# Gate 0 — whole sym_dir assessment
# ---------------------------------------------------------------------------

def assess_structural_transition(
        sym_dir: str,
        force_recompute: bool = False,
) -> dict:
    """Gate 0: Check all AIMD simulations for structural phase transitions.

    Scans ``sym_dir/aimd/`` for completed trajectories, computes or loads the
    DACF for each temperature sub-folder, and fails the gate if **any**
    simulation shows a structural transition (C_∞ > threshold).

    Args:
        sym_dir:         Path to ``database/<material>/<symmetry>/``.
        force_recompute: Re-run DACF computation even if cached results exist.

    Returns:
        Dict with keys:

        ``'available'`` (bool)
            True if at least one trajectory was found.
        ``'sim_results'`` (list[dict])
            Per-simulation summaries (large DACF/lag arrays stripped).
        ``'transitioned_sims'`` (list[str])
            sim_dir paths where a structural transition was detected.
        ``'max_c_inf_ang2'`` (float | None)
            Maximum C_∞ across all simulations (Å²).
        ``'threshold_ang2'`` (float)
            The threshold used (``config.DACF_THRESHOLD_ANG2``).
        ``'status'`` (str)
            ``'ok'`` | ``'fail'`` | ``'missing'``.
        ``'message'`` (str)
            Human-readable explanation.
    """
    aimd_dir = os.path.join(sym_dir, 'aimd')

    if not os.path.isdir(aimd_dir):
        return {
            'available'         : False,
            'sim_results'       : [],
            'transitioned_sims' : [],
            'max_c_inf_ang2'    : None,
            'threshold_ang2'    : config.DACF_THRESHOLD_ANG2,
            'status'            : config.STATUS_MISSING,
            'message'           : (
                'No AIMD directory found — structural transition check skipped.'
            ),
        }

    # ── Discover completed sim_dirs ────────────────────────────────────────
    sim_dirs = []
    for temp_folder in sorted(os.listdir(aimd_dir)):
        if not temp_folder.endswith('K'):
            continue
        sd        = os.path.join(aimd_dir, temp_folder)
        traj_path = os.path.join(sd, 'output.traj')
        if (os.path.isdir(sd)
                and os.path.exists(traj_path)
                and os.path.getsize(traj_path) > 0):
            sim_dirs.append(sd)

    if not sim_dirs:
        return {
            'available'         : False,
            'sim_results'       : [],
            'transitioned_sims' : [],
            'max_c_inf_ang2'    : None,
            'threshold_ang2'    : config.DACF_THRESHOLD_ANG2,
            'status'            : config.STATUS_MISSING,
            'message'           : 'No completed AIMD trajectories found.',
        }

    # ── Compute DACF per sim_dir ───────────────────────────────────────────
    sim_results:       list[dict] = []
    transitioned_sims: list[str]  = []

    for sd in sim_dirs:
        res = assess_structural_transition_for_sim(sd, force_recompute=force_recompute)
        if res is not None:
            sim_results.append(res)
            if res.get('transitioned', False):
                transitioned_sims.append(sd)

    if not sim_results:
        return {
            'available'         : False,
            'sim_results'       : [],
            'transitioned_sims' : [],
            'max_c_inf_ang2'    : None,
            'threshold_ang2'    : config.DACF_THRESHOLD_ANG2,
            'status'            : config.STATUS_MISSING,
            'message'           : 'DACF could not be computed for any trajectory.',
        }

    max_c_inf = max(r['c_inf_ang2'] for r in sim_results)
    thr       = config.DACF_THRESHOLD_ANG2
    T_labels  = [
        os.path.basename(r['sim_dir'])
        for r in sim_results if r.get('transitioned')
    ]

    if transitioned_sims:
        status  = config.STATUS_FAIL
        message = (
            f'Structural transition detected in {len(transitioned_sims)} of '
            f'{len(sim_results)} MD simulation(s) '
            f'({", ".join(T_labels)}). '
            f'Max C\u221e = {max_c_inf:.3f} \u00c5\u00b2 '
            f'(threshold {thr:.3f} \u00c5\u00b2). '
            f'Material is dynamically unstable — excluded from convex hull.'
        )
    else:
        status  = config.STATUS_OK
        message = (
            f'No structural transition detected across {len(sim_results)} '
            f'MD simulation(s). '
            f'Max C\u221e = {max_c_inf:.4f} \u00c5\u00b2 '
            f'(threshold {thr:.3f} \u00c5\u00b2).'
        )

    # Strip large DACF/lag arrays from the per-sim summaries stored in the
    # report (full arrays are preserved in each sim_dir/anharmonic_phonons/dacf.json).
    slim_results = [
        {k: v for k, v in r.items() if k not in ('dacf', 'lag_fs')}
        for r in sim_results
    ]

    return {
        'available'         : True,
        'sim_results'       : slim_results,
        'transitioned_sims' : transitioned_sims,
        'max_c_inf_ang2'    : max_c_inf,
        'threshold_ang2'    : thr,
        'status'            : status,
        'message'           : message,
    }
