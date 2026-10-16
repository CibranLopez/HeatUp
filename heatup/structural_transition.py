"""heatup.structural_transition
================================
Gate 0 - did the structure survive the MD run?

Why not the v0.2 DACF plateau
-----------------------------
v0.2 measured the long-time plateau of ``C(tau) = <du(t) du(t+tau)>`` with
``du = r(t) - r(first production frame)``.  The reference is a *single thermal
snapshot*: du(t) = delta(t) - delta(0), so even a perfectly stable crystal has
``C_inf = <delta^2>`` = the thermal mean-square displacement (0.01-0.1 A^2
at 1000 K) - i.e. the gate threshold (0.10 A^2) sat inside the thermal noise
and the reported value depended on the random first frame.

Present test (all per MD temperature)
-------------------------------------
1. block-mean positions of the first and last ``gates.block_ps`` of the
   production run (thermal noise averages out), centre-of-mass drift removed,
   mapped to Cartesian on the mean cell;
   * rms and max displacement of the non-mobile atoms   -> FAIL above thresholds
   * mobile species (``gates.mobile_species``) are excluded;
2. cell change between the same blocks: deviatoric Green-Lagrange strain and
   volume change                                          -> FAIL above thresholds
3. symmetry breaking: every symmetry operation of the *initial* structure is
   applied to the block-mean structures and the rms mismatch is measured.  A
   thermally vibrating crystal gives the mean-position noise (~0.02 A); a
   polar/tilt distortion gives its amplitude.  -> WARN when the last block is
   more broken than ``gates.symmetry_break_A`` (spglib on noisy *average*
   structures was tried first: it is unreliable in small, shear-fluctuating
   NPT cells);
4. diffusing species not declared as mobile (MSD)           -> reported, and they
   normally trigger (1);
5. the DACF relative to the **first-block mean** is still computed (as a
   diagnostic plot): its plateau is |<d>|^2 and is ~0 for a stable crystal.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from heatup import manifest
from heatup.config import (STATUS_FAIL, STATUS_MISSING, STATUS_OK, STATUS_WARN, cfg, forced)
from heatup.md_analysis import TrajData, autocorr_fft, diffusion_by_species, load_trajectory, unwrap_frac
from heatup.paths import PhaseDir, read_json, write_json

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
    g = cfg.gates
    mobile = list(mobile if mobile is not None else g.mobile_species)
    p = td.production()
    nb = int(round(g.block_ps * 1000.0 / p.dt_fs))
    nb = max(1, min(nb, p.n_frames // 2))
    short = p.n_frames < 2 * int(round(g.block_ps * 1000.0 / p.dt_fs))

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
        "symmetry_broken": bool(np.isfinite(sb1) and sb1 > g.symmetry_break_A),
        "dacf_A2": dacf[::sub], "dacf_lag_fs": (np.arange(max_lag) * p.dt_fs)[::sub],
        "dacf_plateau_A2": float(dacf[int(0.8 * max_lag):].mean()),
        "block_frames": nb, "production_ps": p.duration_ps, "too_short": bool(short),
        "mobile_species": mobile,
    }


def verdict(m: dict) -> tuple[str, list[str]]:
    g = cfg.gates
    fails, warns = [], []
    if m["rms_A"] > g.transition_rms_A:
        fails.append(f"rms block displacement {m['rms_A']:.2f} A > {g.transition_rms_A:g}")
    if m["max_A"] > g.transition_max_A:
        fails.append(f"max atomic displacement {m['max_A']:.2f} A > {g.transition_max_A:g}")
    if m["cell_shape_strain"] > g.cell_shape_strain:
        fails.append(f"cell shape strain {m['cell_shape_strain']:.3f} > {g.cell_shape_strain:g}")
    if abs(m["cell_volume_change"]) > g.cell_volume_change:
        fails.append(f"volume change {100 * m['cell_volume_change']:+.1f}% (> {100 * g.cell_volume_change:g}%)")
    if m["symmetry_broken"]:
        warns.append(f"symmetry of the initial structure broken by {m['symmetry_break_last_A']:.2f} A "
                     f"(was {m['symmetry_break_first_A']:.2f} A at the start)")
    if m["too_short"]:
        warns.append("production run shorter than two averaging blocks")
    if m["rms_A"] > 0.5 * g.transition_rms_A and not fails:
        warns.append(f"rms block displacement {m['rms_A']:.2f} A is close to the threshold")
    if fails:
        return STATUS_FAIL, fails
    return (STATUS_WARN if warns else STATUS_OK), warns


def assess_sim(phase_dir, temperature: float, force: bool | None = None) -> dict | None:
    """Per-temperature Gate-0 result (cached in ``md/<T>K/transition.json``)."""
    phase = PhaseDir.of(phase_dir)
    sim = phase.md_run(temperature)
    out = os.path.join(sim, "transition.json")
    traj = os.path.join(sim, "output.traj")
    force = forced("analysis", force)
    if (not force and os.path.exists(out)
            and manifest.is_fresh(sim, "analysis", phase.material, [traj])):
        return read_json(out)
    if not os.path.exists(traj) or os.path.getsize(traj) == 0:
        return None
    td = load_trajectory(sim)
    m = transition_metrics(td)
    status, notes = verdict(m)
    diff = diffusion_by_species(td)
    undeclared = [sp for sp, v in diff.items()
                  if v["mobile"] and sp not in cfg.gates.mobile_species]
    m.update({"status": status, "notes": notes, "temperature_K": temperature,
              "diffusive_species": [sp for sp, v in diff.items() if v["mobile"]],
              "undeclared_diffusive_species": undeclared})
    write_json(out, m, indent=None)
    manifest.write_manifest(sim, "analysis", phase.material, [traj])
    return m


def assess_structural_transition(phase_dir, force: bool | None = None) -> dict:
    """Gate 0 over every available MD temperature."""
    phase = PhaseDir.of(phase_dir)
    temps = phase.md_temperatures()
    if not temps:
        return {"available": False, "status": STATUS_MISSING, "sim_results": [],
                "message": "No MD trajectory found - run step 'md' first."}
    sims = []
    for T in temps:
        try:
            r = assess_sim(phase, T, force=force)
        except Exception as exc:  # noqa: BLE001
            log.warning("Gate 0 failed for %s %dK: %s", phase.tag, T, exc)
            r = None
        if r:
            sims.append(r)
    if not sims:
        return {"available": False, "status": STATUS_MISSING, "sim_results": [],
                "message": "MD trajectories could not be analysed."}
    order = {STATUS_OK: 0, STATUS_WARN: 1, STATUS_FAIL: 2}
    worst = max(sims, key=lambda r: order[r["status"]])
    failed = [f"{int(r['temperature_K'])}K" for r in sims if r["status"] == STATUS_FAIL]
    if worst["status"] == STATUS_FAIL:
        msg = (f"Structural transition/instability at {', '.join(failed)}: "
               + "; ".join(worst["notes"]))
    elif worst["status"] == STATUS_WARN:
        msg = "; ".join(worst["notes"])
    else:
        msg = (f"Structure retained at all {len(sims)} MD temperature(s) "
               f"(max rms block displacement {max(r['rms_A'] for r in sims):.2f} A).")
    slim = [{k: v for k, v in r.items() if k not in ("dacf_A2", "dacf_lag_fs")} for r in sims]
    return {"available": True, "status": worst["status"], "message": msg,
            "sim_results": slim, "max_rms_A": max(r["rms_A"] for r in sims),
            "failed_temperatures": failed}
