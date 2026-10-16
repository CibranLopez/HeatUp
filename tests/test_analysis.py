"""Analysis tests on synthetic trajectories with known answers (no calculator needed)."""
import numpy as np
import pytest

from heatup import cfg
from heatup.constants import H_MEV_FS
from heatup.md_analysis import TrajData
from heatup.structural_transition import transition_metrics, verdict
from heatup.vdos import compute_vdos
from heatup.vibthermo import ModeSet


def _traj(frac, cell, mom=None, dt=5.0, symbols=None, T=300.0):
    n_t, n = frac.shape[:2]
    return TrajData(frac=frac, cell=cell, mom=mom, masses=np.full(n, 63.5),
                    symbols=symbols or ["Cu"] * n, epot=np.zeros(n_t), stress=None,
                    dt_fs=dt, T_target=T, n_equil=0)


def _lattice(n_side=2, a=3.6):
    g = np.array([[i, j, k] for i in range(n_side) for j in range(n_side) for k in range(n_side)], float)
    return (g + 0.25) / n_side, np.eye(3) * a * n_side


def test_vdos_einstein_peak_normalisation_and_free_energy():
    w0, dt, n_t, n_at = 30.0, 5.0, 4000, 27
    nu = w0 / H_MEV_FS                                               # cycles / fs
    t = np.arange(n_t) * dt
    rng = np.random.default_rng(1)
    phase = rng.uniform(0, 2 * np.pi, (n_at, 3))
    p = -np.sin(2 * np.pi * nu * t[:, None, None] + phase[None]) * 2 * np.pi * nu * 63.5 * 0.05
    frac, cell = _lattice(3)
    frac = np.broadcast_to(frac, (n_t, n_at, 3)).copy()
    td = _traj(frac, np.broadcast_to(cell, (n_t, 3, 3)).copy(), mom=p)
    r = compute_vdos(td)
    ms = r["modes"]
    assert ms.total_weight == pytest.approx(3.0, rel=1e-6)
    peak = ms.omega[np.argmax(ms.weight)]
    assert abs(peak - w0) < r["resolution_meV"]
    ein = ModeSet([w0], [3.0])
    for T in (100.0, 300.0, 600.0):
        assert ms.f_vib(T) == pytest.approx(ein.f_vib(T), abs=2e-3)   # resolution-limited, < 2 meV/atom
    assert r["alias_weight_top10pct"] < 1e-3


def _stable_traj(n_t=1200, dt=5.0, shift=None, strain=None):
    rng = np.random.default_rng(2)
    f0, h = _lattice(3)
    n = len(f0)
    frac = f0[None] + rng.normal(0, 0.02, (n_t, n, 3)) / 3.0
    cell = np.broadcast_to(h, (n_t, 3, 3)).copy()
    ramp = (np.arange(n_t) > n_t // 2)[:, None, None]
    if shift is not None:
        frac = frac + ramp * shift                                    # second half displaced
    if strain is not None:
        cell = np.where(ramp, h @ (np.eye(3) + strain), h)
    return _traj(frac, cell)


def test_gate0_stable_crystal_passes():
    cfg.gates.block_ps = 1.0
    m = transition_metrics(_stable_traj())
    st, why = verdict(m)
    assert st == "ok", why
    assert m["rms_A"] < 0.1


def test_gate0_collective_shift_removed_but_internal_displacement_fails():
    cfg.gates.block_ps = 1.0
    # rigid translation of the whole crystal is NOT a transition
    st, _ = verdict(transition_metrics(_stable_traj(shift=np.array([0.1, 0.0, 0.0]))))
    assert st == "ok"
    # half the atoms move by 1.5 A -> transition
    tr = _stable_traj()
    d = np.zeros((tr.n_frames, tr.n_atoms, 3))
    d[tr.n_frames // 2:, : tr.n_atoms // 2, 0] = 1.5 / tr.cell[0][0, 0]
    tr.frac = tr.frac + d
    st, why = verdict(transition_metrics(tr))
    assert st == "fail" and any("displacement" in w for w in why)


def test_gate0_cell_shear_and_volume():
    cfg.gates.block_ps = 1.0
    shear = np.zeros((3, 3)); shear[0, 1] = 0.12
    st, why = verdict(transition_metrics(_stable_traj(strain=shear)))
    assert st == "fail" and any("shape" in w for w in why)
    st, why = verdict(transition_metrics(_stable_traj(strain=np.eye(3) * 0.06)))
    assert st == "fail" and any("volume" in w for w in why)
    st, _ = verdict(transition_metrics(_stable_traj(strain=np.eye(3) * 0.01)))
    assert st == "ok"                                                 # 3% volume change is thermal expansion
