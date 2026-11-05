"""heatup.compute.anharmonic  --  explicit anharmonic free energy at fixed cell (DRAFT)
=========================================================================================
Why: for a dynamically stable phase, G = QHA + F_anh, where F_anh is the *classical* free energy of the true
potential relative to the harmonic crystal at the same cell.  F_anh is small and has a variance-reduced estimator,
so a few short NVT runs suffice (tested against an exact reference: see ``test_anharmonic.py`` and the reply).

Equations
---------
Generalised equipartition (any classical canonical ensemble, internal coordinates u = r - r0 about a stationary
point r0 of U at the fixed cell):

    < u . grad U > = (3N - 3) kB T                                   [Hellman 2013 uses the same identity implicitly
                                                                      in Eq. 21; derivation: textbook, e.g. Frenkel &
                                                                      Smit, "Understanding Molecular Simulation"]

so that the anharmonic excess energy

    dE(T) = < U - E0 + 1/2 sum_i u_i . F_i >_T     with F = -grad U, E0 = U(r0)       (= U0(T) - E0)

vanishes identically for a harmonic crystal (zero-variance control variate) and carries only the anharmonic part.
Gibbs-Helmholtz at fixed cell, d(F/T)/dT = -E_int/T^2 with E_int = <U> + (3N-3) kB T/2, harmonic E_int = E0 + (3N-3) kB T:

    F_anh(T) = F(T) - F_harm,cl(T) = - T * int_0^T  dE(T') / T'^2  dT'        (classical statistics)

dE ~ c T^2 at low T, so the integrand is constant there and the integral starts exactly at T = 0.

Limits (be honest about them)
-----------------------------
* CLASSICAL statistics: valid for T >~ Theta_D / 2; the quantum anharmonic correction is NOT included.
* Needs a harmonically stable reference r0 (no imaginary modes at this cell); hopping between minima invalidates it
  (flag ``hopping``: block means of dE drift).
* Fixed cell: use the cell of the thermal state (QHA volume or the mean NPT cell).  Isotropic scaling of the relaxed
  cell is exact only for cubic symmetry.  One cell is used for the whole T grid unless you call it per cell.
* Total free energy  G = G_QHA(T) + F_anh(T)  (add ``f_anh`` next to the QHA method, NEVER next to HA/VDOS-EHA:
  those already contain part of it).  QHA must use the same volume definition.
Validation (EMT Cu, 64 atoms, V/V0 = 1.04): F_anh(600 K) = +1.28, F_anh(900 K) = +2.64 meV/atom from the long run
(dE errors 0.1-0.2 meV/atom); ``test_anharmonic.py`` reproduces the 600 K value with a shorter run.
"""
from __future__ import annotations

import logging
import os
import warnings

import numpy as np

log = logging.getLogger("heatup")


def relax_ions_fixed_cell(atoms, calc, fmax: float = 1e-4, steps: int = 400):
    """Return a copy of *atoms* with ions relaxed at the given cell (reference sites r0 and E0)."""
    from ase.optimize import BFGS
    a = atoms.copy()
    a.calc = calc
    BFGS(a, logfile=None).run(fmax=fmax, steps=steps)
    return a


def equipartition_excess(atoms0, calc, temperature_K: float, n_equil: int = 1500, n_prod: int = 6000,
                         interval: int = 5, timestep_fs: float = 1.0, friction_per_fs: float = 0.01,
                         seed: int = 0, n_blocks: int = 10) -> dict:
    """dE(T) per atom (eV) with a block-average error, from Langevin NVT at fixed cell.

    ``atoms0`` must be at its reference sites r0 (forces ~ 0); ``calc`` any ASE calculator (model-agnostic)."""
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

    n = len(atoms0)
    r0 = atoms0.get_positions().copy()
    a = atoms0.copy()
    a.calc = calc
    e0 = float(a.get_potential_energy())
    fres = float(np.abs(a.get_forces()).max())
    if fres > 5e-3:
        log.warning("reference sites are not stationary (max |F| = %.4f eV/A): relax the ions first", fres)
    rng = np.random.default_rng(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        MaxwellBoltzmannDistribution(a, temperature_K=temperature_K, rng=rng)
        Stationary(a)
        dyn = Langevin(a, timestep_fs * units.fs, temperature_K=temperature_K,
                       friction=friction_per_fs / units.fs, rng=rng, fixcm=False)
    dE, u2 = [], []

    def sample():
        u = a.get_positions() - r0
        dE.append(a.get_potential_energy() - e0 + 0.5 * float((u * a.get_forces()).sum()))
        u2.append(float(((u - u.mean(axis=0)) ** 2).sum() / n))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dyn.run(n_equil)
        dyn.attach(sample, interval=interval)
        dyn.run(n_prod)
    dE = np.array(dE)
    blocks = dE[: len(dE) // n_blocks * n_blocks].reshape(n_blocks, -1).mean(axis=1)
    slope = np.polyfit(np.arange(n_blocks), blocks, 1)[0] * n_blocks
    return {"T_K": float(temperature_K), "dE_eV_per_atom": float(dE.mean() / n),
            "dE_err_eV_per_atom": float(blocks.std(ddof=1) / np.sqrt(n_blocks) / n),
            "rms_u_A": float(np.sqrt(np.mean(u2))), "block_drift_eV_per_atom": float(slope / n),
            "n_samples": int(len(dE)), "E0_eV": e0}


def integrate_anharmonic(T, dE, dE_err=None) -> dict:
    """F_anh(T_i) = -T_i int_0^{T_i} dE/T'^2 dT' by the trapezoid rule on the nodes [0, T_0, ..., T_i], with the
    integrand g = dE/T^2 taken constant (= g_0) on [0, T_0] (dE ~ T^2 at low T).  Errors: the statistical errors of
    g_j propagated in quadrature through the trapezoid weights (systematic error of the T grid not included)."""
    T = np.asarray(T, float)
    order = np.argsort(T)
    T, dE = T[order], np.asarray(dE, float)[order]
    err = np.zeros_like(dE) if dE_err is None else np.asarray(dE_err, float)[order]
    g, ge = dE / T ** 2, err / T ** 2
    n = len(T)
    F, Fe = np.zeros(n), np.zeros(n)
    for i in range(n):
        w = np.zeros(n)
        w[0] += T[0]                                   # [0, T_0]: constant g_0
        for k in range(1, i + 1):
            h = T[k] - T[k - 1]
            w[k - 1] += 0.5 * h
            w[k] += 0.5 * h
        F[i] = -T[i] * float((w * g).sum())
        Fe[i] = T[i] * float(np.sqrt(((w * ge) ** 2).sum()))
    return {"T_K": T, "dE_eV_per_atom": dE, "F_anh_eV_per_atom": F, "F_anh_err_eV_per_atom": Fe}


def anharmonic_free_energy(atoms0, calc, temperatures, seed: int = 0, **md_kwargs) -> dict:
    """F_anh(T) (classical, per atom) for the cell of *atoms0* (ions relaxed here).  ``md_kwargs`` -> equipartition_excess."""
    ref = relax_ions_fixed_cell(atoms0, calc)
    runs = []
    for i, T in enumerate(sorted(float(t) for t in temperatures)):
        runs.append(equipartition_excess(ref, calc, T, seed=seed + i, **md_kwargs))
        log.info("  anharmonic dE(%g K) = %.3f +- %.3f meV/atom", T, 1e3 * runs[-1]["dE_eV_per_atom"],
                 1e3 * runs[-1]["dE_err_eV_per_atom"])
    out = integrate_anharmonic([r["T_K"] for r in runs], [r["dE_eV_per_atom"] for r in runs],
                               [r["dE_err_eV_per_atom"] for r in runs])
    flags = [f"hopping at {r['T_K']:g} K: dE drifts by {1e3 * r['block_drift_eV_per_atom']:.2f} meV/atom over the run "
             "- reference sites may not be the thermal minimum"
             for r in runs if r["dE_err_eV_per_atom"] > 0 and abs(r["block_drift_eV_per_atom"]) > 4 * r["dE_err_eV_per_atom"]]
    out.update({"runs": runs, "statistics": "classical", "n_atoms": len(atoms0), "flags": flags})
    return out


def run_anharmonic(phase_dir, device: str = "cpu", force: bool = False, **overrides) -> None:
    """Step ``anharmonic`` (ledger runtime contract) -> ``phonons/anharmonic.json``.  Settings: ``cfg.anharmonic``
    (``overrides`` replace single fields for one call, e.g. ``temperatures=[300, 600]``).  The cell is the QHA cell
    (``cell_from="qha"``: isotropic scaling of the MD supercell to V_eq(reference_T_K)), so run ``qha`` first."""
    from heatup.compute.calculator import build_calculator, calculator_label, release_calculator
    from heatup.core import ledger
    from heatup.core.config import cfg
    from heatup.core.layout import PhaseDir, read_json, write_json
    from heatup.prepare.export import build_md_cell

    phase = PhaseDir.of(phase_dir)
    p = {k: overrides.get(k, getattr(cfg.anharmonic, k)) for k in (
        "temperatures", "cell_from", "reference_T_K", "n_equil", "n_prod", "interval", "timestep_fs",
        "friction_per_fs", "seed")}
    out = os.path.join(phase.phonon_dir, "anharmonic.json")
    if not force and os.path.exists(out) and ledger.is_fresh(phase, "anharmonic"):
        log.info("[cached] %s/anharmonic (up to date, not recomputed)", phase.tag)
        return
    if not phase.has_relaxation():
        raise FileNotFoundError(f"{phase.tag}: converged relaxation required first")
    os.makedirs(phase.phonon_dir, exist_ok=True)
    atoms = build_md_cell(phase).copy()
    scale = 1.0
    if p["cell_from"] == "qha":
        q = read_json(os.path.join(phase.phonon_dir, "qha.json"))
        if not q:
            raise FileNotFoundError(f"{phase.tag}: phonons/qha.json missing (cell_from='qha'; run step 'qha')")
        qt = np.asarray(q["temperatures"], float)
        T_ref = float(qt.max() if p["reference_T_K"] is None else p["reference_T_K"])
        scale = (float(np.interp(T_ref, qt, q["V_eq_A3_per_atom"])) / float(q["V0_ref_A3_per_atom"])) ** (1.0 / 3.0)
        atoms.set_cell(atoms.cell.array * scale, scale_atoms=True)
    elif p["cell_from"] != "relaxed":
        raise ValueError(f"cfg.anharmonic.cell_from = {p['cell_from']!r}: choose 'qha' | 'relaxed'")
    calc = build_calculator(device, material=phase.material)
    try:
        res = anharmonic_free_energy(atoms, calc, p["temperatures"], seed=p["seed"], n_equil=p["n_equil"],
                                     n_prod=p["n_prod"], interval=p["interval"], timestep_fs=p["timestep_fs"],
                                     friction_per_fs=p["friction_per_fs"])
    finally:
        release_calculator(calc)
    res.update({"cell_from": p["cell_from"], "linear_scale": scale, "calculator": calculator_label(phase.material)})
    write_json(out, {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in res.items()})
    ledger.record(phase, "anharmonic")
    if res["flags"]:
        for f in res["flags"]:
            log.warning("%s: %s", phase.tag, f)


def f_anh(phase, T) -> np.ndarray:
    """Contribution (eV/atom) from ``phonons/anharmonic.json`` for the assembler.  Below the lowest computed T it
    follows T^2 (classical low-T law); above the highest it raises (no extrapolation)."""
    from heatup.analyze.thermo import MissingData
    from heatup.core.layout import read_json
    d = read_json(os.path.join(phase.phonon_dir, "anharmonic.json"))
    if not d:
        raise MissingData(f"{phase.tag}: phonons/anharmonic.json missing (run_anharmonic)")
    Td, Fd = np.asarray(d["T_K"], float), np.asarray(d["F_anh_eV_per_atom"], float)
    T = np.asarray(T, float)
    if T.max() > Td.max() + 1e-9:
        raise MissingData(f"{phase.tag}: F_anh computed up to {Td.max():g} K, hull needs {T.max():g} K")
    return np.where(T < Td[0], Fd[0] * (T / Td[0]) ** 2, np.interp(T, Td, Fd))
