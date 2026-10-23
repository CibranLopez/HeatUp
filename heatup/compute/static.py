"""heatup.compute.static
========================
The two static calculations of a phase: ``relax`` and ``elastic``.

``relax``  -> ``relaxation/{CONTCAR, energy.json, run.traj}``
    Geometry optimisation of the reference structure.  ``energy.json``::

        energy_eV_per_atom      potential energy of the relaxed cell
        enthalpy_eV_per_atom    E + P V   (== energy at P = 0)  <- hull reference
        volume_A3_per_atom, n_atoms, converged, fmax_final_eV_A, stress_GPa, pressure_GPa
        spacegroup_input / spacegroup_relaxed, calculator

    FixSymmetry (assess *this* phase), tight default fmax, FrechetCellFilter (``ExpCellFilter`` is deprecated and
    biases the strain coordinates), finite pressure handled properly (enthalpy).  ``relax.optimize = False`` (``"relax"``
    not asked for in ``cfg.plan.gates``): the given structure is accepted as relaxed - CONTCAR is a copy of it and
    energy.json holds its single-point energy, residual force and stress (``"optimized": false``).  A relaxation that does not
    converge is an error, **its manifest is not written** (so it is never reported as cached) and CONTCAR/energy.json
    stay for inspection.

``elastic`` -> ``elastic/elastic.json``
    Stiffness tensor C (6x6, GPa) by central-difference stress-strain with **relaxed ions**.  Voigt order xx, yy,
    zz, yz, xz, xy; shear strains are engineering strains.  Relaxed-ion constants (clamped ions over-stiffen every
    structure with internal degrees of freedom and can *hide* a Born instability - e.g. a ferroelectric/tilt
    softening shows up only after the ions relax); Voigt, Reuss and Hill averages (the Born criterion itself is on
    C, not on the averages); strain large enough to be above the float32 noise floor; the residual stress is
    reported so a mis-relaxed cell is visible.
"""

from __future__ import annotations

import logging
import os

import numpy as np
from ase.constraints import FixSymmetry
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS

from heatup.compute.calculator import build_calculator, calculator_label, release_calculator
from heatup.core import ledger
from heatup.core.config import EV_A3_TO_GPA, GPA_TO_EV_A3, cfg
from heatup.core.layout import PhaseDir, read_structure, spacegroup, write_json, write_poscar
from heatup.prepare.export import VOIGT_LABELS, elastic_cells

log = logging.getLogger("heatup")




# ---------------------------------------------------------------------------
# relax
# ---------------------------------------------------------------------------

log = logging.getLogger("heatup")


def relax_atoms(atoms, calc, *, fmax: float, max_steps: int, relax_cell: bool,
                constant_volume: bool = False, pressure_GPa: float = 0.0,
                fix_symmetry: bool = False, symprec: float = 1e-3,
                trajectory: str | None = None):
    """Relax *atoms* in place.  Returns ``(converged, final_max_force)``."""
    atoms.calc = calc
    if fix_symmetry:
        atoms.set_constraint(FixSymmetry(atoms, symprec=symprec))
    target = (FrechetCellFilter(atoms, constant_volume=constant_volume,
                                scalar_pressure=pressure_GPa * GPA_TO_EV_A3)
              if relax_cell else atoms)
    dyn = BFGS(target, trajectory=trajectory, logfile=None)
    converged = bool(dyn.run(fmax=fmax, steps=max_steps))
    forces = target.get_forces()
    return converged, float(np.sqrt((forces ** 2).sum(axis=1)).max())


def run_relaxation(phase_dir, device: str = "cpu", force: bool = False) -> None:
    phase = PhaseDir.of(phase_dir)
    rcfg = cfg.relax
    if not force and phase.has_relaxation() and ledger.is_fresh(phase, "relax"):
        log.info("[cached] %s/relaxation (up to date, not recomputed)", phase.tag)
        return
    if not os.path.exists(phase.poscar):
        raise FileNotFoundError(phase.poscar)

    os.makedirs(phase.relax_dir, exist_ok=True)
    atoms = read_structure(phase.poscar)
    sg_in = spacegroup(atoms)
    calc = build_calculator(device, material=phase.material)
    try:
        if rcfg.optimize:
            log.info("Relaxing %s  (fmax=%g, cell=%s, P=%g GPa, %s)", phase.tag, rcfg.fmax,
                     rcfg.relax_cell, rcfg.pressure_GPa, calculator_label(phase.material))
            try:
                conv, fres = relax_atoms(
                    atoms, calc, fmax=rcfg.fmax, max_steps=rcfg.max_steps,
                    relax_cell=rcfg.relax_cell, constant_volume=rcfg.constant_volume,
                    pressure_GPa=rcfg.pressure_GPa, fix_symmetry=rcfg.fix_symmetry,
                    symprec=rcfg.symprec, trajectory=os.path.join(phase.relax_dir, "run.traj"))
            except Exception as exc:  # noqa: BLE001
                stress_problem = any(s in str(exc) for s in
                                     ("compute_atomic_stresses", "get_stress", "too many indices"))
                if not (rcfg.relax_cell and stress_problem):
                    raise
                log.warning("model cannot provide stress (%s): relaxing positions only", exc)
                atoms = read_structure(phase.poscar)
                conv, fres = relax_atoms(
                    atoms, calc, fmax=rcfg.fmax, max_steps=rcfg.max_steps, relax_cell=False,
                    fix_symmetry=rcfg.fix_symmetry, symprec=rcfg.symprec)
        else:                                # already relaxed elsewhere: energy, forces and stress at the given structure
            log.info("Using %s as given (relax not requested): single point with %s", phase.tag,
                     calculator_label(phase.material))
            atoms.calc = calc
            conv = True
            fres = float(np.sqrt((atoms.get_forces() ** 2).sum(axis=1)).max())
            if fres > rcfg.fmax:
                log.warning("%s: max |F| = %.3f eV/A at the given structure (relax.fmax = %g): phonons and elastic "
                            "constants will carry this residual force", phase.tag, fres, rcfg.fmax)

        atoms.set_constraint()               # constraints must not leak into CONTCAR/MD
        n = len(atoms)
        e = float(atoms.get_potential_energy())
        vol = float(atoms.get_volume())
        try:
            stress = (atoms.get_stress(voigt=True) * EV_A3_TO_GPA).tolist()
        except Exception:  # noqa: BLE001
            stress = None
        write_poscar(phase.contcar, atoms)
        write_json(phase.energy_json, {
            "energy_eV_per_atom": e / n,
            "enthalpy_eV_per_atom": (e + rcfg.pressure_GPa * GPA_TO_EV_A3 * vol) / n,
            "volume_A3_per_atom": vol / n,
            "n_atoms": n,
            "pressure_GPa": rcfg.pressure_GPa,
            "converged": conv,
            "optimized": rcfg.optimize,
            "fmax_final_eV_A": fres,
            "stress_GPa": stress,
            "spacegroup_input": list(sg_in),
            "spacegroup_relaxed": list(spacegroup(atoms, 1e-2)),
            "calculator": calculator_label(phase.material),
        })
        log.info("  E0 = %.6f eV/atom, SG %s -> %s, converged=%s",
                 e / n, sg_in[1], spacegroup(atoms, 1e-2)[1], conv)
        if not conv:
            raise RuntimeError(
                f"{phase.tag}: relaxation did not converge in {rcfg.max_steps} steps "
                f"(max |F| = {fres:.4f} eV/A)")
        ledger.record(phase, "relax")                    # only a converged relaxation is a finished one
    finally:
        atoms.calc = None
        release_calculator(calc)


# ---------------------------------------------------------------------------
# elastic
# ---------------------------------------------------------------------------

log = logging.getLogger("heatup")

def elastic_from_stresses(stress_GPa: dict, delta: float) -> np.ndarray:
    """Stiffness tensor C (6x6, GPa, not symmetrised) from the stresses of the +/- strained cells: keys ``<voigt>_plus`` /
    ``<voigt>_minus`` (xx_plus, ..., xy_minus), each a Voigt 6-vector in GPa (tension positive), ions relaxed at fixed cell."""
    C = np.zeros((6, 6))
    for beta, lab in enumerate(VOIGT_LABELS):
        C[:, beta] = (np.asarray(stress_GPa[f"{lab}_plus"], float) - np.asarray(stress_GPa[f"{lab}_minus"], float)) / (2.0 * delta)
    return C


def elastic_tensor(atoms, calc, delta: float, relax_ions: bool, fmax: float) -> np.ndarray:
    """Stiffness tensor in GPa from the strained cells of ``prepare.export.elastic_cells`` evaluated with *calc*."""
    stresses = {}
    for name, a in elastic_cells(atoms, delta).items():
        a.calc = calc
        if relax_ions:
            BFGS(a, logfile=None).run(fmax=fmax, steps=300)
        stresses[name] = a.get_stress(voigt=True) * EV_A3_TO_GPA
        a.calc = None
    return elastic_from_stresses(stresses, delta)


def moduli(C: np.ndarray) -> dict:
    """Voigt / Reuss / Hill averages and derived constants from a 6x6 C (GPa)."""
    S = np.linalg.inv(C)
    BV = (C[0, 0] + C[1, 1] + C[2, 2] + 2 * (C[0, 1] + C[0, 2] + C[1, 2])) / 9.0
    GV = (C[0, 0] + C[1, 1] + C[2, 2] - C[0, 1] - C[0, 2] - C[1, 2]
          + 3 * (C[3, 3] + C[4, 4] + C[5, 5])) / 15.0
    BR = 1.0 / (S[0, 0] + S[1, 1] + S[2, 2] + 2 * (S[0, 1] + S[0, 2] + S[1, 2]))
    GR = 15.0 / (4 * (S[0, 0] + S[1, 1] + S[2, 2]) - 4 * (S[0, 1] + S[0, 2] + S[1, 2])
                 + 3 * (S[3, 3] + S[4, 4] + S[5, 5]))
    B, G = 0.5 * (BV + BR), 0.5 * (GV + GR)
    E = 9 * B * G / (3 * B + G)
    nu = (3 * B - 2 * G) / (2 * (3 * B + G))
    return {
        "bulk_voigt_GPa": BV, "bulk_reuss_GPa": BR, "bulk_hill_GPa": B,
        "shear_voigt_GPa": GV, "shear_reuss_GPa": GR, "shear_hill_GPa": G,
        "youngs_hill_GPa": E, "poisson_hill": nu,
        "universal_anisotropy": 5 * GV / GR + BV / BR - 6.0,
    }


def run_elastic(phase_dir, device: str = "cpu", force: bool = False) -> None:
    phase = PhaseDir.of(phase_dir)
    out = phase.elastic_json
    if not force and os.path.exists(out) and ledger.is_fresh(phase, "elastic"):
        log.info("[cached] %s/elastic (up to date, not recomputed)", phase.tag)
        return
    if not phase.has_relaxation():
        raise FileNotFoundError(f"{phase.tag}: converged relaxation required first")

    atoms = read_structure(phase.contcar)
    calc = build_calculator(device, material=phase.material)
    ecfg = cfg.elastic
    try:
        log.info("Elastic tensor %s (strain=±%g, relax_ions=%s)", phase.tag, ecfg.strain, ecfg.relax_ions)
        atoms.calc = calc
        sigma0 = atoms.get_stress(voigt=True) * EV_A3_TO_GPA
        atoms.calc = None
        C = elastic_tensor(atoms, calc, ecfg.strain, ecfg.relax_ions, ecfg.fmax)
        mod = _write_elastic(phase, C, sigma0, calculator_label(phase.material))
        ledger.record(phase, "elastic")
    finally:
        atoms.calc = None
        release_calculator(calc)


def _write_elastic(phase: PhaseDir, C: np.ndarray, sigma0: np.ndarray, calculator: str) -> dict:
    """``elastic/elastic.json`` from a raw tensor - shared by the MLIP step and by :func:`collect_elastic`."""
    asym = float(np.abs(C - C.T).max())
    C = 0.5 * (C + C.T)
    mod = moduli(C)
    write_json(phase.elastic_json, {
        "C_GPa": C.tolist(), "voigt_labels": VOIGT_LABELS,
        "moduli": mod, "asymmetry_GPa": asym, "residual_stress_GPa": np.asarray(sigma0, float).tolist(),
        "strain": cfg.elastic.strain, "relax_ions": cfg.elastic.relax_ions, "calculator": calculator,
    })
    log.info("  B=%.1f G=%.1f E=%.1f GPa nu=%.3f", mod["bulk_hill_GPa"],
             mod["shear_hill_GPa"], mod["youngs_hill_GPa"], mod["poisson_hill"])
    return mod


def collect_elastic(phase_dir, stresses_GPa: dict, residual_stress_GPa=None, calculator: str = "external") -> dict:
    """``elastic/elastic.json`` from stresses computed with ANY code on the cells of ``elastic/strain/`` (``prepare.export``).

    ``stresses_GPa``: ``{"xx_plus": [6 values], "xx_minus": ..., ..., "xy_minus": ...}`` - Voigt xx yy zz yz xz xy, GPa, tension
    positive, ions relaxed at fixed cell; ``residual_stress_GPa``: the stress of the unstrained cell (reported by Gate 1).  No
    manifest is written: the result is ``external``.  The strain amplitude is ``cfg.elastic.strain`` of the export."""
    phase = PhaseDir.of(phase_dir)
    C = elastic_from_stresses(stresses_GPa, cfg.elastic.strain)
    mod = _write_elastic(phase, C, np.zeros(6) if residual_stress_GPa is None else residual_stress_GPa, calculator)
    return mod
