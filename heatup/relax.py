"""heatup.relax
================
Geometry optimisation of the reference structure (step ``relax``).

Writes ``relaxation/{CONTCAR, energy.json, run.traj, manifest.json}``.

``energy.json``::

    energy_eV_per_atom      potential energy of the relaxed cell
    enthalpy_eV_per_atom    E + P V   (== energy at P = 0)  <- hull reference
    volume_A3_per_atom, n_atoms, converged, fmax_final, stress_GPa
    spacegroup_input / spacegroup_relaxed

Changes vs v0.2: FixSymmetry (assess *this* phase), tight default fmax,
FrechetCellFilter (``ExpCellFilter`` is deprecated and biases the strain
coordinates), finite pressure handled properly (enthalpy), convergence is
checked and failure is an error instead of a silent "success".
"""

from __future__ import annotations

import logging
import os

import numpy as np
from ase.constraints import FixSymmetry
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS

from heatup import manifest
from heatup.calculator import build_calculator, calculator_label, release_calculator
from heatup.config import cfg
from heatup.constants import EV_A3_TO_GPA, GPA_TO_EV_A3
from heatup.paths import PhaseDir, write_json
from heatup.structure import read_structure, spacegroup, write_poscar

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
    inputs = [phase.poscar]
    if (not force and os.path.exists(phase.energy_json)
            and manifest.is_fresh(phase.relax_dir, "relax", phase.material, inputs)):
        log.info("[cached] %s/relaxation (up to date, not recomputed)", phase.tag)
        return
    if not os.path.exists(phase.poscar):
        raise FileNotFoundError(phase.poscar)

    os.makedirs(phase.relax_dir, exist_ok=True)
    atoms = read_structure(phase.poscar)
    sg_in = spacegroup(atoms)
    calc = build_calculator(device, material=phase.material)
    try:
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
            "fmax_final_eV_A": fres,
            "stress_GPa": stress,
            "spacegroup_input": list(sg_in),
            "spacegroup_relaxed": list(spacegroup(atoms, 1e-2)),
            "calculator": calculator_label(phase.material),
        })
        manifest.write_manifest(phase.relax_dir, "relax", phase.material, inputs)
        log.info("  E0 = %.6f eV/atom, SG %s -> %s, converged=%s",
                 e / n, sg_in[1], spacegroup(atoms, 1e-2)[1], conv)
        if not conv:
            raise RuntimeError(
                f"{phase.tag}: relaxation did not converge in {rcfg.max_steps} steps "
                f"(max |F| = {fres:.4f} eV/A)")
    finally:
        atoms.calc = None
        release_calculator(calc)
