"""heatup.md
=============
Molecular dynamics with the interatomic potential (step ``md``).

(v0.2 called this "AIMD" - it is classical MD on an MLIP potential-energy
surface, so the directory is now ``md/``.)

Layout::

    md/POSCAR, md/cell.json          supercell built from relaxation/CONTCAR
    md/<T>K/output.traj              frames every ``md.nblock`` steps
                                     (positions, cell, **momenta**, energy, stress)
    md/<T>K/md.json                  parameters, run state
    md/<T>K/manifest.json

Fixes vs v0.2
-------------
* barostat mass: ``pfactor = ptime^2 * B`` with the real bulk modulus B
  (from ``elastic.json`` when available).  v0.2 used ``ptime^2 * 1 GPa`` - a
  barostat ~100x too light, which makes the cell ring wildly;
* momenta are written to the trajectory and used by the VDOS (v0.2 threw them
  away and re-derived "velocities" from positions 20 fs apart);
* ``nblock`` = 5 instead of 20 (Nyquist 413 meV instead of 103 meV);
* the run can **resume** after a wall-time kill (HPC) instead of deleting the
  partial trajectory;
* the ``bool * NoneType`` NPT blow-up is reported as such;
* reproducible seed; centre-of-mass momentum removed.
"""

from __future__ import annotations

import logging
import os
import time

import numpy as np
from ase import units
from ase.io import read as ase_read
from ase.io.trajectory import Trajectory
try:                                    # ASE >= 3.23 renamed the class
    from ase.md.melchionna import MelchionnaNPT as NPT
except ImportError:                     # pragma: no cover
    from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

from heatup import manifest
from heatup.calculator import build_calculator, calculator_label, release_calculator
from heatup.config import AUTO_MIN_LENGTH_A, cfg
from heatup.paths import PhaseDir, read_json, write_json
from heatup.structure import read_structure, resolve_supercell, upper_triangular, write_poscar

log = logging.getLogger("heatup")


def build_md_cell(phase: PhaseDir, force: bool = False):
    """MD supercell from the relaxed cell: ``md.supercell`` if set, else perpendicular width >= ``md.min_cell_A``."""
    poscar = os.path.join(phase.md_dir, "POSCAR")
    info = os.path.join(phase.md_dir, "cell.json")
    mcfg = cfg.md
    request = {"supercell": None if mcfg.supercell is None else list(mcfg.supercell),
               "min_cell_A": mcfg.min_cell_A, "max_atoms": mcfg.max_atoms}
    if os.path.exists(poscar) and not force:
        old = read_json(info) or {}
        if old.get("request") == request and old.get("source") == manifest.file_hash(phase.contcar):
            return read_structure(poscar)
    unit = upper_triangular(read_structure(phase.contcar))
    reps = resolve_supercell(unit.cell.array, len(unit), supercell=mcfg.supercell, min_length=mcfg.min_cell_A,
                             max_atoms=mcfg.max_atoms, auto_min_length=AUTO_MIN_LENGTH_A["md"], what="MD supercell")
    sc = unit.repeat(reps)
    os.makedirs(phase.md_dir, exist_ok=True)
    write_poscar(poscar, sc)
    write_json(info, {"repetitions": list(reps), "n_atoms_unit": len(unit),
                      "n_atoms": len(sc), "request": request,
                      "source": manifest.file_hash(phase.contcar)})
    log.info("MD supercell %s: reps=%s, %d atoms", phase.tag, reps, len(sc))
    return sc


def barostat_bulk_modulus(phase: PhaseDir) -> float:
    el = read_json(phase.elastic_json)
    if el:
        return float(el["moduli"]["bulk_hill_GPa"])
    return cfg.md.default_bulk_modulus_GPa


def _n_frames(path: str) -> int:
    try:
        with Trajectory(path) as t:
            return len(t)
    except Exception:  # noqa: BLE001
        return 0


def run_md(phase_dir, device: str = "cpu", force: bool = False, temperature: float = 300.0) -> None:
    phase = PhaseDir.of(phase_dir)
    m = cfg.md
    sim = phase.md_run(temperature)
    traj_path = os.path.join(sim, "output.traj")
    inputs = [phase.contcar]
    expected = m.n_steps // m.nblock + 1

    if (not force and os.path.exists(traj_path) and _n_frames(traj_path) >= expected
            and manifest.is_fresh(sim, "md", phase.material, inputs)):
        log.info("[cached] %s/md/%dK (up to date, not recomputed)", phase.tag, temperature)
        return
    if not phase.has_relaxation():
        raise FileNotFoundError(f"{phase.tag}: converged relaxation required first")
    if m.ensemble.upper() not in ("NPT", "NVT"):
        raise ValueError(f"md.ensemble must be NPT or NVT, got {m.ensemble!r}")

    os.makedirs(sim, exist_ok=True)
    stale = (not manifest.is_fresh(sim, "md", phase.material, inputs))
    if force or stale:
        for f in ("output.traj", "analysis.json", "vdos.json", "transition.json"):
            p = os.path.join(sim, f)
            if os.path.exists(p):
                os.remove(p)

    B = barostat_bulk_modulus(phase)
    done_frames = _n_frames(traj_path) if os.path.exists(traj_path) else 0
    steps_left = m.n_steps
    if done_frames > 1:
        steps_left = m.n_steps - (done_frames - 1) * m.nblock
        atoms = ase_read(traj_path, index=-1)
        resumed = True
        log.info("Resuming %s %dK from frame %d (%d steps left)", phase.tag, temperature,
                 done_frames, steps_left)
    else:
        atoms = build_md_cell(phase, force=force or stale).copy()
        resumed = False

    calc = build_calculator(device, material=phase.material)
    atoms.calc = calc
    t0 = time.time()
    try:
        # fail early (and retryably) if the device cannot evaluate what the ensemble needs
        atoms.get_forces()
        if m.ensemble.upper() == "NPT":
            atoms.get_stress()
        if not resumed:
            rng = np.random.default_rng(m.seed)
            MaxwellBoltzmannDistribution(atoms, temperature_K=2.0 * temperature, rng=rng)
            Stationary(atoms)
        dt = m.timestep_fs * units.fs
        kw = dict(temperature_K=temperature, ttime=m.ttime_fs * units.fs)
        if m.ensemble.upper() == "NPT":
            kw.update(externalstress=m.pressure_GPa * units.GPa,
                      pfactor=(m.ptime_fs * units.fs) ** 2 * B * units.GPa)
        else:
            kw.update(pfactor=None)
        dyn = NPT(atoms, timestep=dt, logfile=os.path.join(sim, f"{m.ensemble.lower()}.log"), **kw)
        traj = Trajectory(traj_path, "a" if resumed else "w", atoms,
                          properties=["energy", "stress"])
        dyn.attach(traj.write, interval=m.nblock)
        log.info("%s MD %s %g K: %d steps x %g fs (%.1f ps), nblock=%d, B=%.0f GPa, %d atoms, %s",
                 m.ensemble, phase.tag, temperature, steps_left, m.timestep_fs,
                 steps_left * m.timestep_fs / 1e3, m.nblock, B, len(atoms),
                 calculator_label(phase.material))
        try:
            dyn.run(steps_left)
        except TypeError as exc:
            if "NoneType" in str(exc):
                raise RuntimeError("NPT barostat diverged (cell determinant -> 0); check the "
                                   "structure / lower md.timestep_fs / raise md.ptime_fs") from exc
            raise
        finally:
            traj.close()
    finally:
        atoms.calc = None
        release_calculator(calc)

    write_json(os.path.join(sim, "md.json"), {
        "temperature_K": temperature, "ensemble": m.ensemble, "timestep_fs": m.timestep_fs,
        "n_steps": m.n_steps, "nblock": m.nblock, "frame_dt_fs": m.timestep_fs * m.nblock,
        "equil_ps": m.equil_ps, "bulk_modulus_GPa": B, "n_atoms": len(atoms),
        "resumed": resumed, "wall_s": time.time() - t0,
        "calculator": calculator_label(phase.material), "seed": m.seed,
    })
    manifest.write_manifest(sim, "md", phase.material, inputs)
