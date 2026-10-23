"""heatup.compute.md
====================
Molecular dynamics with the interatomic potential (step ``md``).  Classical MD on an MLIP potential-energy
surface (v0.2 called it "AIMD", hence the old ``aimd/`` directory).

Layout::

    md/POSCAR, md/cell.json          supercell built from relaxation/CONTCAR
    md/<T>K/output.traj              frames every ``md.nblock`` steps (positions, cell, **momenta**, energy, stress)
    md/<T>K/md.json                  parameters - written LAST: its presence means the run is complete
    md/<T>K/manifest_md.json, status_md.json

The run **resumes** after a wall-time kill (HPC) from the last stored frame (``resume.json`` holds the settings it
started with; a different configuration restarts from scratch).  Any restart from scratch (forced,
or the manifest is stale) removes ``output.traj`` *and* ``md.json`` *and* the analysis results derived from them, so
a half-written new trajectory is never mistaken for a finished one.

Barostat mass: ``pfactor = ptime^2 * B`` with the real bulk modulus B (from ``elastic.json`` when available, so run
``elastic`` before ``md``: ``run_plan`` keeps that order).  Momenta are written to the trajectory and used by the VDOS; ``nblock`` = 5 gives
a Nyquist of 413 meV; reproducible seed; centre-of-mass momentum removed; the ``bool * NoneType`` NPT blow-up is
reported as a diverged barostat.
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

from heatup.compute.calculator import build_calculator, calculator_label, release_calculator
from heatup.core import ledger
from heatup.core.config import cfg
from heatup.core.layout import PhaseDir, read_json, trajectory_frames, write_json
from heatup.prepare.export import build_md_cell

log = logging.getLogger("heatup")


try:                                    # ASE >= 3.23 renamed the class
    from ase.md.melchionna import MelchionnaNPT as NPT
except ImportError:                     # pragma: no cover
    from ase.md.npt import NPT

log = logging.getLogger("heatup")


def barostat_bulk_modulus(phase: PhaseDir) -> float:
    el = read_json(phase.elastic_json)
    if el:
        return float(el["moduli"]["bulk_hill_GPa"])
    return cfg.md.default_bulk_modulus_GPa


def run_md(phase_dir, device: str = "cpu", force: bool = False, temperature: float = 300.0) -> None:
    phase = PhaseDir.of(phase_dir)
    m = cfg.md
    sim = phase.md_run(temperature)
    traj_path = os.path.join(sim, "output.traj")
    expected = m.n_steps // m.nblock + 1

    if (not force and os.path.exists(os.path.join(sim, "md.json")) and trajectory_frames(traj_path) >= expected
            and ledger.is_fresh(phase, "md", temperature)):
        log.info("[cached] %s/md/%dK (up to date, not recomputed)", phase.tag, temperature)
        return
    if not phase.has_relaxation():
        raise FileNotFoundError(f"{phase.tag}: converged relaxation required first")
    if m.ensemble.upper() not in ("NPT", "NVT"):
        raise ValueError(f"md.ensemble must be NPT or NVT, got {m.ensemble!r}")

    os.makedirs(sim, exist_ok=True)
    key = {"fingerprint": ledger.fingerprint("md", phase.material), "contcar": ledger.file_hash(phase.contcar)}
    key_file = os.path.join(sim, "resume.json")
    if os.path.exists(traj_path):
        # a finished run is judged by its manifest; an interrupted one by the key written when it started
        stale = (not ledger.is_fresh(phase, "md", temperature) if ledger.read_manifest(phase, "md", temperature)
                 else read_json(key_file) != key)
    else:
        stale = False
    if force or stale:                       # restart from scratch: nothing derived from the old run may survive
        for f in ("output.traj", "md.json", *(ledger.STEPS[s].outputs[0] for s in ledger.ANALYSIS_STEPS)):
            p = os.path.join(sim, f)
            if os.path.exists(p):
                os.remove(p)
    elif os.path.exists(os.path.join(sim, "md.json")):       # resume of a run that was extended/interrupted
        os.remove(os.path.join(sim, "md.json"))

    write_json(key_file, key)
    B = barostat_bulk_modulus(phase)
    done_frames = trajectory_frames(traj_path) if os.path.exists(traj_path) else 0
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
        "bulk_modulus_GPa": B, "n_atoms": len(atoms),
        "resumed": resumed, "wall_s": time.time() - t0,
        "calculator": calculator_label(phase.material), "seed": m.seed,
    })
    ledger.record(phase, "md", temperature)
    if os.path.exists(key_file):
        os.remove(key_file)
