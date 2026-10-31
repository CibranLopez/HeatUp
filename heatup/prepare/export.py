"""heatup.prepare.export
========================
The structures every calculation runs on, and their export to files for **any code** (DFT, another potential).

One definition of each cell, used twice:

* here, by :func:`export_inputs` (notebook 1), which writes them into the folders ``prepare_project`` created;
* by the compute steps (notebook 2), which build the same cells in memory with the same functions - so a DFT run
  and an MLIP run of a step always see identical supercells and displacements.

What is written, from ``relaxation/CONTCAR`` (VASP ``POSCAR`` files, atom order = phonopy/phono3py order)::

    phonons/ha/     SPOSCAR  phonopy_disp.yaml  disp-001/POSCAR ...      step ``phonons``  (phonopy displacements)
    phonons/qha/    V01/POSCAR ... volumes.json                          step ``qha``      (isotropically scaled cells)
    phonons/fc3/    SPOSCAR  phono3py_disp.yaml  disp-00001/POSCAR ...   step ``fc3``      (phono3py displacements)
    elastic/strain/ xx_plus/POSCAR xx_minus/POSCAR ... strains.json      step ``elastic``  (12 strained cells)
    md/             POSCAR  cell.json                                    step ``md``       (supercell to start from)

What was written is recorded in the ``export`` section of the step record (``step_<step>.json``: files, hash of the CONTCAR,
digest of the options): exporting again is a no-op unless the structure or the options changed, only the files HeatUp wrote are
ever replaced, and a folder is created only when its files could be written.

Every calculation folder also gets the structure it starts from as ``POSCAR`` (:func:`stage_input`): ``relaxation/POSCAR`` the
given structure (to compare with ``CONTCAR``), ``elastic/POSCAR`` and ``phonons/POSCAR`` the relaxed cell, ``md/<T>K/POSCAR`` the
supercell of that run.

Bringing results back (anything that writes the files notebook 3 reads is fine): ``heatup.collect_ha`` turns forces of the
``phonons/ha`` cells into ``ha.json``, ``heatup.collect_elastic`` turns stresses of the strained cells into ``elastic.json``;
trajectories and energies are files in the layout of ``heatup.core.layout`` (see the last cell of notebook 3).  Phonon forces:
eV/A of every ``disp-*`` cell (and ``perfect_forces`` of ``SPOSCAR`` to subtract its residual); elastic stresses: GPa, tension
positive, Voigt order xx yy zz yz xz xy of the cell after relaxing the ions at fixed cell.

A structure that is not relaxed yet (``relax`` asked for, not run) is skipped; ``relax`` *not* asked for -> ``CONTCAR`` is a copy
of the given ``POSCAR`` and everything is exported from it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil

import numpy as np
from ase import Atoms
from ase.io import write as ase_write

from heatup.core import ledger
from heatup.core.config import AUTO_MIN_LENGTH_A, cfg
from heatup.core.layout import (PhaseDir, read_json, read_structure, resolve_supercell, upper_triangular,
                                write_json, write_poscar)

log = logging.getLogger("heatup")


# ---------------------------------------------------------------------------
# the cells (shared with the compute steps)
# ---------------------------------------------------------------------------

def choose_supercell(atoms: Atoms) -> tuple[int, int, int]:
    """Phonon supercell: ``phonon.supercell`` if set, else perpendicular widths >= ``phonon.min_length_A``."""
    pc = cfg.phonon
    return resolve_supercell(atoms.cell.array, len(atoms), supercell=pc.supercell, min_length=pc.min_length_A,
                             max_atoms=pc.max_atoms, auto_min_length=AUTO_MIN_LENGTH_A["phonon"], what="phonon supercell")


def phonopy_object(atoms: Atoms, sc: tuple[int, int, int]):
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    pa = PhonopyAtoms(symbols=atoms.get_chemical_symbols(), cell=atoms.cell.array,
                      scaled_positions=atoms.get_scaled_positions())
    return Phonopy(pa, supercell_matrix=np.diag(sc), primitive_matrix="P",   # identity: n_bands = 3 n_atoms (phonopy >= 4 defaults to "auto")
                   symprec=cfg.phonon.symprec)


def to_ase(sc) -> Atoms:
    """A phonopy/phono3py supercell as ASE atoms (atom order preserved)."""
    return Atoms(sc.symbols, cell=sc.cell, scaled_positions=sc.scaled_positions, pbc=True)


def phonon_cells(atoms: Atoms, sc: tuple[int, int, int] | None = None):
    """``(phonopy object with displacements generated, supercell repetitions)`` of the relaxed *atoms*.
    ``sc``: use these repetitions instead of choosing them (QHA: the same ones at every volume)."""
    sc = tuple(sc) if sc is not None else choose_supercell(atoms)
    ph = phonopy_object(atoms, sc)
    ph.generate_displacements(distance=cfg.phonon.delta_A)
    return ph, sc


VOIGT_LABELS = ["xx", "yy", "zz", "yz", "xz", "xy"]


def voigt_strain(i: int, delta: float) -> np.ndarray:
    e = np.zeros((3, 3))
    if i < 3:
        e[i, i] = delta
    else:
        a, b = {3: (1, 2), 4: (0, 2), 5: (0, 1)}[i]
        e[a, b] = e[b, a] = delta / 2.0      # engineering shear strain = delta
    return e


def strained(atoms, eps: np.ndarray):
    out = atoms.copy()
    out.set_cell(atoms.cell.array @ (np.eye(3) + eps).T, scale_atoms=True)
    return out


def elastic_cells(atoms: Atoms, delta: float) -> dict[str, Atoms]:
    """The 12 strained cells, keyed ``<voigt>_plus`` / ``<voigt>_minus`` (xx_plus, ..., xy_minus)."""
    return {f"{lab}_{name}": strained(atoms, sign * voigt_strain(i, delta))
            for i, lab in enumerate(VOIGT_LABELS) for sign, name in ((+1.0, "plus"), (-1.0, "minus"))}


def _md_repetitions(unit, mcfg) -> tuple[int, int, int]:
    """Repetitions of the MD cell.  An explicit ``supercell`` or no ``max_atoms`` -> the common rule; otherwise the target
    width ``min_cell_A`` is lowered in steps of 1 A (down to ``min_cell_floor_A``) until the cell respects ``max_atoms``."""
    kw = dict(supercell=mcfg.supercell, max_atoms=mcfg.max_atoms, auto_min_length=AUTO_MIN_LENGTH_A["md"], what="MD supercell")
    if mcfg.supercell is not None or mcfg.min_cell_A is None or mcfg.max_atoms is None or mcfg.min_cell_floor_A is None:
        return resolve_supercell(unit.cell.array, len(unit), min_length=mcfg.min_cell_A, **kw)
    target = float(mcfg.min_cell_A)
    while True:
        try:
            reps = resolve_supercell(unit.cell.array, len(unit), min_length=target, **kw)
            if target < mcfg.min_cell_A:
                log.warning("MD supercell: target width lowered from %g to %g A to respect max_atoms=%d", mcfg.min_cell_A, target, mcfg.max_atoms)
            return reps
        except ValueError:
            if target - 1.0 < mcfg.min_cell_floor_A:
                raise ValueError(f"MD supercell would exceed max_atoms={mcfg.max_atoms} even at {mcfg.min_cell_floor_A:g} A: "
                                 "not suitable for MD with this cap")
            target -= 1.0


def build_md_cell(phase: PhaseDir, force: bool = False):
    """MD supercell from the relaxed cell: ``md.supercell`` if set, else perpendicular width >= ``md.min_cell_A``."""
    poscar = os.path.join(phase.md_dir, "POSCAR")
    info = os.path.join(phase.md_dir, "cell.json")
    mcfg = cfg.md
    request = {"supercell": None if mcfg.supercell is None else list(mcfg.supercell),
               "min_cell_A": mcfg.min_cell_A, "max_atoms": mcfg.max_atoms, "min_cell_floor_A": mcfg.min_cell_floor_A}
    if os.path.exists(poscar) and not force:
        old = read_json(info) or {}
        if old.get("request") == request and old.get("source") == ledger.file_hash(phase.contcar):
            return read_structure(poscar)
    unit = upper_triangular(read_structure(phase.contcar))
    reps = _md_repetitions(unit, mcfg)
    sc = unit.repeat(reps)
    os.makedirs(phase.md_dir, exist_ok=True)
    write_poscar(poscar, sc)
    write_json(info, {"repetitions": list(reps), "n_atoms_unit": len(unit),
                      "n_atoms": len(sc), "request": request,
                      "source": ledger.file_hash(phase.contcar)})
    log.info("MD supercell %s: reps=%s, %d atoms", phase.tag, reps, len(sc))
    return sc


def qha_scales() -> np.ndarray:
    """V/V0 of the QHA volumes."""
    qc = cfg.qha
    return np.linspace(1 - qc.volume_range, 1 + qc.volume_range, qc.n_volumes)


def scaled_cell(ref: Atoms, s: float) -> Atoms:
    """*ref* with its volume scaled by *s* (isotropic, fractional positions kept)."""
    a = ref.copy()
    a.set_cell(ref.cell.array * s ** (1 / 3), scale_atoms=True)
    return a


def fc3_supercell(atoms: Atoms) -> tuple[int, int, int]:
    c = cfg.fc3
    return resolve_supercell(atoms.cell.array, len(atoms), supercell=c.supercell, min_length=c.min_length_A,
                             max_atoms=c.max_atoms, auto_min_length=AUTO_MIN_LENGTH_A["fc3"], what="fc3 supercell")


def phono3py_object(atoms: Atoms, sc3, sc2):
    import warnings
    from phono3py import Phono3py
    from phonopy.structure.atoms import PhonopyAtoms
    pa = PhonopyAtoms(symbols=atoms.get_chemical_symbols(), cell=atoms.cell.array,
                      scaled_positions=atoms.get_scaled_positions())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                     # phono3py 4: primitive_matrix default notice
        return Phono3py(pa, supercell_matrix=np.diag(sc3),
                        phonon_supercell_matrix=np.diag(sc2) if sc2 is not None else None,
                        primitive_matrix="auto", symprec=cfg.fc3.symprec)


def fc3_cells(atoms: Atoms):
    """``(phono3py object with fc3 (and fc2) displacements generated, fc3 supercell, fc2 supercell or None)``."""
    c = cfg.fc3
    sc3 = fc3_supercell(atoms)
    sc2 = None
    if c.separate_fc2:
        sc2 = choose_supercell(atoms)
        if tuple(sc2) == tuple(sc3):
            sc2 = None
    ph = phono3py_object(atoms, sc3, sc2)
    ph.generate_displacements(distance=c.delta_A, cutoff_pair_distance=c.cutoff_pair_A)
    if sc2 is not None:
        ph.generate_fc2_displacements(distance=cfg.phonon.delta_A)
    return ph, sc3, sc2


# ---------------------------------------------------------------------------
# writing them out
# ---------------------------------------------------------------------------

def _write_cell(folder: str, atoms: Atoms) -> None:
    os.makedirs(folder, exist_ok=True)
    ase_write(os.path.join(folder, "POSCAR"), atoms, format="vasp", direct=True)       # atom order is kept (no sorting)


def _cells(d: str, prefix: str, cells, width: int) -> list[str]:
    """``<prefix>-NNN/POSCAR`` for every displaced cell (``None`` = excluded by a pair cutoff: skipped)."""
    files = []
    for i, sc in enumerate(cells, start=1):
        if sc is not None:
            _write_cell(os.path.join(d, f"{prefix}-{i:0{width}d}"), to_ase(sc))
            files.append(f"{prefix}-{i:0{width}d}/POSCAR")
    return files


def _phonons(phase: PhaseDir, atoms: Atoms, d: str, plan: ledger.Plan) -> list[str]:
    ph, _ = phonon_cells(atoms)
    ph.save(os.path.join(d, "phonopy_disp.yaml"))
    ase_write(os.path.join(d, "SPOSCAR"), to_ase(ph.supercell), format="vasp", direct=True)
    return ["phonopy_disp.yaml", "SPOSCAR", *_cells(d, "disp", ph.supercells_with_displacements, 3)]


def _qha(phase: PhaseDir, atoms: Atoms, d: str, plan: ledger.Plan) -> list[str]:
    scales = qha_scales()
    files = []
    for i, s in enumerate(scales, start=1):
        _write_cell(os.path.join(d, f"V{i:02d}"), scaled_cell(atoms, s))
        files.append(f"V{i:02d}/POSCAR")
    write_json(os.path.join(d, "volumes.json"), {"V_over_V0": scales.tolist(), "V0_A3": atoms.get_volume(),
               "note": "relax the ions at each fixed volume, then make phonopy displacements of the relaxed cell"})
    return ["volumes.json", *files]


def _fc3(phase: PhaseDir, atoms: Atoms, d: str, plan: ledger.Plan) -> list[str]:
    ph, _, sc2 = fc3_cells(atoms)
    ph.save(os.path.join(d, "phono3py_disp.yaml"))
    ase_write(os.path.join(d, "SPOSCAR"), to_ase(ph.supercell), format="vasp", direct=True)
    files = ["phono3py_disp.yaml", "SPOSCAR", *_cells(d, "disp", ph.supercells_with_displacements, 5)]
    if sc2 is not None:
        files += _cells(d, "disp_fc2", ph.phonon_supercells_with_displacements, 4)
    return files


def _elastic(phase: PhaseDir, atoms: Atoms, d: str, plan: ledger.Plan) -> list[str]:
    files = []
    for name, cell in elastic_cells(atoms, cfg.elastic.strain).items():
        _write_cell(os.path.join(d, name), cell)
        files.append(f"{name}/POSCAR")
    write_json(os.path.join(d, "strains.json"), {"strain": cfg.elastic.strain, "voigt_labels": VOIGT_LABELS,
               "note": "relax the ions at fixed cell, report the stress (GPa, tension positive, Voigt xx yy zz yz xz xy)"})
    return ["strains.json", *files]


def _md(phase: PhaseDir, atoms: Atoms, d: str, plan: ledger.Plan) -> list[str]:
    """The supercell as the starting structure of every planned run: ``md/<T>K/POSCAR`` (``md/POSCAR`` + ``cell.json`` are
    the cache the compute step reads)."""
    sc = build_md_cell(phase, force=True)
    files = []
    for T in plan.md_temperatures_K:
        sub = f"{int(round(T))}K"
        os.makedirs(os.path.join(d, sub), exist_ok=True)
        write_poscar(os.path.join(d, sub, "POSCAR"), sc)
        files.append(f"{sub}/POSCAR")
    return files


#: step -> (folder relative to the phase, writer)
EXPORTERS = {
    "phonons": (os.path.join("phonons", "ha"), _phonons),
    "qha": (os.path.join("phonons", "qha"), _qha),
    "fc3": (os.path.join("phonons", "fc3"), _fc3),
    "elastic": (os.path.join("elastic", "strain"), _elastic),
    "md": ("md", _md),
}


def stage_input(phase: PhaseDir, step: str, T: float | None = None) -> str | None:
    """Put the structure *step* starts from into its folder as ``POSCAR`` (``md``: the supercell, written by the export)."""
    if step == "md":
        return None
    src = phase.poscar if step == "relax" else phase.contcar
    if not os.path.exists(src):
        return None
    dst = os.path.join(ledger.STEPS[step].where(phase, T), "POSCAR")
    if not (os.path.exists(dst) and ledger.file_hash(dst) == ledger.file_hash(src)):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
    return dst


def _digest(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _rmdir_up(folder: str, stop: str) -> None:
    """Remove *folder* and its parents while they are empty, never *stop* or above."""
    folder, stop = os.path.abspath(folder), os.path.abspath(stop)
    while folder != stop and folder.startswith(stop + os.sep):
        try:
            os.rmdir(folder)
        except OSError:
            return
        folder = os.path.dirname(folder)


def _export(phase: PhaseDir, plan: ledger.Plan, step: str, atoms: Atoms) -> dict:
    rel_dir, writer = EXPORTERS[step]
    spec = ledger.STEPS[step]
    d = os.path.join(phase.path, rel_dir)
    keys = set(ledger.plan_options([step]))
    key = {"source": ledger.file_hash(phase.contcar),
           "options": _digest({k: v for k, v in sorted(plan.options.items()) if k in keys}),
           "temperatures": sorted(plan.md_temperatures_K) if spec.per_T else None}
    home = [T for s, T in plan.tasks() if s == step][0]               # the record that carries the export section
    old = ledger.get_export(phase, step, home)
    if old and old.get("key") == key and all(os.path.exists(os.path.join(phase.path, f)) for f in old["files"]):
        return {"phase": phase.tag, "step": step, "dir": rel_dir, "n_files": len(old["files"]), "state": "up to date"}
    for f in (old or {}).get("files", []):                       # only what HeatUp wrote before; user files are never touched
        p = os.path.join(phase.path, f)
        if os.path.exists(p):
            os.remove(p)
        _rmdir_up(os.path.dirname(p), phase.path)
    os.makedirs(d, exist_ok=True)
    try:
        with ledger.applied(plan):
            files = [os.path.join(rel_dir, f) for f in writer(phase, atoms, d, plan)]
    except BaseException:                                        # no empty folder is left behind
        _rmdir_up(d, phase.path)
        raise
    ledger.set_export(phase, step, home, {"key": key, "dir": rel_dir, "files": files})
    return {"phase": phase.tag, "step": step, "dir": rel_dir, "n_files": len(files),
            "state": "updated" if old else "written"}


def export_inputs(prepared=None, steps: list[str] | None = None) -> list[dict]:
    """Write the cells of every planned step of every phase (default: all phases of the database with a plan, or those of
    *prepared*), from ``relaxation/CONTCAR``.  Phases whose structure is taken as given (``relax`` not asked for) get
    ``CONTCAR`` from ``POSCAR`` first.  Returns one row per (phase, step): ``{phase, step, dir, n_files, state}``."""
    from heatup.core.layout import iter_phase_dirs
    if prepared is not None:
        dirs = list(prepared.targets) + [c.phase_dir for c in prepared.competitors]
    else:
        dirs = list(iter_phase_dirs(cfg.phases.database_root))
    rows: list[dict] = []
    for pd in dirs:
        phase = PhaseDir.of(pd)
        plan = ledger.read_plan(phase)
        if plan is None:
            continue
        if not os.path.exists(phase.contcar) and plan.options.get("relax.optimize") is False:
            os.makedirs(phase.relax_dir, exist_ok=True)
            write_poscar(phase.contcar, read_structure(phase.poscar))          # taken as relaxed: no calculation
        if not os.path.exists(phase.contcar):
            rows.append({"phase": phase.tag, "step": "*", "dir": "", "n_files": 0,
                         "state": "skipped: relaxation/CONTCAR missing (run the relaxation first)"})
            continue
        atoms = read_structure(phase.contcar)
        for step, T in plan.tasks():                             # the structure each calculation starts from, in its folder
            stage_input(phase, step, T)
        for step in plan.steps:
            if step in EXPORTERS and (steps is None or step in steps):
                try:
                    rows.append(_export(phase, plan, step, atoms))
                except Exception as exc:  # noqa: BLE001  one phase's cell problem (e.g. max_atoms) never stops the others
                    log.warning("%s: export of %s failed: %s", phase.tag, step, exc)
                    rows.append({"phase": phase.tag, "step": step, "dir": EXPORTERS[step][0], "n_files": 0,
                                 "state": f"failed: {exc}"})
    return rows
