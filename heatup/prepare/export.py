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

Every export folder has an ``export_<step>.json`` (what was written, from which CONTCAR hash, with which options): exporting
again is a no-op unless the structure or the options changed, and only the files HeatUp wrote are ever replaced.

Bringing results back (anything that writes the files notebook 3 reads is fine): ``heatup.collect_ha`` turns forces of the
``phonons/ha`` cells into ``ha.json``, ``heatup.collect_elastic`` turns stresses of the strained cells into ``elastic.json``;
trajectories and energies are files in the layout of ``heatup.core.layout`` (see the last cell of notebook 3).  Phonon forces:
eV/A of every ``disp-*`` cell (and ``perfect_forces`` of ``SPOSCAR`` to subtract its residual); elastic stresses: GPa, tension
positive, Voigt order xx yy zz yz xz xy of the cell after relaxing the ions at fixed cell.

A structure that is not relaxed yet (``relax`` asked for, not run) is skipped; ``relax`` *not* asked for -> ``CONTCAR`` is a copy
of the given ``POSCAR`` and everything is exported from it.
"""

from __future__ import annotations

import logging
import os

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


def phonon_cells(atoms: Atoms):
    """``(phonopy object with displacements generated, supercell repetitions)`` of the relaxed *atoms*."""
    sc = choose_supercell(atoms)
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


def build_md_cell(phase: PhaseDir, force: bool = False):
    """MD supercell from the relaxed cell: ``md.supercell`` if set, else perpendicular width >= ``md.min_cell_A``."""
    poscar = os.path.join(phase.md_dir, "POSCAR")
    info = os.path.join(phase.md_dir, "cell.json")
    mcfg = cfg.md
    request = {"supercell": None if mcfg.supercell is None else list(mcfg.supercell),
               "min_cell_A": mcfg.min_cell_A, "max_atoms": mcfg.max_atoms}
    if os.path.exists(poscar) and not force:
        old = read_json(info) or {}
        if old.get("request") == request and old.get("source") == ledger.file_hash(phase.contcar):
            return read_structure(poscar)
    unit = upper_triangular(read_structure(phase.contcar))
    reps = resolve_supercell(unit.cell.array, len(unit), supercell=mcfg.supercell, min_length=mcfg.min_cell_A,
                             max_atoms=mcfg.max_atoms, auto_min_length=AUTO_MIN_LENGTH_A["md"], what="MD supercell")
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


def _phonons(phase: PhaseDir, atoms: Atoms, d: str) -> list[str]:
    ph, _ = phonon_cells(atoms)
    ph.save(os.path.join(d, "phonopy_disp.yaml"))
    ase_write(os.path.join(d, "SPOSCAR"), to_ase(ph.supercell), format="vasp", direct=True)
    return ["phonopy_disp.yaml", "SPOSCAR", *_cells(d, "disp", ph.supercells_with_displacements, 3)]


def _qha(phase: PhaseDir, atoms: Atoms, d: str) -> list[str]:
    scales = qha_scales()
    files = []
    for i, s in enumerate(scales, start=1):
        _write_cell(os.path.join(d, f"V{i:02d}"), scaled_cell(atoms, s))
        files.append(f"V{i:02d}/POSCAR")
    write_json(os.path.join(d, "volumes.json"), {"V_over_V0": scales.tolist(), "V0_A3": atoms.get_volume(),
               "note": "relax the ions at each fixed volume, then make phonopy displacements of the relaxed cell"})
    return ["volumes.json", *files]


def _fc3(phase: PhaseDir, atoms: Atoms, d: str) -> list[str]:
    ph, _, sc2 = fc3_cells(atoms)
    ph.save(os.path.join(d, "phono3py_disp.yaml"))
    ase_write(os.path.join(d, "SPOSCAR"), to_ase(ph.supercell), format="vasp", direct=True)
    files = ["phono3py_disp.yaml", "SPOSCAR", *_cells(d, "disp", ph.supercells_with_displacements, 5)]
    if sc2 is not None:
        files += _cells(d, "disp_fc2", ph.phonon_supercells_with_displacements, 4)
    return files


def _elastic(phase: PhaseDir, atoms: Atoms, d: str) -> list[str]:
    files = []
    for name, cell in elastic_cells(atoms, cfg.elastic.strain).items():
        _write_cell(os.path.join(d, name), cell)
        files.append(f"{name}/POSCAR")
    write_json(os.path.join(d, "strains.json"), {"strain": cfg.elastic.strain, "voigt_labels": VOIGT_LABELS,
               "note": "relax the ions at fixed cell, report the stress (GPa, tension positive, Voigt xx yy zz yz xz xy)"})
    return ["strains.json", *files]


def _md(phase: PhaseDir, atoms: Atoms, d: str) -> list[str]:
    build_md_cell(phase, force=True)
    return ["POSCAR", "cell.json"]


#: step -> (folder relative to the phase, writer)
EXPORTERS = {
    "phonons": (os.path.join("phonons", "ha"), _phonons),
    "qha": (os.path.join("phonons", "qha"), _qha),
    "fc3": (os.path.join("phonons", "fc3"), _fc3),
    "elastic": (os.path.join("elastic", "strain"), _elastic),
    "md": ("md", _md),
}


def _export(phase: PhaseDir, plan: ledger.Plan, step: str, atoms: Atoms) -> dict:
    rel_dir, writer = EXPORTERS[step]
    d = os.path.join(phase.path, rel_dir)
    keys = set(ledger.plan_options([step]))
    key = {"source": ledger.file_hash(phase.contcar),
           "options": {k: v for k, v in sorted(plan.options.items()) if k in keys}}
    meta_f = os.path.join(d, f"export_{step}.json")
    old = read_json(meta_f)
    if old and old.get("key") == key and all(os.path.exists(os.path.join(d, f)) for f in old["files"]):
        return {"phase": phase.tag, "step": step, "dir": rel_dir, "n_files": len(old["files"]), "state": "up to date"}
    for f in (old or {}).get("files", []):                       # only what HeatUp wrote before; user files are never touched
        p = os.path.join(d, f)
        if os.path.exists(p):
            os.remove(p)
        try:
            os.removedirs(os.path.dirname(p))
        except OSError:
            pass
    os.makedirs(d, exist_ok=True)
    with ledger.applied(plan):
        files = writer(phase, atoms, d)
    write_json(meta_f, {"key": key, "files": files})
    for s, T in plan.tasks():                                    # tell the task file(s) of this step
        if s == step:
            tf = os.path.join(ledger.STEPS[s].where(phase, T), f"task_{s}.json")
            task = read_json(tf)
            if task is not None:
                task["exported"] = {"dir": rel_dir, "n_files": len(files), "details": os.path.join(rel_dir, f"export_{step}.json")}
                write_json(tf, task)
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
        for step in plan.steps:
            if step in EXPORTERS and (steps is None or step in steps):
                try:
                    rows.append(_export(phase, plan, step, atoms))
                except Exception as exc:  # noqa: BLE001  one phase's cell problem (e.g. max_atoms) never stops the others
                    log.warning("%s: export of %s failed: %s", phase.tag, step, exc)
                    rows.append({"phase": phase.tag, "step": step, "dir": EXPORTERS[step][0], "n_files": 0,
                                 "state": f"failed: {exc}"})
    return rows
