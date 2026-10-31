"""heatup.core.layout
=====================
Canonical on-disk layout of one phase, JSON I/O, and the structure helpers shared by every step.

::

    <root>/<material>/<label>/            = PhaseDir
        POSCAR                            input structure (never modified)
        metadata.json                     provenance (MP id, origin, generator) - written by ``prepare``
        plan.json                         WHAT to compute here and with which options - written by ``prepare``
        relaxation/   CONTCAR energy.json run.traj          (step ``relax``)
        elastic/      elastic.json                          (step ``elastic``)
        phonons/      ha.json [qha.json] [fc3.json] phonon.png
        md/           POSCAR (supercell) cell.json
            <T>K/     output.traj md.json                   (step ``md``; ``md.json`` is written last = run complete)
                      mdstats.json vdos.json transition.json   (analysis steps ``mdstats vdos transition``)
        electronic/ magnetic/ disorder/                     optional data files (see ``analyze.thermo``)
        stability/    report_<stage>.json report_<stage>.png hull_vs_T.json zero_K_hull.json ...
        <step dir>/   POSCAR (the structure it starts from)  step_<step>.json (task / run / result / export: ``core.ledger``)

Two roots only.  ``cfg.phases.candidates_root`` holds *structures* (yours, Materials Project downloads, generated
ones); ``cfg.phases.database_root`` holds everything that is *computed* - targets and competing phases alike, told
apart by the ``role`` in their ``plan.json``.  A structure is copied from candidates to the database when it is needed.

``md/`` is classical MLIP MD (v0.2 called it the misleading ``aimd/``).
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write

log = logging.getLogger("heatup")


# ---------------------------------------------------------------------------
# JSON helpers and PhaseDir
# ---------------------------------------------------------------------------

def read_json(path: str, default: Any = None) -> Any:
    if not os.path.exists(path):
        return default
    try:
        with open(path) as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return default


class _NumpyEncoder(json.JSONEncoder):
    def default(self, o):  # noqa: D401
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (set, tuple)):
            return list(o)
        return super().default(o)


def write_json(path: str, obj: Any, indent: int | None = 2) -> str:
    """Atomic JSON write (temp file + rename) - a killed job never leaves a half file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(obj, fh, indent=indent, cls=_NumpyEncoder)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return path


class PhaseDir:
    """Path helper for one ``<material>/<label>`` directory."""

    def __init__(self, path: str | "PhaseDir"):
        if isinstance(path, PhaseDir):
            path = path.path
        self.path = os.path.abspath(path)
        self.label = os.path.basename(self.path)
        self.material = os.path.basename(os.path.dirname(self.path))

    @classmethod
    def of(cls, x: "str | PhaseDir") -> "PhaseDir":
        return x if isinstance(x, PhaseDir) else cls(x)

    def __repr__(self) -> str:
        return f"PhaseDir({self.tag})"

    def __fspath__(self) -> str:
        return self.path

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PhaseDir) and other.path == self.path

    def __hash__(self) -> int:
        return hash(self.path)

    # -- names --------------------------------------------------------------
    @property
    def tag(self) -> str:
        return f"{self.material}/{self.label}"

    def join(self, *parts: str) -> str:
        return os.path.join(self.path, *parts)

    # -- input / relaxation -------------------------------------------------
    @property
    def poscar(self) -> str:
        return self.join("POSCAR")

    @property
    def plan_json(self) -> str:
        return self.join("plan.json")

    @property
    def metadata_json(self) -> str:
        return self.join("metadata.json")

    @property
    def relax_dir(self) -> str:
        return self.join("relaxation")

    @property
    def contcar(self) -> str:
        return self.join("relaxation", "CONTCAR")

    @property
    def energy_json(self) -> str:
        return self.join("relaxation", "energy.json")

    # -- other steps ----------------------------------------------------------
    @property
    def phonon_dir(self) -> str:
        return self.join("phonons")

    @property
    def elastic_dir(self) -> str:
        return self.join("elastic")

    @property
    def elastic_json(self) -> str:
        return self.join("elastic", "elastic.json")

    @property
    def md_dir(self) -> str:
        return self.join("md")

    def md_run(self, T: float) -> str:
        return os.path.join(self.md_dir, f"{int(round(T))}K")

    @property
    def stability_dir(self) -> str:
        return self.join("stability")

    # -- queries --------------------------------------------------------------
    def has_relaxation(self) -> bool:
        e = read_json(self.energy_json)
        return bool(e) and bool(e.get("converged", False)) and os.path.exists(self.contcar)

    def md_temperatures(self, complete: bool = True) -> list[int]:
        """Temperatures (K) of the MD runs on disk.  ``complete=True``: finished runs only (``output.traj`` and
        ``md.json``, which ``run_md`` writes last); ``False``: every ``<T>K`` directory holding a trajectory."""
        out: list[int] = []
        if not os.path.isdir(self.md_dir):
            return out
        for name in os.listdir(self.md_dir):
            m = re.fullmatch(r"(\d+)K", name)
            if not m:
                continue
            tr = os.path.join(self.md_dir, name, "output.traj")
            if not (os.path.exists(tr) and os.path.getsize(tr) > 0):
                continue
            if complete and not os.path.exists(os.path.join(self.md_dir, name, "md.json")):
                continue
            out.append(int(m.group(1)))
        return sorted(out)

    def energy(self) -> dict | None:
        return read_json(self.energy_json)


def iter_phase_dirs(root: str):
    """Yield every ``PhaseDir`` (``root/<material>/<label>`` with a POSCAR)."""
    if not os.path.isdir(root):
        return
    for mat in sorted(os.listdir(root)):
        md = os.path.join(root, mat)
        if not os.path.isdir(md):
            continue
        for lab in sorted(os.listdir(md)):
            pd = os.path.join(md, lab)
            if os.path.isfile(os.path.join(pd, "POSCAR")):
                yield PhaseDir(pd)


# ---------------------------------------------------------------------------
# structures, symmetry, supercells
# ---------------------------------------------------------------------------

def read_structure(path: str) -> Atoms:
    return ase_read(path)


def write_poscar(path: str, atoms: Atoms) -> None:
    ase_write(path, atoms, format="vasp", direct=True, sort=True)


def spacegroup(atoms: Atoms, symprec: float = 1e-3) -> tuple[int, str]:
    """(ITA number, Hermann-Mauguin symbol) or (0, '?')."""
    try:
        import spglib
        ds = spglib.get_symmetry_dataset(
            (atoms.cell.array, atoms.get_scaled_positions(), atoms.numbers), symprec=symprec)
        if ds is None:
            return 0, "?"
        num = ds.number if hasattr(ds, "number") else ds["number"]
        sym = ds.international if hasattr(ds, "international") else ds["international"]
        return int(num), str(sym)
    except Exception:  # noqa: BLE001
        return 0, "?"


def trajectory_frames(path: str) -> int:
    """Number of frames of an ASE trajectory file (0 if missing or unreadable)."""
    try:
        from ase.io.trajectory import Trajectory
        with Trajectory(path) as t:
            return len(t)
    except Exception:  # noqa: BLE001
        return 0


def upper_triangular(atoms: Atoms) -> Atoms:
    """Same lattice (metric tensor), upper-triangular matrix, as required by ASE's NPT.

    With row-vector convention the NPT integrator needs h[1,0]=h[2,0]=h[2,1]=0
    (as stated by ASE).  Built from the Cholesky-like decomposition of the
    metric G = A A^T.  Fractional coordinates are preserved.
    """
    A = atoms.cell.array
    if A[1, 0] == 0.0 and A[2, 0] == 0.0 and A[2, 1] == 0.0:       # exact: ASE tests with ==, not a tolerance
        return atoms.copy()
    G = A @ A.T
    f = np.sqrt(G[2, 2])
    e = G[1, 2] / f
    d = np.sqrt(max(G[1, 1] - e * e, 0.0))
    c = G[0, 2] / f
    b = (G[0, 1] - c * e) / d
    a = np.sqrt(max(G[0, 0] - b * b - c * c, 0.0))
    new = atoms.copy()
    new.set_cell(np.array([[a, b, c], [0, d, e], [0, 0, f]]), scale_atoms=True)
    return new


def perpendicular_widths(cell: np.ndarray) -> np.ndarray:
    """Distances between opposite faces of the cell (A) - the periodic-image distance along each axis."""
    vol = abs(np.linalg.det(cell))
    return np.array([vol / np.linalg.norm(np.cross(cell[(i + 1) % 3], cell[(i + 2) % 3])) for i in range(3)])


def supercell_reps(cell: np.ndarray, min_length: float, max_atoms: int | None = None,
                   n_atoms_cell: int = 1) -> tuple[int, int, int]:
    """Smallest (na,nb,nc) with *perpendicular* widths >= min_length.

    v0.2 used the lattice-vector *lengths*, which for a non-orthogonal cell
    (hexagonal, rhombohedral, triclinic) overestimates the distance between
    periodic images and gives images closer than intended.
    """
    widths = perpendicular_widths(cell)
    reps = np.maximum(np.ceil(min_length / widths - 1e-9).astype(int), 1)
    if max_atoms is not None and reps.prod() * n_atoms_cell > max_atoms:
        raise ValueError(
            f"supercell {tuple(reps)} would have {reps.prod() * n_atoms_cell} atoms "
            f"(> max_atoms={max_atoms}); reduce min length or raise max_atoms")
    return int(reps[0]), int(reps[1]), int(reps[2])


def resolve_supercell(cell: np.ndarray, n_atoms_cell: int, *, supercell=None, min_length=None, max_atoms=None,
                      auto_min_length: float, what: str = "supercell") -> tuple[int, int, int]:
    """The one rule behind every ``supercell / min_length_A / max_atoms`` triple of the configuration.

    * ``supercell`` given  -> used as is; ``min_length`` is not needed (a value there is ignored, with a warning).
    * ``supercell is None`` -> smallest repetition whose perpendicular widths are >= ``min_length``; when
      ``min_length`` is ``None`` too, the built-in ``auto_min_length`` of that step is used.
    * ``max_atoms is None`` -> no limit; otherwise exceeding it raises (also for an explicit ``supercell``).
    """
    if supercell is not None:
        reps = tuple(int(x) for x in supercell)
        if len(reps) != 3 or min(reps) < 1:
            raise ValueError(f"{what}: supercell must be three positive integers, got {supercell!r}")
        if min_length is not None:
            log.warning("%s: min_length is ignored because an explicit supercell %s is set (set it to None)",
                        what, reps)
        n = int(np.prod(reps)) * n_atoms_cell
        if max_atoms is not None and n > max_atoms:
            raise ValueError(f"{what}: explicit supercell {reps} has {n} atoms (> max_atoms={max_atoms})")
        return reps  # type: ignore[return-value]
    return supercell_reps(cell, auto_min_length if min_length is None else min_length, max_atoms, n_atoms_cell)


def to_pymatgen(atoms: Atoms):
    from pymatgen.io.ase import AseAtomsAdaptor
    return AseAtomsAdaptor().get_structure(atoms)


def from_pymatgen(structure) -> Atoms:
    from pymatgen.io.ase import AseAtomsAdaptor
    return AseAtomsAdaptor().get_atoms(structure)


def composition(atoms: Atoms) -> dict[str, float]:
    """Atomic fractions {element: x}."""
    syms = atoms.get_chemical_symbols()
    return {s: syms.count(s) / len(syms) for s in sorted(set(syms))}


def reduced_formula(atoms: Atoms) -> str:
    from pymatgen.core import Composition
    return Composition(atoms.get_chemical_formula()).reduced_formula
