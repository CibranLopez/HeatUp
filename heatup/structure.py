"""heatup.structure
====================
Structure helpers shared by all steps (previously duplicated in 3 modules).
"""

from __future__ import annotations

import logging

import numpy as np
from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write


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


def supercell_reps(cell: np.ndarray, min_length: float, max_atoms: int | None = None,
                   n_atoms_cell: int = 1) -> tuple[int, int, int]:
    """Smallest (na,nb,nc) with *perpendicular* widths >= min_length.

    v0.2 used the lattice-vector *lengths*, which for a non-orthogonal cell
    (hexagonal, rhombohedral, triclinic) overestimates the distance between
    periodic images and gives images closer than intended.
    """
    vol = abs(np.linalg.det(cell))
    widths = np.array([vol / np.linalg.norm(np.cross(cell[(i + 1) % 3], cell[(i + 2) % 3]))
                       for i in range(3)])
    reps = np.maximum(np.ceil(min_length / widths - 1e-9).astype(int), 1)
    if max_atoms is not None and reps.prod() * n_atoms_cell > max_atoms:
        raise ValueError(
            f"supercell {tuple(reps)} would have {reps.prod() * n_atoms_cell} atoms "
            f"(> max_atoms={max_atoms}); reduce min length or raise max_atoms")
    return int(reps[0]), int(reps[1]), int(reps[2])


log = logging.getLogger("heatup")


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


def perpendicular_widths(cell: np.ndarray) -> np.ndarray:
    vol = abs(np.linalg.det(cell))
    return np.array([vol / np.linalg.norm(np.cross(cell[(i + 1) % 3], cell[(i + 2) % 3]))
                     for i in range(3)])


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
