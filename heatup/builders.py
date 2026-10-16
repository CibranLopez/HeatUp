"""heatup.builders
===================
Small helpers that create phase directories (``<root>/<material>/<label>/POSCAR``) from ASE atoms or
from an input tree - so notebooks do not have to define them.
"""

from __future__ import annotations

import os
import shutil

from ase import Atoms
from ase.build import bulk

from heatup.config import cfg
from heatup.paths import PhaseDir
from heatup.structure import write_poscar


def write_phase(root: str, material: str, label: str, atoms: Atoms, overwrite: bool = False) -> PhaseDir:
    """Write ``atoms`` as ``<root>/<material>/<label>/POSCAR`` (kept if it already exists)."""
    d = os.path.join(root, material, label)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "POSCAR")
    if overwrite or not os.path.exists(p):
        write_poscar(p, atoms)
    return PhaseDir(d)


def locate_phase(material: str, label: str, database_root: str | None = None,
                 candidates_root: str | None = None) -> PhaseDir | None:
    """The phase directory in the database; if only a candidate POSCAR exists it is *copied* into the
    database (the candidates tree is never written).  ``None`` if neither exists."""
    database_root = database_root or cfg.phases.database_root
    candidates_root = candidates_root or cfg.phases.candidates_root
    dst = os.path.join(database_root, material, label)
    if os.path.exists(os.path.join(dst, "POSCAR")):
        return PhaseDir(dst)
    src = os.path.join(candidates_root, material, label, "POSCAR")
    if os.path.exists(src):
        os.makedirs(dst, exist_ok=True)
        shutil.copyfile(src, os.path.join(dst, "POSCAR"))
        return PhaseDir(dst)
    return None


def fcc(symbol: str, a: float) -> Atoms:
    """Conventional 4-atom fcc cell."""
    return bulk(symbol, "fcc", a=a, cubic=True)


def bcc(symbol: str, a: float) -> Atoms:
    return bulk(symbol, "bcc", a=a, cubic=True)


def l1_0(a_symbol: str, b_symbol: str, a: float) -> Atoms:
    """Ordered AB compound (CuAu-I type, P4/mmm): alternating (001) layers, 4-atom cell."""
    x = fcc(a_symbol, a)
    x.set_chemical_symbols([a_symbol, b_symbol, b_symbol, a_symbol])
    return x


def l1_2(a_symbol: str, b_symbol: str, a: float) -> Atoms:
    """Ordered A3B compound (Cu3Au type, Pm-3m): B at the cube corner, A on the face centres."""
    x = fcc(a_symbol, a)
    x[0].symbol = b_symbol
    return x
