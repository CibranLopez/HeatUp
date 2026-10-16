"""heatup.paths
================
Canonical on-disk layout of one phase plus small I/O helpers.

::

    <root>/<material>/<label>/            = PhaseDir
        POSCAR                            input structure (never modified)
        relaxation/   CONTCAR energy.json manifest.json
        phonons/      ha.json [qha.json] band.json phonon.png manifest.json
        elastic/      elastic.json manifest.json
        md/           POSCAR (supercell) cell.json
            <T>K/     output.traj md.json analysis.json vdos.json
                      transition.json manifest.json
        electronic/ magnetic/ disorder/   optional data files (see free_energy)
        stability/    report.json report.png hull.json

``md/`` replaces the misleading ``aimd/`` of v0.2 (it is classical MLIP MD,
not ab-initio MD).
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from typing import Any

import numpy as np


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

    def md_temperatures(self, with_traj: bool = True) -> list[int]:
        out: list[int] = []
        if not os.path.isdir(self.md_dir):
            return out
        for name in os.listdir(self.md_dir):
            m = re.fullmatch(r"(\d+)K", name)
            if not m:
                continue
            if with_traj:
                tr = os.path.join(self.md_dir, name, "output.traj")
                if not (os.path.exists(tr) and os.path.getsize(tr) > 0):
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
