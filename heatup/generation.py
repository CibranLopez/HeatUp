"""heatup.generation
=====================
Candidate-structure generators (the optional "generative" front end).

Generators are plain functions ``fn(elements_counts: dict[str,int], n: int, seed: int) ->
list[pymatgen.Structure]`` registered by name; the built-ins are

``pyxtal``   random symmetric crystals (PyXtal) - space-group sampled, not exhaustive
``matgen``   MatterGen CLI (``cfg.generation.matgen_command``) - needs the
             ``mattergen`` package/checkpoints; **untested in the sandbox**, the
             wrapper only builds the command line and parses the CIF archive
``custom``   anything you register with :func:`register_generator`

Every structure is written to ``<candidates_root>/<formula>/<sg>_<hash>/POSCAR`` with a
``metadata.json`` recording the generator; nothing is considered a result until it has
been relaxed and has passed the 0 K pre-screen (:func:`heatup.workflow.screen`).

Differences from v0.2: no exhaustive 230-group sweep per sub-composition (hours, mostly
overlapping junk): ``n_per_composition`` samples per stoichiometry, seeded and
reproducible; stoichiometries are enumerated by atom budget; duplicates are removed
with StructureMatcher before anything is relaxed.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import random
import subprocess
import tempfile
import zipfile
from itertools import product
from typing import Callable

from heatup.config import cfg
from heatup.paths import write_json
from heatup.phases import Phase, deduplicate, phase_from_dir, sanitize_label, subsystems
from heatup.structure import from_pymatgen, spacegroup, write_poscar

log = logging.getLogger("heatup")

Generator = Callable[[dict, int, int], list]
_REGISTRY: dict[str, Generator] = {}


def register_generator(name: str, fn: Generator) -> None:
    _REGISTRY[name] = fn


# ---------------------------------------------------------------------------
# stoichiometry enumeration
# ---------------------------------------------------------------------------

def enumerate_stoichiometries(elements, max_atoms: int, max_ratio: int = 6) -> list[dict]:
    """Reduced integer compositions (gcd = 1) of *elements*, each element count <= max_ratio,
    with formula-unit total <= max_atoms.  Single elements -> ``{El: 1}``."""
    els = sorted(elements)
    out = []
    for counts in product(range(1, max_ratio + 1), repeat=len(els)):
        if math.gcd(*counts) != 1 or sum(counts) > max_atoms:
            continue
        out.append(dict(zip(els, counts)))
    return out


def cell_multiplicities(n_fu_atoms: int, max_atoms: int) -> list[int]:
    return [z for z in (1, 2, 3, 4) if z * n_fu_atoms <= max_atoms]


# ---------------------------------------------------------------------------
# built-in generators
# ---------------------------------------------------------------------------

def _pyxtal(counts: dict, n: int, seed: int) -> list:
    from pyxtal import pyxtal
    rng = random.Random(seed)
    sgs = list(cfg.generation.space_groups or range(2, 231))
    els = sorted(counts)
    zs = cell_multiplicities(sum(counts.values()), cfg.generation.max_atoms)
    out, tries = [], 0
    while len(out) < n and tries < 40 * n:
        tries += 1
        z, sg = rng.choice(zs), rng.choice(sgs)
        st = pyxtal()
        try:
            st.from_random(3, sg, els, [counts[e] * z for e in els], random_state=rng.randrange(2**31))
        except Exception:  # noqa: BLE001
            continue
        if getattr(st, "valid", False):
            try:
                out.append(st.to_pymatgen())
            except Exception:  # noqa: BLE001
                pass
    return out


def _matgen(counts: dict, n: int, seed: int) -> list:
    from pymatgen.core import Structure
    g = cfg.generation
    with tempfile.TemporaryDirectory() as td:
        cmd = [g.matgen_command, td, f"--pretrained-name={g.matgen_model}", f"--batch_size={n}",
               "--num_batches=1", *map(str, g.matgen_extra_args)]
        subprocess.run(cmd, check=True)
        zf = os.path.join(td, "generated_crystals_cif.zip")
        structs = []
        with zipfile.ZipFile(zf) as z:
            for name in sorted(z.namelist()):
                if name.endswith(".cif"):
                    structs.append(Structure.from_str(z.read(name).decode(), fmt="cif"))
    allowed = set(counts)
    return [s for s in structs if {str(e) for e in s.composition.elements} <= allowed]


register_generator("pyxtal", _pyxtal)
register_generator("matgen", _matgen)


# ---------------------------------------------------------------------------
# writing / driver
# ---------------------------------------------------------------------------

def _digest(structure) -> str:
    key = f"{structure.composition.reduced_formula}{structure.volume:.2f}{len(structure)}"
    return hashlib.sha1(key.encode()).hexdigest()[:8]


def write_candidate(structure, root: str, source: str) -> Phase | None:
    atoms = from_pymatgen(structure)
    _, sg = spacegroup(atoms, symprec=0.05)
    formula = structure.composition.reduced_formula
    d = os.path.join(root, formula, f"{sanitize_label(sg)}_{source}{_digest(structure)}")
    if os.path.exists(os.path.join(d, "POSCAR")):
        return phase_from_dir(d, source)
    os.makedirs(d, exist_ok=True)
    write_poscar(os.path.join(d, "POSCAR"), atoms)
    write_json(os.path.join(d, "metadata.json"), {"formula": formula, "symmetry": sg,
                                                  "source": source, "generator": cfg.generation.method})
    return phase_from_dir(d, source)


def generate(counts: dict, n: int | None = None, root: str | None = None,
             method: str | None = None, seed: int | None = None) -> list[Phase]:
    """Generate *n* structures for the integer composition *counts*; return de-duplicated Phases."""
    g = cfg.generation
    method = method or g.method
    if method not in _REGISTRY:
        raise KeyError(f"unknown generator {method!r}; registered: {sorted(_REGISTRY)}")
    structs = _REGISTRY[method](counts, n or g.n_per_composition, g.seed if seed is None else seed)
    root = root or cfg.phases.candidates_root
    phases = [p for p in (write_candidate(s, root, "gen") for s in structs) if p]
    return deduplicate(phases)


def generate_for_subsystems(elements, root: str) -> list[Phase]:
    """Candidates for every stoichiometry of every chemical subsystem of *elements*."""
    out: list[Phase] = []
    for sub in subsystems(elements):
        for counts in enumerate_stoichiometries(sub, cfg.generation.max_atoms):
            out += generate(counts, root=root)
    log.info("generated %d candidate structure(s)", len(out))
    return out
