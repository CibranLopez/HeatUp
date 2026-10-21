"""heatup.prepare.phases
========================
Everything that puts structures on disk: building and locating target structures, and discovering the
*competing phases* that span the convex hull.

Two roots, nothing else
-----------------------
``candidates_root``  structures only: yours, Materials Project downloads, generated candidates.  Never computed on.
``database_root``    everything that is computed - targets and competing phases alike.

A competing phase is found in either root.  One already in the database keeps its plan and results (whether they may
be used is decided by comparing their settings with the project's: :func:`heatup.core.ledger.plan_conflicts`); one found
in ``candidates_root`` is **copied** into the database (:func:`promote`, same ``<formula>/<label>``) and prepared like
any other phase.  Candidates stay where they are, so inputs are never altered and the Materials-Project cache survives.

Search order (priority for de-duplication: the first structure wins, so calculations are never duplicated)

``database``    ``<database_root>/<formula>/<label>/POSCAR``  (targets, earlier competitors, your own calculations)
``candidates``  ``<candidates_root>/<formula>/<label>/POSCAR``
``mp-api``      Materials Project: *every* chemical subsystem of the target within ``mp_max_e_above_hull`` of MP's own
                hull, capped per subsystem; structures and the query index are cached under ``candidates_root``
                (v0.2 queried ``elements=[...]``, which only returns phases that contain *all* of them, so binaries and
                elements were missing and the hull was wrong).
``generate``    :mod:`heatup.prepare.generation` (random/AI structures), written to ``candidates_root``.

``local_only=True`` (what the analysis stage uses) reads the database only: no network, no copying, no generation.

Notes: space-group symbols such as ``P4/mmm`` are sanitised before use as directory names; the target's own structure
is excluded from its competitors (it would trivially sit on the hull).
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from dataclasses import dataclass
from itertools import chain, combinations

from ase import Atoms
from ase.build import bulk

from heatup.core import ledger
from heatup.core.config import cfg
from heatup.core.layout import (PhaseDir, composition, read_json, read_structure, reduced_formula, to_pymatgen,
                                write_json, write_poscar)

log = logging.getLogger("heatup")


# ---------------------------------------------------------------------------
# building / locating target structures
# ---------------------------------------------------------------------------

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
    database_root   = database_root   or cfg.phases.database_root
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


log = logging.getLogger("heatup")


@dataclass
class Phase:
    phase_dir: PhaseDir
    formula: str
    composition: dict            # atomic fractions
    elements: frozenset
    source: str
    mp_id: str | None = None

    @property
    def tag(self) -> str:
        return self.phase_dir.tag

    def to_dict(self) -> dict:
        return {"dir": self.phase_dir.path, "tag": self.tag, "formula": self.formula,
                "composition": self.composition, "elements": sorted(self.elements),
                "source": self.source, "mp_id": self.mp_id}


def sanitize_label(s: str) -> str:
    """Space-group symbols contain '/' (P4/mmm) - never use them raw as paths."""
    return re.sub(r"[^A-Za-z0-9_.\-+]", "-", s.replace("/", "-"))


def phase_from_dir(d: "str | PhaseDir", source: str, mp_id: str | None = None) -> Phase | None:
    pd = PhaseDir.of(d)
    if not os.path.exists(pd.poscar):
        return None
    try:
        a = read_structure(pd.poscar)
    except Exception as exc:  # noqa: BLE001
        log.warning("cannot read %s: %s", pd.poscar, exc)
        return None
    if mp_id is None:
        mp_id = (read_json(pd.join("metadata.json")) or {}).get("material_id")
    return Phase(pd, reduced_formula(a), composition(a), frozenset(composition(a)), source, mp_id)


def subsystems(elements) -> list[tuple[str, ...]]:
    els = sorted(elements)
    return list(chain.from_iterable(combinations(els, r) for r in range(1, len(els) + 1)))


# ---------------------------------------------------------------------------
# local
# ---------------------------------------------------------------------------

def scan_root(root: str, elements: frozenset, source: str) -> list[Phase]:
    out: list[Phase] = []
    if not os.path.isdir(root):
        return out
    for mat in sorted(os.listdir(root)):
        mdir = os.path.join(root, mat)
        if not os.path.isdir(mdir):
            continue
        for lab in sorted(os.listdir(mdir)):
            ph = phase_from_dir(os.path.join(mdir, lab), source)
            if ph and ph.elements <= elements:
                out.append(ph)
    return out


# ---------------------------------------------------------------------------
# Materials Project
# ---------------------------------------------------------------------------

def _mp_query(elements: frozenset) -> list:
    """Return MP summary docs of all chemical subsystems within the hull window."""
    key = cfg.phases.mp_api_key.strip()
    if not key:
        log.warning("[mp-api] no API key (cfg.phases.mp_api_key / MP_API_KEY) - skipping")
        return []
    try:
        from mp_api.client import MPRester
    except ImportError:
        log.warning("[mp-api] package 'mp-api' not installed - skipping")
        return []
    docs: list = []
    with MPRester(key) as mpr:
        for sys_ in subsystems(elements):
            chemsys = "-".join(sys_)
            kw: dict = {}
            if cfg.phases.mp_max_e_above_hull is not None:
                kw["energy_above_hull"] = (0.0, cfg.phases.mp_max_e_above_hull)
            res = mpr.materials.summary.search(
                chemsys=chemsys, fields=["material_id", "structure", "energy_above_hull", "formula_pretty",
                                         "symmetry"], **kw)
            res = sorted(res, key=lambda d: d.energy_above_hull)
            if cfg.phases.mp_max_per_chemsys is not None:
                res = res[: cfg.phases.mp_max_per_chemsys]
            log.info("[mp-api] %-12s %3d phase(s)", chemsys, len(res))
            docs += res
    return docs


def _mp_index_path(root: str, elements: frozenset) -> str:
    return os.path.join(root, "_mp_index", "-".join(sorted(elements)) + ".json")


def fetch_mp_phases(elements: frozenset, root: str, docs: list | None = None,
                    refresh: bool = False) -> list[Phase]:
    """Materials Project phases of all subsystems, written under *root* (``candidates_root``: structures only).

    The query is cached (``<root>/_mp_index/<chemsys>.json`` lists the directories written, with the
    query parameters), so repeated screening calls do not hit the network again.
    ``refresh=True`` (or a changed ``mp_max_*`` option) re-queries.
    """
    pc = cfg.phases
    params = {"window": pc.mp_max_e_above_hull, "per_chemsys": pc.mp_max_per_chemsys}
    idx_path = _mp_index_path(root, elements)
    if docs is None and not refresh:
        idx = read_json(idx_path)
        if idx and idx.get("params") == params:
            out = [p for p in (phase_from_dir(d, "mp-api") for d in idx["dirs"]) if p]
            if len(out) == len(idx["dirs"]):
                log.info("[mp-api] %s: %d phase(s) from the local index", "-".join(sorted(elements)), len(out))
                return out
    docs_given = docs is not None
    docs = _mp_query(elements) if docs is None else docs
    out: list[Phase] = []
    for doc in docs:
        s = doc.structure
        atoms_formula = s.composition.reduced_formula
        try:
            sg = str(doc.symmetry.symbol)
        except Exception:  # noqa: BLE001
            sg = "P1"
        mid = str(doc.material_id)
        d = os.path.join(root, atoms_formula, f"{sanitize_label(sg)}_{mid}")
        if not os.path.exists(os.path.join(d, "POSCAR")):
            from heatup.core.layout import from_pymatgen
            os.makedirs(d, exist_ok=True)
            write_poscar(os.path.join(d, "POSCAR"), from_pymatgen(s))
            write_json(os.path.join(d, "metadata.json"), {
                "material_id": mid, "formula": atoms_formula, "symmetry": sg,
                "e_above_hull_MP": float(doc.energy_above_hull), "source": "mp-api"})
        ph = phase_from_dir(d, "mp-api", mid)
        if ph:
            out.append(ph)
    if out and not docs_given:               # a query that returned nothing (no key, no network) is not cached
        write_json(idx_path, {"params": params, "dirs": [p.phase_dir.path for p in out]})
    return out


def promote(ph: Phase) -> Phase:
    """Copy a structure from ``candidates_root`` into ``database_root`` (same ``<formula>/<label>``): calculations only ever
    happen in the database.  An existing database entry wins; a different structure under the same name is reported."""
    src = ph.phase_dir
    dst = PhaseDir(os.path.join(cfg.phases.database_root, src.material, src.label))
    if os.path.exists(dst.poscar):
        if ledger.file_hash(src.poscar) != ledger.file_hash(dst.poscar):
            log.warning("%s: the database holds another structure under this name - the database one is used", dst.tag)
    else:
        os.makedirs(dst.path, exist_ok=True)
        shutil.copyfile(src.poscar, dst.poscar)
        write_json(dst.metadata_json, {**(read_json(src.metadata_json) or {}), "origin": src.path, "source": ph.source})
    return phase_from_dir(dst, ph.source, ph.mp_id) or ph


def to_database(phase_dir: "str | PhaseDir") -> PhaseDir:
    """*phase_dir* itself when it is in ``database_root``, else its copy there (see :func:`promote`)."""
    pd = PhaseDir.of(phase_dir)
    if os.path.abspath(pd.path).startswith(os.path.abspath(cfg.phases.database_root) + os.sep):
        return pd
    ph = phase_from_dir(pd, "candidates")
    if ph is None:
        raise FileNotFoundError(pd.poscar)
    return promote(ph).phase_dir


# ---------------------------------------------------------------------------
# de-duplication / assembly
# ---------------------------------------------------------------------------

def _matcher():
    from pymatgen.analysis.structure_matcher import StructureMatcher
    g = cfg.generation
    return StructureMatcher(ltol=g.ltol, stol=g.stol, angle_tol=g.angle_tol)


def same_structure(a: "str | PhaseDir", b: "str | PhaseDir", use_relaxed: bool = True) -> bool:
    """True when two phase directories hold the same crystal structure."""
    sm = _matcher()

    def load(p):
        p = PhaseDir.of(p)
        f = p.contcar if (use_relaxed and os.path.exists(p.contcar)) else p.poscar
        return to_pymatgen(read_structure(f))
    return bool(sm.fit(load(a), load(b)))


def deduplicate(phases: list[Phase], keep_first: bool = True) -> list[Phase]:
    """Drop structurally identical phases (same reduced formula + StructureMatcher)."""
    sm = _matcher()
    kept: list[tuple[Phase, object]] = []
    for ph in phases:
        s = to_pymatgen(read_structure(ph.phase_dir.poscar))
        if any(k.formula == ph.formula and sm.fit(ks, s) for k, ks in kept):
            continue
        kept.append((ph, s))
    return [k for k, _ in kept]


def find_reference_phases(elements, mp_docs: list | None = None, exclude=(), refresh_mp: bool = False,
                          local_only: bool = False) -> list[Phase]:
    """All known phases whose elements are a subset of *elements*, de-duplicated by source priority
    (database > candidates > Materials Project > generated) and, unless ``local_only``, all in ``database_root``
    (structures found elsewhere are copied there, see :func:`promote`).

    *exclude* = phase directories that must not appear (e.g. the targets themselves; their candidate twins too).
    ``local_only`` = read the database only (analysis stage: no network, no copying).
    """
    els = frozenset(elements)
    pc = cfg.phases
    skip = {PhaseDir.of(e).path for e in exclude}
    skip_names = {(PhaseDir.of(e).material, PhaseDir.of(e).label) for e in exclude}
    found = scan_root(pc.database_root, els, "database")
    if not local_only:
        found += scan_root(pc.candidates_root, els, "candidates")
        if "mp-api" in pc.sources:
            found += [p for p in fetch_mp_phases(els, pc.candidates_root, mp_docs, refresh_mp) if p.elements <= els]
        if "generate" in pc.sources:
            from heatup.prepare.generation import generate_for_subsystems
            found += generate_for_subsystems(els, pc.candidates_root)
    seen: set = set()
    unique = []
    for p in found:                                     # same <formula>/<label> in both roots: the database one
        key = (p.phase_dir.material, p.phase_dir.label)
        if p.phase_dir.path in skip or key in skip_names or key in seen:
            continue
        seen.add(key)
        unique.append(p)
    unique = deduplicate(unique)
    return unique if local_only else [p if p.source == "database" else promote(p) for p in unique]


def find_competing_phases(target_dir: "str | PhaseDir", mp_docs: list | None = None,
                          local_only: bool = False) -> list[Phase]:
    """All competitors of *target_dir* (target and its own structure excluded)."""
    target = PhaseDir.of(target_dir)
    t = phase_from_dir(target, "target")
    if t is None:
        raise FileNotFoundError(target.poscar)
    found = [p for p in find_reference_phases(t.elements, mp_docs, local_only=local_only) if p.phase_dir != target]
    # the target's own structure (e.g. its MP entry) is not a competitor
    tsm = _matcher()
    ts = to_pymatgen(read_structure(target.poscar))
    out = []
    for p in found:
        if p.formula == t.formula and tsm.fit(ts, to_pymatgen(read_structure(p.phase_dir.poscar))):
            log.info("excluding %s: same structure as target", p.tag)
            continue
        out.append(p)
    return out
