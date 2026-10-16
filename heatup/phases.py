"""heatup.phases
=================
Discovery of competing phases for the convex hull.

Sources (``cfg.phases.sources``, in priority order for de-duplication)

``database``    ``<database_root>/<formula>/<label>/POSCAR``  (your own calculations)
``candidates``  ``<candidates_root>/<formula>/<label>/POSCAR`` (read-only; mirrored into ``competing_root``)
``competing``   ``<competing_root>/...``: everything HeatUp fetched, generated or mirrored, with its results
``mp-api``      Materials Project: *every* chemical subsystem of the target
                (v0.2 queried ``elements=[...]`` which returns only phases that
                contain *all* of them, then filtered - so binaries/elements were
                missing and the hull was wrong).  Only phases within
                ``mp_max_e_above_hull`` of MP's own hull are downloaded, capped
                per subsystem; structures are cached under ``competing_root`` (query index included).
``generate``    :mod:`heatup.generation` (random/AI structures).

Fixes vs v0.2: space-group symbols such as ``P4/mmm`` were used as directory
names (creating nested directories) - now sanitised; PyXtal generation of all
230 groups for all sub-compositions was the default (hours, mostly junk) - it is
now opt-in and followed by de-duplication and a 0 K pre-screen; the target's own
structure is excluded from the competitors (it would trivially sit on the hull).
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from dataclasses import dataclass
from itertools import chain, combinations

from heatup import manifest
from heatup.config import cfg
from heatup.paths import PhaseDir, read_json, write_json
from heatup.structure import composition, read_structure, reduced_formula, to_pymatgen, write_poscar

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
            res = mpr.materials.summary.search(
                chemsys=chemsys, energy_above_hull=(0.0, cfg.phases.mp_max_e_above_hull),
                fields=["material_id", "structure", "energy_above_hull", "formula_pretty",
                        "symmetry"])
            res = sorted(res, key=lambda d: d.energy_above_hull)[: cfg.phases.mp_max_per_chemsys]
            log.info("[mp-api] %-12s %3d phase(s)", chemsys, len(res))
            docs += res
    return docs


def _mp_index_path(root: str, elements: frozenset) -> str:
    return os.path.join(root, "_mp_index", "-".join(sorted(elements)) + ".json")


def fetch_mp_phases(elements: frozenset, root: str, docs: list | None = None,
                    refresh: bool = False) -> list[Phase]:
    """Materials Project phases of all subsystems, written under *root* (``competing_root``).

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
            from heatup.structure import from_pymatgen
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


def mirror_candidate(ph: Phase, root: str) -> Phase:
    """Copy a user-supplied structure into the competing-phase tree (calculations never write into
    ``candidates_root``).  Existing results of the copy are kept; the POSCAR is refreshed if the source changed."""
    d = os.path.join(root, ph.formula, f"cand_{sanitize_label(ph.phase_dir.label)}")
    src, dst = ph.phase_dir.poscar, os.path.join(d, "POSCAR")
    if not os.path.exists(dst) or manifest.file_hash(src) != manifest.file_hash(dst):
        os.makedirs(d, exist_ok=True)
        shutil.copyfile(src, dst)
        write_json(os.path.join(d, "metadata.json"), {"formula": ph.formula, "source": "candidates",
                                                      "origin": ph.phase_dir.path})
    return phase_from_dir(d, "candidates") or ph


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


def find_reference_phases(elements, mp_docs: list | None = None, exclude=(),
                          refresh_mp: bool = False) -> list[Phase]:
    """All known phases whose elements are a subset of *elements* (de-duplicated, by source priority).

    Priority: ``database`` (your calculated phases) > ``candidates`` (mirrored) > ``competing``
    (what HeatUp fetched / generated earlier) > ``mp-api`` > ``generate``.  Everything HeatUp creates
    lives under ``phases.competing_root``; ``database_root`` and ``candidates_root`` are only read.
    *exclude* = phase directories that must not appear as references (e.g. the targets themselves).
    """
    els = frozenset(elements)
    pc = cfg.phases
    skip = {PhaseDir.of(e).path for e in exclude}
    found: list[Phase] = []
    if "database" in pc.sources:
        found += scan_root(pc.database_root, els, "database")
    if "candidates" in pc.sources:
        found += [mirror_candidate(p, pc.competing_root)
                  for p in scan_root(pc.candidates_root, els, "candidates") if p.phase_dir.path not in skip]
    if "competing" in pc.sources:
        found += scan_root(pc.competing_root, els, "competing")
    if "mp-api" in pc.sources:
        found += [p for p in fetch_mp_phases(els, pc.competing_root, mp_docs, refresh_mp) if p.elements <= els]
    if "generate" in pc.sources:
        from heatup.generation import generate_for_subsystems
        found += generate_for_subsystems(els, pc.competing_root)
    found = [p for p in found if p.phase_dir.path not in skip]
    return deduplicate(found)


def find_competing_phases(target_dir: "str | PhaseDir", mp_docs: list | None = None) -> list[Phase]:
    """All competitors of *target_dir* (target and its own structure excluded)."""
    target = PhaseDir.of(target_dir)
    t = phase_from_dir(target, "target")
    if t is None:
        raise FileNotFoundError(target.poscar)
    found = [p for p in find_reference_phases(t.elements, mp_docs) if p.phase_dir != target]
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
