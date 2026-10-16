"""heatup.manifest
===================
Cache validity for every computed step.

Each step directory gets a ``manifest.json`` with (i) a *fingerprint* - the
subset of the configuration that can change that step's result (``STEP_KEYS``)
- and (ii) short hashes of the input files it consumed.  A step is "fresh"
only when both match, so changing ``cfg.relax.fmax`` or re-relaxing a
structure automatically invalidates phonons, elastic constants and MD, which
v0.2 never did (a stale ``dos.json`` was happily reused for a different
structure).
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from typing import Iterable

from heatup import __version__
from heatup.config import cfg
from heatup.paths import read_json, write_json

# step -> config sections (whole) and individual dotted keys that affect it
STEP_KEYS: dict[str, dict[str, list[str]]] = {
    "relax":   {"sections": ["relax"],            "keys": ["calc.backend", "calc.dtype", "calc.dispersion"]},
    "phonons": {"sections": ["phonon"],           "keys": ["calc.backend", "calc.dtype", "calc.dispersion", "relax.pressure_GPa"]},
    "qha":     {"sections": ["phonon", "qha"],    "keys": ["calc.backend", "calc.dtype", "calc.dispersion", "relax.fmax"]},
    "fc3":     {"sections": ["phonon", "fc3"],    "keys": ["calc.backend", "calc.dtype", "calc.dispersion"]},
    "elastic": {"sections": ["elastic"],          "keys": ["calc.backend", "calc.dtype", "calc.dispersion"]},
    "md":      {"sections": ["md"],               "keys": ["calc.backend", "calc.dtype", "calc.dispersion"]},
    "analysis": {"sections": [],                  "keys": ["md.equil_ps", "md.stride", "gates.block_ps",
                                                           "gates.symmetry_break_A", "gates.mobile_species"]},
}


def fingerprint(step: str, material: str | None = None) -> dict:
    spec = STEP_KEYS[step]
    fp: dict = {}
    snap = cfg.snapshot()
    for sec in spec["sections"]:
        for k, v in snap[sec].items():
            fp[f"{sec}.{k}"] = v
    for dotted in spec["keys"]:
        sec, key = dotted.split(".", 1)
        fp[dotted] = snap[sec][key]
    fp["calc.model"] = cfg.calc.model_for(material)
    return fp


def file_hash(path: str, nbytes: int = 1 << 24) -> str | None:
    if not os.path.exists(path):
        return None
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        h.update(fh.read(nbytes))
    h.update(str(os.path.getsize(path)).encode())
    return h.hexdigest()[:16]


#: steps that share a directory with ``phonons`` keep their own manifest, otherwise each would invalidate the others
_OWN_MANIFEST = {"qha": "manifest_qha.json", "fc3": "manifest_fc3.json"}


def _manifest_path(step_dir: str, step: str | None = None) -> str:
    return os.path.join(step_dir, _OWN_MANIFEST.get(step or "", "manifest.json"))


def write_manifest(step_dir: str, step: str, material: str | None = None,
                   inputs: Iterable[str] = (), extra: dict | None = None) -> str:
    m = {
        "heatup_version": __version__,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "step": step,
        "fingerprint": fingerprint(step, material),
        "inputs": {os.path.basename(p): file_hash(p) for p in inputs},
    }
    if extra:
        m["extra"] = extra
    return write_json(_manifest_path(step_dir, step), m)


def diff_manifest(step_dir: str, step: str, material: str | None = None,
                  inputs: Iterable[str] = ()) -> dict:
    """Return ``{key: (saved, current)}`` for everything that differs ({} = fresh)."""
    saved = read_json(_manifest_path(step_dir, step))
    if saved is None:
        return {"_manifest": (None, "missing")}
    diffs: dict = {}
    cur_fp = fingerprint(step, material)
    for k, v in cur_fp.items():
        sv = saved.get("fingerprint", {}).get(k)
        if isinstance(v, tuple):
            v = list(v)
        if sv != v:
            diffs[k] = (sv, v)
    for p in inputs:
        name = os.path.basename(p)
        if saved.get("inputs", {}).get(name) != file_hash(p):
            diffs[f"input:{name}"] = (saved.get("inputs", {}).get(name), file_hash(p))
    return diffs


def is_fresh(step_dir: str, step: str, material: str | None = None,
             inputs: Iterable[str] = ()) -> bool:
    return not diff_manifest(step_dir, step, material, inputs)


def manifest_summary(step_dir: str) -> str:
    m = read_json(_manifest_path(step_dir))
    if not m:
        return f"[no manifest in {step_dir}]"
    lines = [f"step={m['step']}  heatup={m['heatup_version']}  at {m['timestamp_utc']}"]
    lines += [f"  {k} = {v}" for k, v in sorted(m["fingerprint"].items())]
    lines += [f"  input {k}: {v}" for k, v in m.get("inputs", {}).items()]
    return "\n".join(lines)
