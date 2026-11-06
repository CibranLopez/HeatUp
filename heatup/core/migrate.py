"""heatup.core.migrate  --  bring a database written by an earlier version to the current layout and names
==========================================================================================================
The vibrational levels were renamed (``heatup.core.provenance``) and every level now keeps ALL its files in its own folder
under ``phonons/``::

    phonons/ha.json, ha/, POSCAR        ->  phonons/harmonic/        (step ``phonons`` became ``harmonic``)
    phonons/qha.json, qha/              ->  phonons/quasiharmonic/   (step ``qha`` became ``quasiharmonic``)
    phonons/fc3.json, fc3/              ->  phonons/perturbative/    (step ``fc3`` became ``perturbative``)
    md/<T>K/vdos.json, vdos.png         ->  phonons/anharmonic/<T>K/ (+ source.json: no calculation, taken from md/<T>K)
    phonons/anha_md.json                ->  phonons/anharmonic/anharmonic.json
    harmonic/ quasiharmonic/ perturbative/ at the top of a phase (the 0.5.0 draft)  ->  phonons/<level>/

The step ``anharmonic`` (extra NVT runs on top of QHA) was retired.  :func:`migrate_database` moves the files and rewrites the
records so that **nothing is recomputed**: results, manifests (fingerprints keep their meaning under the new setting names) and
``plan.json`` follow.  It never deletes a result: what has no place any more (``anharmonic.json`` of the retired step) stays in
``phonons/``.  Safe to run twice; ``dry_run=True`` only reports.

    from heatup.core.migrate import migrate_database
    migrate_database("database", dry_run=True)    # look
    migrate_database("database")                  # do it

Command line:  ``python -m heatup.core.migrate [database_root] [--dry-run]``
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys

from heatup.core import provenance
from heatup.core.layout import iter_phase_dirs, read_json, write_json

log = logging.getLogger("heatup")

STEP_NAMES = {"phonons": "harmonic", "qha": "quasiharmonic", "fc3": "perturbative"}
SECTIONS = {"phonon": "harmonic", "qha": "quasiharmonic", "fc3": "perturbative"}
VIB_MODELS = {"HA": "harmonic", "QHA": "quasiharmonic", "VDOS": "anharmonic"}
RETIRED_OPTION_PREFIXES = ("anharmonic.",)
RETIRED_OPTIONS = ("phonon.fc_order", "hull.phonon_method", "terms.anharmonic")


def _rename_key(key: str) -> str | None:
    """New dotted option / fingerprint key, or None if the setting no longer exists."""
    if key in RETIRED_OPTIONS or key.startswith(RETIRED_OPTION_PREFIXES):
        return None
    sec, _, rest = key.partition(".")
    return f"{SECTIONS.get(sec, sec)}.{rest}"


def _migrate_fingerprint(fp: dict, step: str) -> dict:
    out = {}
    for k, v in fp.items():
        nk = _rename_key(k)
        if nk:
            out[nk] = v
    if step == "perturbative":
        out["perturbative.order"] = 3
    return out


def _migrate_plan(d: dict) -> dict:
    d = dict(d)
    d["steps"] = [STEP_NAMES.get(s, s) for s in d.get("steps", []) if s != "anharmonic"]
    opts = {}
    for k, v in d.get("options", {}).items():
        nk = _rename_key(k)
        if nk:
            opts[nk] = VIB_MODELS.get(v, v) if nk == "terms.vibrational" and isinstance(v, str) else v
    if "perturbative" in d["steps"]:
        opts.setdefault("perturbative.order", 3)
    d["options"] = opts
    return d


class _Mover:
    def __init__(self, dry: bool):
        self.dry, self.actions = dry, []

    def move(self, src: str, dst: str) -> bool:
        if not os.path.exists(src) or os.path.exists(dst):
            return False
        self.actions.append(f"{os.path.relpath(src)} -> {os.path.relpath(dst)}")
        if not self.dry:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
        return True

    def move_tree(self, src: str, dst: str) -> None:
        if os.path.isdir(src):
            for name in sorted(os.listdir(src)):
                self.move(os.path.join(src, name), os.path.join(dst, name))
            if not self.dry:
                try:
                    os.rmdir(src)
                except OSError:
                    pass


def _rewrite_record(path: str, new_step: str, dry: bool) -> None:
    rec = read_json(path)
    if not rec or dry:
        return
    rec["step"] = new_step
    rec.pop("task", None)            # re-created by prepare_project / make_task_folders
    rec.pop("export", None)          # re-written by export_inputs (same files, new folders)
    if "result" in rec and "fingerprint" in rec["result"]:
        rec["result"]["fingerprint"] = _migrate_fingerprint(rec["result"]["fingerprint"], new_step)
    write_json(path, rec)


def _restamp(path: str, kind: str, dry: bool, **extra) -> None:
    d = read_json(path)
    if d is None or dry:
        return
    d["meta"] = provenance.stamp(kind, **extra)
    write_json(path, d, indent=None)


def migrate_phase(phase, dry_run: bool = False) -> list[str]:
    mv = _Mover(dry_run)
    j = os.path.join
    ph = phase.phonon_dir
    H, Q, P, A = phase.harmonic_dir, phase.quasiharmonic_dir, phase.perturbative_dir, phase.anharmonic_dir

    # -- layout of versions before 0.5: everything in phonons/ -------------------------------------------------------
    if os.path.isdir(ph):
        mv.move(j(ph, "ha.json"), j(H, "harmonic.json"))
        mv.move(j(ph, "phonon.png"), j(H, "phonon.png"))
        mv.move(j(ph, "step_phonons.json"), j(H, "step_harmonic.json"))
        mv.move(j(ph, "POSCAR"), j(H, "POSCAR"))
        mv.move_tree(j(ph, "ha"), H)
        mv.move(j(ph, "qha.json"), j(Q, "quasiharmonic.json"))
        mv.move(j(ph, "step_qha.json"), j(Q, "step_quasiharmonic.json"))
        mv.move_tree(j(ph, "qha"), Q)
        mv.move(j(ph, "fc3.json"), j(P, "perturbative.json"))
        mv.move(j(ph, "step_fc3.json"), j(P, "step_perturbative.json"))
        mv.move_tree(j(ph, "fc3"), P)
        mv.move(j(ph, "anha_md.json"), j(A, "anharmonic.json"))
    # -- the 0.5.0 draft: one folder per level at the top of the phase ---------------------------------------------------
    for name, dst in (("harmonic", H), ("quasiharmonic", Q), ("perturbative", P)):
        top = phase.join(name)
        if os.path.isdir(top) and not os.path.exists(dst):
            mv.move(top, dst)
        else:
            mv.move_tree(top, dst)
    mv.move(j(phase.md_dir, "anharmonic.json"), j(A, "anharmonic.json"))
    # -- the VDOS of every MD run moves next to the other phonons ----------------------------------------------------
    moved_vdos = False
    if os.path.isdir(phase.md_dir):
        for name in sorted(os.listdir(phase.md_dir)):
            if name.endswith("K") and name[:-1].isdigit():
                for f in ("vdos.json", "vdos.png", "step_vdos.json"):
                    moved_vdos |= mv.move(j(phase.md_dir, name, f), j(A, name, f))
    if not dry_run and (moved_vdos or os.path.exists(j(A, "anharmonic.json"))):
        from heatup.analyze.anharmonic import write_source_marker
        temps = [int(n[:-1]) for n in os.listdir(A) if n.endswith("K") and n[:-1].isdigit()]
        write_source_marker(phase, temps)
        mv.actions.append("phonons/anharmonic/source.json written")
    if not dry_run and os.path.isdir(ph):
        try:
            os.rmdir(ph)
        except OSError:
            left = sorted(n for n in os.listdir(ph) if n not in ("harmonic", "quasiharmonic", "perturbative", "anharmonic"))
            if left:
                log.info("%s: left in phonons/ (retired, not deleted): %s", phase.tag, left)

    # -- records, stamps, plan ----------------------------------------------------------------------------------------
    if not dry_run:
        for path, kind, extra in ((j(H, "harmonic.json"), "harmonic", {}), (j(Q, "quasiharmonic.json"), "quasiharmonic", {}),
                                  (j(P, "perturbative.json"), "perturbative", {"order": 3}),
                                  (j(A, "anharmonic.json"), "anharmonic", {})):
            if os.path.exists(path):
                _restamp(path, kind, dry_run, **extra)
        for d, step in ((H, "harmonic"), (Q, "quasiharmonic"), (P, "perturbative")):
            rp = os.path.join(d, f"step_{step}.json")
            rec = read_json(rp) if os.path.exists(rp) else None
            if rec and (rec.get("step") != step or "task" in rec or "export" in rec
                        or any(k.split(".")[0] in SECTIONS for k in rec.get("result", {}).get("fingerprint", {}))):
                _rewrite_record(rp, step, dry_run)
                mv.actions.append(f"{os.path.relpath(rp)}: record rewritten")
    pj = phase.plan_json
    plan = read_json(pj)
    if plan and (any(s in STEP_NAMES or s == "anharmonic" for s in plan.get("steps", []))
                 or any(_rename_key(k) != k for k in plan.get("options", {}))
                 or any(v in VIB_MODELS for k, v in plan.get("options", {}).items() if k == "terms.vibrational")):
        mv.actions.append("plan.json: step and option names")
        if not dry_run:
            write_json(pj, _migrate_plan(plan))
    return mv.actions


def migrate_database(root: str | None = None, dry_run: bool = False) -> dict[str, list[str]]:
    """Migrate every phase under *root* (default ``cfg.phases.database_root``); ``{phase tag: actions}`` of the phases that changed."""
    if root is None:
        from heatup.core.config import cfg
        root = cfg.phases.database_root
    out = {}
    for phase in iter_phase_dirs(root):
        acts = migrate_phase(phase, dry_run)
        if acts:
            out[phase.tag] = acts
    verb = "would change" if dry_run else "changed"
    log.info("migrate: %s %d phase(s)", verb, len(out))
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    res = migrate_database(args[0] if args else None, dry_run="--dry-run" in sys.argv)
    print(json.dumps(res, indent=1))
