#!/usr/bin/env python3
"""Rewrite ``heatup`` imports of a v0.3 module (e.g. ``fc3.py``) for the v0.4 layout.

    python tools/migrate_imports.py path/to/fc3.py [more files ...]      # edits in place, keeps <file>.bak

* ``from heatup.<old> import a, b``  ->  one import per new module (names that moved are routed individually)
* ``from heatup import manifest``    ->  ``from heatup.core import ledger``
* ``manifest.is_fresh(<dir>, "<step>", phase.material, inputs)`` / ``manifest.write_manifest(...)``
                                     ->  ``ledger.is_fresh(phase, "<step>")`` / ``ledger.record(phase, "<step>")``
  (the ledger derives directory, inputs and fingerprint from ``ledger.STEPS``: check that the ``fc3`` entry lists the
  input files your step really depends on)
* prints a WARNING for everything it cannot rewrite safely (removed config options, ``manifest`` calls with other
  arguments, ...).
"""
import ast
import re
import shutil
import sys

MODULE_MAP = {
    "config": "heatup.core.config", "constants": "heatup.core.config",
    "paths": "heatup.core.layout", "structure": "heatup.core.layout",
    "builders": "heatup.prepare.phases", "phases": "heatup.prepare.phases",
    "generation": "heatup.prepare.generation",
    "calculator": "heatup.compute.calculator", "runtime": "heatup.compute.runtime",
    "relax": "heatup.compute.static", "elastic": "heatup.compute.static", "fc3": "heatup.compute.fc3",
    "phonons": "heatup.compute.phonons", "md": "heatup.compute.md",
    "vibthermo": "heatup.analyze.thermo", "free_energy": "heatup.analyze.thermo",
    "md_analysis": "heatup.analyze.trajectory", "vdos": "heatup.analyze.trajectory",
    "structural_transition": "heatup.analyze.trajectory",
    "mechanical": "heatup.analyze.gates", "vibrational": "heatup.analyze.gates",
    "thermodynamic": "heatup.analyze.hull",
    "pipeline": "heatup.analyze.evaluate",
    "report": "heatup.analyze.report", "plotting": "heatup.analyze.report",
    "workflow": "heatup.screen", "api": "heatup.screen",
}
NAME_MAP = {  # (old module, name) -> (new module, new name)
    ("phonons", "load_ha"): ("heatup.analyze.thermo", None), ("phonons", "load_qha"): ("heatup.analyze.thermo", None),
    ("structural_transition", "assess_structural_transition"): ("heatup.analyze.gates", None),
    ("structural_transition", "verdict"): ("heatup.analyze.gates", "transition_verdict"),
    ("structural_transition", "assess_sim"): ("heatup.analyze.trajectory", "run_transition"),
    ("md_analysis", "analyse_md"): ("heatup.analyze.trajectory", "run_mdstats"),
    ("pipeline", "run_pipeline"): ("heatup.analyze.evaluate", "evaluate"),
    ("pipeline", "run_batch"): ("heatup.analyze.evaluate", "evaluate_all"),
    ("pipeline", "database_status"): ("heatup.analyze.evaluate", "status"),
    ("workflow", "prescreen"): ("heatup.screen", None),
    ("fc3", "load_fc3"): ("heatup.analyze.thermo", None), ("fc3", "anharmonic_summary"): ("heatup.analyze.thermo", None),
}
REMOVED_OPTIONS = ["cfg.md.equil_ps", "cfg.md.stride", "cfg.gates.block_ps", "cfg.gates.mobile_species",
                   "cfg.hull.prune_window_eV"]
HINT = {"cfg.md.equil_ps": "cfg.analysis.equil_ps", "cfg.md.stride": "cfg.analysis.stride",
        "cfg.gates.block_ps": "cfg.analysis.block_ps", "cfg.gates.mobile_species": "cfg.analysis.mobile_species",
        "cfg.hull.prune_window_eV": "cfg.plan.prune_window_eV"}


def migrate(path: str) -> None:
    src = open(path).read()
    lines = src.splitlines()
    tree = ast.parse(src)
    edits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("heatup"):
            parts = node.module.split(".")
            if node.module == "heatup":
                if any(a.name == "manifest" for a in node.names):
                    rest = [a for a in node.names if a.name != "manifest"]
                    new = ["from heatup.core import ledger"]
                    if rest:
                        new.append("from heatup import " + ", ".join(a.name + (f" as {a.asname}" if a.asname else "") for a in rest))
                    edits.append((node.lineno, node.end_lineno, new))
                continue
            old = parts[1] if len(parts) == 2 else None
            if old == "manifest":
                edits.append((node.lineno, node.end_lineno, ["from heatup.core import ledger"]))
                continue
            if old not in MODULE_MAP:
                continue
            groups: dict[str, list[str]] = {}
            for a in node.names:
                mod, new_name = NAME_MAP.get((old, a.name), (MODULE_MAP[old], None))
                n = new_name or a.name
                groups.setdefault(mod, []).append(n + (f" as {a.asname}" if a.asname else (f" as {a.name}" if new_name else "")))
            edits.append((node.lineno, node.end_lineno, [f"from {m} import {', '.join(sorted(set(n)))}" for m, n in sorted(groups.items())]))
    for start, end, new in sorted(edits, reverse=True):
        lines[start - 1:end] = new
    text = "\n".join(lines) + "\n"
    text = re.sub(r'manifest\.is_fresh\([^,]+,\s*"(\w+)",\s*[^,]+,\s*[^)]*\)', r'ledger.is_fresh(phase, "\1")', text)
    text = re.sub(r'manifest\.write_manifest\([^,]+,\s*"(\w+)",\s*[^,]+,\s*[^)]*\)', r'ledger.record(phase, "\1")', text)
    for opt in REMOVED_OPTIONS:
        if opt in text:
            print(f"WARNING {path}: {opt} was moved - use {HINT[opt]}")
    for m in re.finditer(r"manifest\.\w+", text):
        print(f"WARNING {path}: '{m.group(0)}' not rewritten - use heatup.core.ledger (record / is_fresh / diff / read_manifest)")
    shutil.copy(path, path + ".bak")
    open(path, "w").write(text)
    print(f"{path}: {len(edits)} import(s) rewritten (backup: {path}.bak)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for p in sys.argv[1:]:
        migrate(p)
