"""heatup.compute.execute
=========================
Notebook 2: execute the plans.

:func:`run_plan` runs **everything the plans list** - for every phase with a ``plan.json`` (targets and competitors,
``database_root``; the role in the plan tells targets from competitors) - in dependency order::

    relax -> elastic -> harmonic -> quasiharmonic -> perturbative -> md    (each for all phases before the next step)

Nothing about the calculations is configured here, only the calculator (``cfg.calc``) and the device: the options
of each task are the ones stored in its plan (``ledger.applied``).  A task is run when its state
(:func:`heatup.core.ledger.step_state`, calculator included) is pending, stale, unfinished or crashed - and failed,
unless ``retry_failed=False`` - or when ``cfg.run.force`` lists it.  Finished tasks are skipped without starting a
process, a task whose prerequisite is not usable is reported ``blocked``, a failure never stops the loop, each task
runs in its own process (GPU memory is released), and an interrupted MD run resumes.  Re-running is always safe.

Any protocol is just a sequence of calls - nothing here is mandatory::

    run_plan(steps=["relax", "elastic"])        # a cheap first stage ...
    status()                                    # ... look at it, drop what you do not like ...
    run_plan()                                  # ... then everything else
    run_plan(targets=["database/BaTiO3/P4mm"])  # only some targets (and their competitors)

The only optional intelligence is the 0 K pruning (``cfg.plan.prune_window_eV``, off by default): if you set a
window, competitors far above the 0 K hull get no phonons/MD.

Returns one row per considered task: ``{phase, step, T_K, action, detail}`` with ``action`` in
``ran / failed / done / blocked / pruned / running / would run`` (``dry_run=True`` only reports).
"""

from __future__ import annotations

import logging
from collections import Counter

from heatup.analyze.hull import pruned_competitors, usable_competitors
from heatup.compute.runtime import run_step
from heatup.core import ledger
from heatup.core.config import cfg
from heatup.core.layout import PhaseDir, iter_phase_dirs

log = logging.getLogger("heatup")


def _phases(targets, include_competing: bool) -> tuple[list[PhaseDir], list[PhaseDir]]:
    """``(targets, competitors)`` to work on.  Whole project: every phase of the database, split by the plan's role.
    Given targets: those, plus their competitors that the settings check lets through (``usable_competitors``)."""
    if targets is None:
        allp = list(iter_phase_dirs(cfg.phases.database_root))
        comp = [p for p in allp if (ledger.read_plan(p) or ledger.Plan()).role == "competitor"]
        return [p for p in allp if p not in comp], (comp if include_competing else [])
    t = [PhaseDir.of(x) for x in targets]
    c: dict[str, PhaseDir] = {}
    if include_competing:
        for pd in t:
            for p in usable_competitors(pd)[0]:
                pl = ledger.read_plan(p.phase_dir)
                if pl is not None and pl.role == "competitor":
                    c[p.phase_dir.path] = p.phase_dir
    return t, list(c.values())


def run_plan(targets=None, device: str | None = None, steps: list[str] | None = None, dry_run: bool = False,
             retry_failed: bool = True, include_competing: bool = True) -> list[dict]:
    """Run every unfinished planned step.  ``targets``: phase directories (default: the whole project);
    ``steps``: only these step names (default: all that are planned)."""
    tdirs, cdirs = _phases(targets, include_competing)
    phases = tdirs + cdirs
    if not any(ledger.read_plan(p) for p in phases):
        log.warning("no plan.json found under %s - run prepare_project first (notebook 1)", cfg.phases.database_root)
    rows: list[dict] = []
    scheduled: set = set()                    # dry run: prerequisites that this pass would produce
    pruned: dict | None = None                # only evaluated if a prune window is set (after the relaxations ran)

    def add(ph, step, T, action, detail=""):
        rows.append({"phase": ph.tag, "step": step, "T_K": T, "action": action, "detail": detail})

    for step in (s for s in ledger.COMPUTE_STEPS if steps is None or s in steps):
        spec = ledger.STEPS[step]
        for ph in phases:
            plan = ledger.read_plan(ph)
            if plan is None or step not in plan.steps:
                continue
            if plan.pruned:
                add(ph, step, None, "pruned", plan.pruned.get("reason", ""))
                continue
            if plan.role == "competitor" and spec.tier == 2:
                pruned = pruned_competitors(tdirs) if pruned is None else pruned
                if ph.tag in pruned:
                    add(ph, step, None, "pruned", pruned[ph.tag])
                    continue
            for T in (plan.md_temperatures_K if spec.per_T else [None]):
                with ledger.applied(plan):
                    state, detail = ledger.step_state(ph, step, T, calc=True)
                    blocked = [n for n in spec.needs
                               if ledger.step_state(ph, n, None, calc=True)[0] not in ledger.GOOD_STATES
                               and not (dry_run and (ph.path, n) in scheduled)]
                    forced = cfg.run.is_forced(step)
                    if state == ledger.RUNNING:
                        add(ph, step, T, "running", detail)
                        continue
                    if state in ledger.GOOD_STATES and not forced:
                        add(ph, step, T, "done")
                        continue
                    if state == ledger.FAILED and not retry_failed and not forced:
                        add(ph, step, T, "failed", detail)
                        continue
                    if blocked:
                        add(ph, step, T, "blocked", "needs " + ", ".join(blocked))
                        continue
                    if dry_run:
                        add(ph, step, T, "would run", f"{state}" + (f": {detail}" if detail else ""))
                        scheduled.add((ph.path, step))
                        continue
                    log.info("--- %s %s%s (%s)", ph.tag, step, "" if T is None else f" {T:g} K", state)
                    ok = run_step(step, ph, device=device, **({} if T is None else {"temperature": T}))
                    add(ph, step, T, "ran" if ok else "failed",
                        "" if ok else (ledger.read_mark(ph, step, T) or {}).get("error", ""))
    n = Counter(r["action"] for r in rows)
    log.info("run_plan: %s", ", ".join(f"{k} {v}" for k, v in sorted(n.items())) or "nothing to do")
    return rows
