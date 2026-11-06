"""heatup.core.ledger
=====================
The bookkeeping of the whole package, in one place:

1. **STEPS** - the registry of every computed step: where it writes, which files prove it finished, which
   input files and which configuration keys its result depends on, which step it needs, whether it needs a
   calculator.  Nothing else in the package lists step names, directories or fingerprint keys.
2. **the step record** - ONE file per step folder, ``step_<step>.json``, with four sections that never repeat each other
   or the plan: ``task`` (what the step needs, reads and writes - for people and codes working in that folder),
   ``run`` (running / done / failed + error, written by :func:`tracking`; a ``running`` record whose process is gone means
   ``crashed``), ``result`` (provenance of the finished result: the options *actually used* + input hashes - the thing
   a changed plan is compared with), ``export`` (cells written for outside codes).  Planned options live only in
   ``plan.json``.  (v0.3 shared one ``manifest.json`` between ``md`` and the analysis steps; the analysis overwrote it and
   the next ``run_md`` discarded the finished trajectory.  Older per-step ``manifest_/status_/task_/export_`` files are read
   and folded into the record on the next write.)
3. **plan.json** - what *should* be computed for a phase, with the options to use (written by
   ``heatup.prepare``, executed by ``heatup.compute``).
4. **step_state / phase_rows** - the single classifier (done / stale / pending / unfinished / failed / ...)
   used by notebook 2 (what to run) and notebook 3 (what to warn about).

States
------
``done``       outputs exist, the record's ``result`` matches the plan options and the input hashes
``stale``      outputs exist but options or inputs changed
``pending``    nothing there yet
``running``    a live process holds the run marker
``crashed``    run marker says running but the process is gone (an MD run resumes from its trajectory)
``unfinished`` partial output (MD trajectory without ``md.json``)
``failed``     the step raised (the record's ``run`` has the error) or its result is unusable (relaxation not converged)
``pruned``     not needed: pruned by the 0 K screen (competitors) or the screening funnel (targets)
``external``   outputs without a HeatUp ``result`` (e.g. computed with DFT) - accepted as they are
"""

from __future__ import annotations

import contextlib
import dataclasses as dc
import hashlib
import logging
import os
import socket
import time
from datetime import datetime, timezone
from typing import Callable, Iterator

from heatup import __version__
from heatup.core.config import cfg, override
from heatup.core.layout import PhaseDir, read_json, trajectory_frames, write_json

log = logging.getLogger("heatup")

DONE, STALE, PENDING, RUNNING, CRASHED = "done", "stale", "pending", "running", "crashed"
UNFINISHED, FAILED, PRUNED, EXTERNAL = "unfinished", "failed", "pruned", "external"
GOOD_STATES = (DONE, EXTERNAL)                         # evidence that can be used
RUNNABLE_STATES = (PENDING, STALE, UNFINISHED, CRASHED)  # (+ FAILED when retrying)


# ---------------------------------------------------------------------------
# 1. the step registry
# ---------------------------------------------------------------------------

@dc.dataclass(frozen=True)
class Step:
    name: str
    func: str                                           # "module:function"
    where: Callable[[PhaseDir, "float | None"], str]    # directory holding outputs, manifest and run marker
    outputs: tuple                                      # files (relative to ``where``) that exist when finished
    inputs: Callable[[PhaseDir, "float | None"], list]  # files whose hash is recorded
    needs: tuple = ()                                   # steps that must be usable first
    tier: int = 1                                       # cost class: 1 cheap | 2 expensive (prunable competitors) | 3 analysis (no calculator)
    sections: tuple = ()                                # config sections (whole) that fingerprint the step ...
    keys: tuple = ()                                    # ... plus individual dotted keys
    calc: bool = True                                   # needs an interatomic-potential calculator
    per_T: bool = False                                 # one run per MD temperature
    group: str | None = None                            # ``cfg.run.force`` group name
    partial: tuple = ()                                 # files that show a started-but-unfinished run


CALC_KEYS = ("calc.backend", "calc.dtype", "calc.dispersion")        # + calc.model (per material)
_MD_SOURCE = lambda p, T: [os.path.join(p.md_run(T), "output.traj")]
_CONTCAR = lambda p, T: [p.contcar]

STEPS: dict[str, Step] = {s.name: s for s in [
    Step("relax", "heatup.compute.static:run_relaxation", lambda p, T: p.relax_dir, ("energy.json", "CONTCAR"),
         lambda p, T: [p.poscar], tier=1, sections=("relax",)),
    Step("elastic", "heatup.compute.static:run_elastic", lambda p, T: p.elastic_dir, ("elastic.json",),
         _CONTCAR, needs=("relax",), tier=1, sections=("elastic",)),
    Step("harmonic", "heatup.compute.harmonic:run_harmonic", lambda p, T: p.harmonic_dir, ("harmonic.json",),
         _CONTCAR, needs=("relax",), tier=2, sections=("harmonic",), keys=("relax.pressure_GPa",)),
    Step("quasiharmonic", "heatup.compute.quasiharmonic:run_quasiharmonic", lambda p, T: p.quasiharmonic_dir,
         ("quasiharmonic.json",), _CONTCAR, needs=("relax",), tier=2, sections=("harmonic", "quasiharmonic"),
         keys=("relax.fmax", "relax.max_steps", "relax.pressure_GPa")),
    Step("perturbative", "heatup.compute.perturbative:run_perturbative", lambda p, T: p.perturbative_dir,
         ("perturbative.json",), _CONTCAR, needs=("relax",), tier=2, sections=("harmonic", "perturbative")),
    Step("md", "heatup.compute.md:run_md", lambda p, T: p.md_run(T), ("output.traj", "md.json"),
         _CONTCAR, needs=("relax",), tier=2, sections=("md",), per_T=True, partial=("output.traj",)),
    # analysis of a finished MD run (calculator-free; run by notebook 3, one manifest each)
    Step("mdstats", "heatup.analyze.trajectory:run_mdstats", lambda p, T: p.md_run(T), ("mdstats.json",),
         _MD_SOURCE, needs=("md",), tier=3, calc=False, per_T=True, group="analysis",
         keys=("analysis.equil_ps", "analysis.stride")),
    Step("vdos", "heatup.analyze.trajectory:run_vdos", lambda p, T: p.anharmonic_run(T), ("vdos.json",),
         _MD_SOURCE, needs=("md",), tier=3, calc=False, per_T=True, group="analysis",
         keys=("analysis.equil_ps", "analysis.stride", "analysis.vdos_max_lag_ps")),
    Step("transition", "heatup.analyze.trajectory:run_transition", lambda p, T: p.md_run(T), ("transition.json",),
         _MD_SOURCE, needs=("md",), tier=3, calc=False, per_T=True, group="analysis",
         keys=("analysis.equil_ps", "analysis.stride", "analysis.block_ps", "analysis.mobile_species")),
]}
COMPUTE_STEPS = tuple(n for n, s in STEPS.items() if s.tier < 3)
ANALYSIS_STEPS = tuple(n for n, s in STEPS.items() if s.tier == 3)

#: step that produces each level of vibrational evidence (``cfg.vibrational.sources`` names the levels)
LEVEL_STEPS = {"harmonic": "harmonic", "quasiharmonic": "quasiharmonic", "anharmonic": "md"}
#: steps whose output the gates read (``vibrational`` follows ``cfg.vibrational.sources``; ``thermodynamic`` the selected terms)
GATE_STEPS = {"mechanical": ("elastic",), "structural_transition": ("md",), "thermodynamic": ()}
#: the hull's evidence per vibrational model (``cfg.terms.vibrational``), for the target AND every competing phase
THERMO_STEPS = {"off": (), "zpe": ("harmonic",), "harmonic": ("harmonic",), "quasiharmonic": ("quasiharmonic",),
                "anharmonic": ("md",)}


def vibrational_steps(sources=None) -> tuple:
    """Compute steps behind the vibrational evidence *sources* (default ``cfg.vibrational.sources``)."""
    sources = cfg.vibrational.sources if sources is None else sources
    bad = [x for x in sources if x not in LEVEL_STEPS]
    if bad:
        raise ValueError(f"cfg.vibrational.sources: unknown {bad}; choose from {list(LEVEL_STEPS)}")
    return tuple(dict.fromkeys(LEVEL_STEPS[x] for x in sources))


def gate_steps(gate: str, model: str = "harmonic", phase=None) -> tuple:
    """Steps whose output *gate* reads.  The hull's are those of the selected terms (``heatup.core.contributions``):
    the vibrational *model* plus whatever the other selected terms need (resolved for *phase* as prepared)."""
    if gate == "vibrational":
        return vibrational_steps()
    base = tuple(GATE_STEPS.get(gate, ()))
    if gate != "thermodynamic":
        return base
    from heatup.core import contributions
    extra = contributions.required_steps(phase=phase, exclude=("vibrational",))
    return base + tuple(dict.fromkeys(THERMO_STEPS[model.lower()] + extra))


def steps_for(gates, model: str = "harmonic", role: str = "target") -> list[str]:
    """Compute steps needed to evaluate *gates* with vibrational *model* (registry order).  A competing phase only
    serves the hull, so it needs just the hull's evidence (``THERMO_STEPS``).  Used by notebook 1 and by the one-call
    facade ``heatup.screen``; ``cfg.plan.target_steps`` / ``competitor_steps`` override it."""
    need = {"relax"}                                      # "relax" is no gate: it is always planned (see relax.optimize)
    if role == "target":
        for g in gates:
            need.update(gate_steps(g, model))
    elif "thermodynamic" in gates:
        need.update(gate_steps("thermodynamic", model))
    return [n for n in COMPUTE_STEPS if n in need]


# ---------------------------------------------------------------------------
# 2. the step record (one file per step folder)
# ---------------------------------------------------------------------------

def fingerprint(step: str, material: str | None = None, calc: bool = True) -> dict:
    """The part of the configuration that can change *step*'s result.  ``calc=False`` leaves the calculator out
    (used when judging results that may come from another code, and always for analysis steps)."""
    spec = STEPS[step]
    snap = cfg.snapshot()
    fp: dict = {}
    for sec in spec.sections:
        for k, v in snap[sec].items():
            fp[f"{sec}.{k}"] = v
    keys = list(spec.keys) + (list(CALC_KEYS) if spec.calc and calc else [])
    for dotted in keys:
        sec, key = dotted.split(".", 1)
        fp[dotted] = snap[sec][key]
    if spec.calc and calc:
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


_LEGACY_FILES = ("manifest_{step}.json", "status_{step}.json", "task_{step}.json", "export_{step}.json")


def _record_file(step_dir: str, step: str) -> str:
    return os.path.join(step_dir, f"step_{step}.json")


def _read_record_at(step_dir: str, step: str) -> dict:
    rec = read_json(_record_file(step_dir, step))
    if rec is not None:
        return rec
    rec = {}                                           # no record yet: fold in files of earlier versions, if any
    m = read_json(os.path.join(step_dir, f"manifest_{step}.json"))
    if m is None:
        shared = read_json(os.path.join(step_dir, "manifest.json"))          # v0.3: one shared file, honoured only for its step
        m = shared if shared and shared.get("step") == step else None
    if m:
        rec["result"] = {k: m[k] for k in ("heatup_version", "timestamp_utc", "fingerprint", "inputs", "extra") if k in m}
    s = read_json(os.path.join(step_dir, f"status_{step}.json"))
    if s:
        rec["run"] = {k: v for k, v in s.items() if k not in ("step", "temperature_K")}
    return rec


def read_record(phase, step: str, T: float | None = None) -> dict:
    """The step record (``{}`` if none): sections ``task``, ``run``, ``result``, ``export``."""
    return _read_record_at(STEPS[step].where(PhaseDir.of(phase), T), step)


def _update(phase, step: str, T: float | None, create: bool = True, **sections) -> None:
    """Replace the given sections of the step record (``None`` removes one); the file is written atomically."""
    phase = PhaseDir.of(phase)
    d = STEPS[step].where(phase, T)
    rec = _read_record_at(d, step)
    rec.update(step=step, temperature_K=T)
    for name, value in sections.items():
        if value is None:
            rec.pop(name, None)
        else:
            rec[name] = value
    path = _record_file(d, step)
    if not (rec.keys() & {"task", "run", "result", "export"}):
        if os.path.exists(path):
            os.remove(path)
    elif create or os.path.exists(path):
        write_json(path, rec)
    for pat in _LEGACY_FILES:                          # folded into the record
        old = os.path.join(d, pat.format(step=step))
        if os.path.exists(old):
            os.remove(old)


def read_manifest(phase, step: str, T: float | None = None) -> dict | None:
    """The ``result`` section: options actually used + input hashes of the finished step (None: nothing recorded)."""
    return read_record(phase, step, T).get("result")


def record(phase, step: str, T: float | None = None, extra: dict | None = None) -> None:
    """Record a finished step: the fingerprint under the *current* configuration + input hashes."""
    phase = PhaseDir.of(phase)
    spec = STEPS[step]
    m = {"heatup_version": __version__, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
         "fingerprint": fingerprint(step, phase.material),
         "inputs": {os.path.basename(p): file_hash(p) for p in spec.inputs(phase, T)}}
    if extra:
        m["extra"] = extra
    _update(phase, step, T, result=m)


def set_task(phase, step: str, T: float | None, task: dict) -> None:
    _update(phase, step, T, task=task)


def clear_task(phase, step: str, T: float | None) -> None:
    """Drop the ``task`` section of a step that is no longer planned (the record goes if nothing else is in it)."""
    if read_record(phase, step, T).get("task") is not None:
        _update(phase, step, T, create=False, task=None)


def get_export(phase, step: str, T: float | None = None) -> dict | None:
    return read_record(phase, step, T).get("export")


def set_export(phase, step: str, T: float | None, export: dict) -> None:
    _update(phase, step, T, export=export)


def diff(phase, step: str, T: float | None = None, calc: bool = True) -> dict:
    """``{key: (saved, current)}`` for everything that differs ({} = fresh)."""
    phase = PhaseDir.of(phase)
    spec = STEPS[step]
    saved = read_manifest(phase, step, T)
    if saved is None:
        return {"_manifest": (None, "missing")}
    diffs: dict = {}
    sf = saved.get("fingerprint", {})
    for k, v in fingerprint(step, phase.material, calc).items():
        if sf.get(k) != v:
            diffs[k] = (sf.get(k), v)
    for p in spec.inputs(phase, T):
        name = os.path.basename(p)
        if saved.get("inputs", {}).get(name) != file_hash(p):
            diffs[f"input:{name}"] = (saved.get("inputs", {}).get(name), file_hash(p))
    return diffs


def is_fresh(phase, step: str, T: float | None = None, calc: bool = True) -> bool:
    return not diff(phase, step, T, calc)


def manifest_summary(phase, step: str, T: float | None = None) -> str:
    m = read_manifest(phase, step, T)
    if not m:
        return f"[no manifest for {PhaseDir.of(phase).tag}:{step}]"
    lines = [f"step={step}  heatup={m['heatup_version']}  at {m['timestamp_utc']}"]
    lines += [f"  {k} = {v}" for k, v in sorted(m["fingerprint"].items())]
    lines += [f"  input {k}: {v}" for k, v in m.get("inputs", {}).items()]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. run markers (the ``run`` section of the step record)
# ---------------------------------------------------------------------------

def mark(phase, step: str, T: float | None, state: str, **extra) -> None:
    _update(phase, step, T, run={"state": state, **extra})


def read_mark(phase, step: str, T: float | None = None) -> dict | None:
    return read_record(phase, step, T).get("run")


@contextlib.contextmanager
def tracking(phase, step: str, T: float | None = None) -> Iterator[None]:
    """Write the run marker around a step: running -> done | failed(error)."""
    phase = PhaseDir.of(phase)
    t0 = time.time()
    mark(phase, step, T, "running", pid=os.getpid(), host=socket.gethostname(),
         started_utc=datetime.now(timezone.utc).isoformat())
    try:
        yield
    except BaseException as exc:  # noqa: BLE001  (KeyboardInterrupt too: an interrupted step is a failed one)
        mark(phase, step, T, "failed", error=f"{type(exc).__name__}: {exc}"[-600:],
             finished_utc=datetime.now(timezone.utc).isoformat(), wall_s=time.time() - t0)
        raise
    else:
        mark(phase, step, T, "done", finished_utc=datetime.now(timezone.utc).isoformat(), wall_s=time.time() - t0)


def _pid_alive(m: dict) -> bool | None:
    """True/False for a process on this host, None when the marker comes from another machine."""
    if m.get("host") != socket.gethostname():
        return None
    try:
        os.kill(int(m["pid"]), 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OSError, KeyError, ValueError):
        return True
    return True


# ---------------------------------------------------------------------------
# 4. plan.json
# ---------------------------------------------------------------------------

PLAN_SCHEMA = 1


@dc.dataclass
class Plan:
    """What to compute for one phase (``<phase>/plan.json``)."""
    role: str = "target"                                 # "target" | "competitor"
    steps: list = dc.field(default_factory=list)         # names of compute steps, in execution order
    gates: list = dc.field(default_factory=list)         # gates this phase was prepared for (what notebook 3 reports)
    operating_T_K: float | None = None                   # temperature at which the thermodynamic check judges this phase
    md_temperatures_K: list = dc.field(default_factory=list)
    options: dict = dc.field(default_factory=dict)       # dotted config values of the planned steps (no calculator)
    source: str = ""                                     # database | candidates | mp-api | generate | ...
    pruned: dict | None = None                           # {"reason": ...}: dropped by the screening funnel
    schema: int = PLAN_SCHEMA
    created_utc: str = ""

    def tasks(self) -> list[tuple[str, "float | None"]]:
        out: list = []
        for s in self.steps:
            if STEPS[s].per_T:
                out += [(s, float(T)) for T in self.md_temperatures_K]
            else:
                out.append((s, None))
        return out

    def to_dict(self) -> dict:
        return dc.asdict(self)


def plan_options(steps) -> dict:
    """Dotted values of the configuration that *steps* depend on (calculator excluded: that is notebook 2's)."""
    snap = cfg.snapshot()
    out: dict = {}
    for s in steps:
        spec = STEPS[s]
        for sec in spec.sections:
            for k, v in snap[sec].items():
                out[f"{sec}.{k}"] = v
        for dotted in spec.keys:
            sec, key = dotted.split(".", 1)
            out[dotted] = snap[sec][key]
    return out


def plan_conflicts(existing: Plan, steps, options: dict) -> dict:
    """``{key: (existing, wanted)}``: options of *steps* that *existing* also plans, with a different value.
    A phase whose results were produced with other settings must not be mixed into a hull (nor silently recomputed)."""
    common = [s for s in steps if s in existing.steps]
    out = {k: (existing.options[k], options[k]) for k in sorted(plan_options(common))
           if k in existing.options and k in options and existing.options[k] != options[k]}
    out.update({k: (existing.options[k], options[k]) for k in sorted(options)            # the SAME terms for every phase
                if k.startswith("terms.") and k in existing.options and existing.options[k] != options[k]})
    return out


def merge_plans(existing: Plan, wanted: Plan) -> Plan:
    """*existing* plus the steps (and their options) that only *wanted* has.  The role, operating temperature and
    source of *existing* stay (a target that also serves as a competitor remains a target)."""
    out = dc.replace(existing)
    out.steps = [s for s in COMPUTE_STEPS if s in existing.steps or s in wanted.steps]
    out.options = {**wanted.options, **existing.options}
    out.gates = sorted(set(existing.gates) | set(wanted.gates))
    out.md_temperatures_K = sorted(set(existing.md_temperatures_K) | set(wanted.md_temperatures_K))
    return out


def read_plan(phase) -> Plan | None:
    d = read_json(PhaseDir.of(phase).plan_json)
    if not d:
        return None
    old = [s for s in d.get("steps", []) if s not in STEPS]
    if old:
        raise RuntimeError(f"{PhaseDir.of(phase).tag}: plan.json lists steps of an earlier version {old}.  Run "
                           "heatup.core.migrate.migrate_database() once (renames the folders and records; nothing is recomputed).")
    known = {f.name for f in dc.fields(Plan)}
    return Plan(**{k: v for k, v in d.items() if k in known})


def write_plan(phase, plan: Plan) -> str:
    plan.created_utc = plan.created_utc or datetime.now(timezone.utc).isoformat()
    return write_json(PhaseDir.of(phase).plan_json, plan.to_dict())


@contextlib.contextmanager
def applied(plan: Plan | None) -> Iterator[None]:
    """Run with the options stored in *plan* (so manifests fingerprint exactly what the plan says)."""
    if plan is None or not plan.options:
        yield
    else:
        with override(**plan.options):
            yield


def resolve_gates(phase, explicit: list | None = None) -> list[str]:
    """Gates to report: *explicit* > ``cfg.gates.enabled`` > the phase's plan > all."""
    from heatup.core.config import GATES
    if explicit is not None:
        return list(explicit)
    if cfg.gates.enabled is not None:
        return list(cfg.gates.enabled)
    plan = read_plan(phase)
    return [g for g in plan.gates if g in GATES] if plan and plan.gates else list(GATES)


def vibrational_model(phase, explicit: str | None = None) -> str:
    """Vibrational model of the hull: *explicit* > ``cfg.terms.vibrational`` > what the phase was prepared with > the
    default of the term (``heatup.core.contributions.resolve``).  One of ``off zpe harmonic quasiharmonic anharmonic``."""
    from heatup.core import contributions
    return contributions.resolve(phase, vibrational=explicit)["vibrational"].lower()


phonon_method = vibrational_model                      # name of earlier versions (kept for callers not updated yet)


def operating_T(phase, default: float | None = None) -> float:
    """The phase's own operating temperature (plan) > *default* > ``cfg.hull.operating_T``."""
    plan = read_plan(phase)
    if plan and plan.operating_T_K is not None:
        return float(plan.operating_T_K)
    return float(cfg.hull.operating_T if default is None else default)


def prune_window(phase) -> float | None:
    """0 K pruning window (eV/atom) fixed when the project was prepared (``plan.prune_window_eV``)."""
    plan = read_plan(phase)
    if plan and "plan.prune_window_eV" in plan.options:
        return plan.options["plan.prune_window_eV"]
    return cfg.plan.prune_window_eV


def md_temperatures(phase) -> list[int]:
    """The MD runs evidence is taken from: finished runs on disk, restricted to the planned temperatures when
    the plan lists any (a stray older run never silently enters the structural-transition and vibrational checks)."""
    phase = PhaseDir.of(phase)
    found = phase.md_temperatures()
    plan = read_plan(phase)
    if plan and plan.md_temperatures_K:
        planned = {int(round(t)) for t in plan.md_temperatures_K}
        return [t for t in found if t in planned]
    return found


# ---------------------------------------------------------------------------
# 5. state classification
# ---------------------------------------------------------------------------

def _relax_problem(phase: PhaseDir, T) -> str | None:
    e = read_json(phase.energy_json) or {}
    if e.get("converged", False) and os.path.exists(phase.contcar):
        return None
    return f"relaxation not converged (max |F| = {e.get('fmax_final_eV_A', float('nan')):.3f} eV/A)"


def _md_progress(phase: PhaseDir, T) -> str:
    n = trajectory_frames(os.path.join(phase.md_run(T), "output.traj"))
    return f"{n} of {cfg.md.n_steps // cfg.md.nblock + 1} frames"


def step_state(phase, step: str, T: float | None = None, calc: bool = False) -> tuple[str, str]:
    """``(state, detail)`` of one step of one phase.  ``calc=True`` also compares the calculator (notebook 2);
    ``calc=False`` judges the results whatever produced them (notebook 3)."""
    phase = PhaseDir.of(phase)
    spec = STEPS[step]
    d = spec.where(phase, T)
    m = read_mark(phase, step, T)
    if m and m.get("state") == "running":
        alive = _pid_alive(m)
        extra = f" ({_md_progress(phase, T)})" if step == "md" else ""
        if alive is False:
            return CRASHED, f"process {m.get('pid')} is gone{extra}"
        return RUNNING, f"pid {m.get('pid')}@{m.get('host')}{extra}" if alive is None else f"pid {m.get('pid')}{extra}"
    if not all(os.path.exists(os.path.join(d, o)) for o in spec.outputs):
        if m and m.get("state") == "failed":
            return FAILED, m.get("error", "")
        if any(os.path.exists(os.path.join(d, f)) for f in spec.partial):
            return UNFINISHED, _md_progress(phase, T) if step == "md" else "partial output"
        return PENDING, ""
    if step == "relax" and _relax_problem(phase, T):
        return FAILED, _relax_problem(phase, T) or ""
    if read_manifest(phase, step, T) is None:
        return EXTERNAL, "outputs without a HeatUp record"
    df = diff(phase, step, T, calc)
    if df:
        return STALE, ", ".join(sorted(df))[:200]
    return DONE, ""


def calculator_of(phase) -> str:
    return (read_json(PhaseDir.of(phase).energy_json) or {}).get("calculator", "")


def phase_rows(phase, calc: bool = False, pruned: dict | None = None) -> list[dict]:
    """One row per planned compute step (per MD temperature) plus one per analysis step of every MD run found.

    ``pruned``: ``{phase.tag: reason}`` of competitors that the 0 K screen drops - their expensive steps are
    reported ``pruned`` instead of ``pending``.
    """
    phase = PhaseDir.of(phase)
    plan = read_plan(phase)
    calc_label = calculator_of(phase)
    rows: list[dict] = []

    def add(step, T, state, detail=""):
        rows.append({"phase": phase.tag, "role": plan.role if plan else "unplanned", "step": step,
                     "T_K": T, "state": state, "detail": detail, "calculator": calc_label})

    if plan is not None and plan.pruned:
        add("(all)", None, PRUNED, plan.pruned.get("reason", ""))
        return rows
    tasks = plan.tasks() if plan else [(s, None) for s in COMPUTE_STEPS if not STEPS[s].per_T] + \
        [("md", float(T)) for T in phase.md_temperatures(complete=False)]
    with applied(plan):
        for step, T in tasks:
            state, detail = step_state(phase, step, T, calc)
            if plan is None and state == PENDING:
                continue                                         # unplanned: only report what exists
            if state == PENDING and pruned and phase.tag in pruned and STEPS[step].tier >= 2:
                state, detail = PRUNED, pruned[phase.tag]
            add(step, T, state, detail)
        for T in md_temperatures(phase):
            for step in ANALYSIS_STEPS:
                state, detail = step_state(phase, step, float(T), calc=False)
                add(step, float(T), state, detail or ("analysis: run_analysis() in notebook 03" if state == PENDING else ""))
    return rows
