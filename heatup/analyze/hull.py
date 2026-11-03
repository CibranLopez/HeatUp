"""heatup.analyze.hull
======================
Gate 3 - temperature-dependent convex hull, from cached results only.

Workflow
--------
1. competing phases = the phases of the database whose elements are a subset of the target's, with a converged
   relaxation (put there by ``prepare_project``, computed by ``run_plan``) **and calculated with the target's
   settings**: a hull compares free energies, so a phase whose plan has other relaxation/phonon options than the
   target's is refused with a warning, never silently mixed in (:func:`usable_competitors`);
2. optional **0 K pruning**: the 0 K hull of the relaxed enthalpies; every competitor more than the
   plan's ``prune_window_eV`` above it is left out (opt-in; default: no pruning).  :func:`prune_competitors` is the *same* function notebook 2 uses to
   decide which competitors get phonons, so the two stages always agree on the competitor set;
3. G(T) for every phase with the *same* method (:mod:`heatup.analyze.thermo`): the target is **not** treated with
   a different method than its competitors (v0.2 compared an anharmonic target with harmonic competitors, which
   makes the hull distance depend on the method offset, not on chemistry);
4. hull at every temperature; the target's *signed* distance to the hull of its competitors (negative = stable
   with margin) at the phase's operating temperature;
5. optional anharmonic overlay: dF_anh(T) = F_VDOS(T) - F_HA(T) of the target is added to its distance (an
   estimate, reported next to - never instead of - the consistent number).

Not included (documented limitations): gas-phase corrections for O2/N2/F2 references, spin/orbital contributions
beyond ``magnetic/moments.json``, finite-size / charged defects.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict

import numpy as np

from heatup.analyze.thermo import free_energy, reference_energy
from heatup.core import ledger
from heatup.core.config import STATUS_FAIL, STATUS_MISSING, STATUS_OK, STATUS_WARN, cfg
from heatup.core.layout import PhaseDir, write_json
from heatup.prepare.phases import Phase, find_competing_phases, phase_from_dir

log = logging.getLogger("heatup")


log = logging.getLogger("heatup")


# ---------------------------------------------------------------------------
# hull arithmetic (pure functions - unit tested analytically)
# ---------------------------------------------------------------------------

def _entry(comp: dict, e_per_atom: float, name: str):
    from pymatgen.analysis.phase_diagram import PDEntry
    from pymatgen.core import Composition
    return PDEntry(Composition(comp), float(e_per_atom), name=name)   # fractions sum to 1 -> E per atom


def has_element_references(elements, comps) -> list[str]:
    """Elements for which no pure-element competitor exists (hull would be undefined)."""
    pure = {next(iter(c)) for c in comps if len(c) == 1}
    return sorted(set(elements) - pure)


def hull_distance(target_comp: dict, target_G: float, competitors: dict) -> dict:
    """Signed distance (eV/atom) of the target above the hull of *competitors*.

    ``competitors``: ``{name: (composition_fractions, G_per_atom)}``.
    Negative => the target lies below the competitors' hull (stable by that margin).
    """
    from pymatgen.analysis.phase_diagram import PhaseDiagram
    missing = has_element_references(target_comp, [c for c, _ in competitors.values()])
    if missing:
        raise ValueError(f"no elemental reference for {missing}: add the pure elements to the competitors")
    entries = [_entry(c, g, n) for n, (c, g) in competitors.items()]
    pd = PhaseDiagram(entries)
    t = _entry(target_comp, target_G, "TARGET")
    decomp, e = pd.get_decomp_and_e_above_hull(t, allow_negative=True)
    return {"e_hull_eV": float(e),
            "decomposition": {k.name: float(v) for k, v in decomp.items()} if decomp else {},
            "stable_phases": sorted(x.name for x in pd.stable_entries)}


def hull_scan(target_comp: dict, target_G: np.ndarray, competitors: dict) -> dict:
    """:func:`hull_distance` at every temperature. ``competitors[name] = (comp, G(T) array)``."""
    n = len(target_G)
    e = np.zeros(n)
    dec: list[dict] = []
    stable: list[list[str]] = []
    for i in range(n):
        r = hull_distance(target_comp, float(target_G[i]),
                          {k: (c, float(g[i])) for k, (c, g) in competitors.items()})
        e[i] = r["e_hull_eV"]
        dec.append(r["decomposition"])
        stable.append(r["stable_phases"])
    return {"e_hull_eV": e, "decomposition": dec, "stable_phases": stable}


# ---------------------------------------------------------------------------
# 0 K pre-screen
# ---------------------------------------------------------------------------

def zero_k_screen(target: Phase, competitors: list[Phase]) -> dict:
    """0 K enthalpy hull of target + competitors -> distance of each competitor (>= 0)."""
    from pymatgen.analysis.phase_diagram import PhaseDiagram
    ent = []
    ref = {}
    for p in [target] + competitors:
        h0 = reference_energy(p.phase_dir)
        if h0 is None:
            continue
        ref[p.tag] = h0
        ent.append(_entry(p.composition, h0, p.tag))
    pd = PhaseDiagram(ent)
    dist = {e.name: float(pd.get_e_above_hull(e, allow_negative=True)) for e in ent}
    return {"e_above_hull_0K": dist, "H0": ref}


def THERMO_STEPS_OF(target_dir) -> tuple:
    """The calculations whose settings must agree between target and competitor for the hull's method."""
    return ledger.gate_steps("thermodynamic", ledger.phonon_method(target_dir), target_dir)


def usable_competitors(target_dir) -> tuple[list[Phase], dict[str, dict]]:
    """``(usable, refused)``: the database's competitors of *target_dir*, split by whether their calculation settings
    (plan options of the steps the hull uses) equal the target's.  ``refused = {tag: {option: (theirs, target's)}}``.
    Phases without a plan (results from elsewhere) are judged by what exists."""
    target_dir = PhaseDir.of(target_dir)
    tplan = ledger.read_plan(target_dir)
    comps = find_competing_phases(target_dir, local_only=True)
    if tplan is None:
        return comps, {}
    hull_steps = ["relax", *THERMO_STEPS_OF(target_dir)]
    ok, refused = [], {}
    for p in comps:
        cplan = ledger.read_plan(p.phase_dir)
        conflicts = ledger.plan_conflicts(cplan, hull_steps, tplan.options) if cplan else {}
        if conflicts:
            refused[p.tag] = conflicts
        else:
            ok.append(p)
    return ok, refused


def refusal_note(tag: str, conflicts: dict) -> str:
    what = ", ".join(f"{k}: {a} vs {b}" for k, (a, b) in list(conflicts.items())[:4])
    return (f"[reuse refused] {tag} was prepared/calculated with other settings ({what}) - not used as a competing phase; "
            "delete its folder to recompute it with the current settings")


def prune_competitors(target: Phase, competitors: list[Phase], window: float | None):
    """``(keep, dropped, zero_K)``: competitors within *window* eV/atom of the 0 K hull (None = keep all)."""
    z = zero_k_screen(target, competitors)
    keep, dropped = [], []
    for p in competitors:
        d = z["e_above_hull_0K"].get(p.tag)
        (keep if d is not None and (window is None or d <= window) else dropped).append(p)
    return keep, dropped, z


def pruned_competitors(target_dirs=None) -> dict[str, str]:
    """``{competitor tag: reason}`` for the competitors of the plan (``role == "competitor"``) that the 0 K screen
    drops for *every* target they compete with.  A competitor whose relaxation (or whose target's) is missing is
    undecided and therefore not pruned.  Used by notebook 2 (skip their phonons/MD) and the status table."""
    from heatup.core.layout import iter_phase_dirs
    targets = [PhaseDir.of(t) for t in (target_dirs if target_dirs is not None else iter_phase_dirs(cfg.phases.database_root))]
    seen: dict[str, int] = defaultdict(int)
    dropped_for: dict[str, list[str]] = defaultdict(list)
    for t in targets:
        if ledger.prune_window(t) is None:               # pruning is opt-in: nothing is ever pruned without a window
            continue
        tph = phase_from_dir(t, "target")
        if tph is None or not t.has_relaxation():
            continue
        comps = usable_competitors(t)[0]
        relaxed = [p for p in comps if p.phase_dir.has_relaxation()]
        try:
            _, dropped, z = prune_competitors(tph, relaxed, ledger.prune_window(t))
        except Exception as exc:  # noqa: BLE001
            log.debug("0 K screen for %s failed: %s", t.tag, exc)
            continue
        for p in comps:
            seen[p.tag] += 1
        for p in dropped:
            d = z["e_above_hull_0K"].get(p.tag)
            dropped_for[p.tag].append("not on the 0 K hull" if d is None else f"{1e3 * d:.0f} meV/atom above the 0 K hull of {t.tag}")
    out = {}
    for tag, why in dropped_for.items():
        pl = ledger.read_plan(os.path.join(cfg.phases.database_root, *tag.split("/")))
        if len(why) == seen[tag] and pl is not None and pl.role == "competitor":
            out[tag] = f"pruned: {why[0]}"
    return out


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------

def assess_thermodynamic_stability(target_dir, operating_T: float | None = None) -> dict:
    """Gate 3 for one target.  ``operating_T=None`` -> the phase's own value from ``plan.json``."""
    target_dir = PhaseDir.of(target_dir)
    hc = cfg.hull
    T = np.asarray(hc.temperatures, float)
    op = ledger.operating_T(target_dir) if operating_T is None else float(operating_T)
    if not (T.min() <= op <= T.max()):
        return {"available": False, "status": STATUS_MISSING,
                "message": f"operating T = {op:g} K outside the hull grid [{T.min():g}, {T.max():g}] K"}
    target = phase_from_dir(target_dir, "target")
    if target is None or not target_dir.has_relaxation():
        return {"available": False, "status": STATUS_MISSING,
                "message": "target has no converged relaxation (run step 'relax' in notebook 2)."}

    stab = target_dir.stability_dir
    os.makedirs(stab, exist_ok=True)
    flags: list[str] = []
    method = ledger.phonon_method(target_dir)

    # 1. competitors on disk
    usable, refused = usable_competitors(target_dir)
    flags += [refusal_note(tag, c) for tag, c in refused.items()]
    competitors = [p for p in usable if p.phase_dir.has_relaxation()]
    log.info("%d relaxed competing phase(s) found", len(competitors))

    # 2. 0 K pruning (same rule and window as the compute stage)
    window = ledger.prune_window(target_dir)
    try:
        keep, dropped, z = prune_competitors(target, competitors, window)
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "status": STATUS_MISSING, "flags": flags,
                "message": f"0 K hull could not be built: {exc}" + (f" ({len(refused)} competitor(s) refused: other settings)" if refused else "")}
    log.info("0 K pruning (%s): %d kept, %d dropped",
             "off" if window is None else f"> {1e3 * window:.0f} meV", len(keep), len(dropped))
    write_json(os.path.join(stab, "zero_K_hull.json"), {
        "prune_window_eV": window, "e_above_hull_0K": z["e_above_hull_0K"],
        "kept": [p.tag for p in keep], "dropped": [p.tag for p in dropped]})

    # 3. G(T) for everyone, same method
    gt = free_energy(target_dir, T, method)
    if not gt.get("available"):
        return {"available": False, "status": STATUS_MISSING,
                "message": f"target free energy unavailable: {gt.get('message')}"}
    flags += [f"target: {f}" for f in gt["flags"]]
    comps: dict[str, tuple[dict, np.ndarray]] = {}
    labels: dict[str, str] = {}
    for p in keep:
        g = free_energy(p.phase_dir, T, method)
        if not g.get("available"):
            flags.append(f"{p.tag}: {g.get('message')} - left out of the hull (computed in notebook 2?)")
            continue
        flags += [f"{p.tag}: {f}" for f in g["flags"]]
        comps[p.tag] = (p.composition, g["G_eV_per_atom"])
        labels[p.tag] = (p.phase_dir.energy() or {}).get("calculator", "?")
    tlabel = (target_dir.energy() or {}).get("calculator", "?")
    mism = sorted({v for v in labels.values() if v != tlabel})
    if mism:
        flags.append(f"energy-scale mismatch: target {tlabel} vs competitors {mism} "
                     "- hull distances are not comparable across calculators")

    # 4. hull
    try:
        scan = hull_scan(target.composition, gt["G_eV_per_atom"], comps)
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "status": STATUS_MISSING,
                "message": f"hull could not be built: {exc}", "flags": flags}
    e_T = float(np.interp(op, T, scan["e_hull_eV"]))
    i_op = int(np.argmin(np.abs(T - op)))
    decomp = scan["decomposition"][i_op]

    # 5. anharmonic overlay for the target
    overlay = None
    if hc.target_overlay.upper() == "VDOS" and method != "VDOS":
        try:
            gv = free_energy(target_dir, T, "VDOS")
            if gv.get("available"):
                # dF_anh (HA subtracted *with the same ZPE/T grid*)
                dF = gv["G_eV_per_atom"] - gt["G_eV_per_atom"]
                overlay = {"dF_anh_eV": dF, "e_hull_eV": scan["e_hull_eV"] + dF,
                           "e_hull_at_T_eV": float(np.interp(op, T, scan["e_hull_eV"] + dF)),
                           "md_temperatures": ledger.md_temperatures(target_dir)}
        except Exception as exc:  # noqa: BLE001
            flags.append(f"VDOS overlay failed: {exc}")

    if len(comps) < len(target.elements):
        flags.append("few competing phases - hull may be incomplete")
    status = (STATUS_OK if e_T <= cfg.gates.hull_ok_eV else
              STATUS_WARN if e_T <= cfg.gates.hull_fail_eV else STATUS_FAIL)
    if status == STATUS_OK and mism:
        status = STATUS_WARN
    if e_T <= cfg.gates.hull_ok_eV:
        msg = (f"On the hull at {op:g} K ({1e3 * e_T:+.1f} meV/atom vs competitors)"
               if e_T <= 1e-4 else f"{1e3 * e_T:.1f} meV/atom above hull at {op:g} K (within tolerance)")
    else:
        dec = ", ".join(f"{v:.2f} {k}" for k, v in sorted(decomp.items(), key=lambda kv: -kv[1]))
        msg = f"{1e3 * e_T:.1f} meV/atom above the hull at {op:g} K; decomposes to {dec}"
    if mism:
        msg += " [calculator mismatch - see flags]"

    res = {"available": True, "status": status, "message": msg, "method": method,
           "operating_T_K": op, "e_hull_at_T_eV": e_T,
           "decomposition": decomp, "n_competing": len(comps),
           "n_pruned_0K": len(dropped), "temperatures": T, "e_hull_eV": scan["e_hull_eV"],
           "stable_phases": scan["stable_phases"][i_op],
           "overlay": overlay, "flags": flags,
           "G_target_eV": gt["G_eV_per_atom"], "contributions": gt["contributions"],
           "competitors": {k: {"formula": next(p.formula for p in keep if p.tag == k)} for k in comps}}
    write_json(os.path.join(stab, "hull_vs_T.json"), res)
    write_json(os.path.join(stab, "secondary_phases.json"),
               [p.to_dict() for p in keep] + [dict(p.to_dict(), pruned=True) for p in dropped])
    return res


# ---------------------------------------------------------------------------
# 0 K hull for the pre-MD stage (no phonons, no MD)
# ---------------------------------------------------------------------------

def zero_k_hull(target_dir) -> dict:
    """Signed distance of the target's relaxed enthalpy to the 0 K hull of its competitors.

    Needs only relaxations.  ``{"available": False, "message": ...}`` when the hull cannot
    be built (no competitors / missing elemental references).
    """
    target_dir = PhaseDir.of(target_dir)
    target = phase_from_dir(target_dir, "target")
    if target is None or not target_dir.has_relaxation():
        return {"available": False, "message": "target has no converged relaxation"}
    usable, refused = usable_competitors(target_dir)
    comps = [p for p in usable if p.phase_dir.has_relaxation()]
    ref = {}
    labels = set()
    for p in comps:
        h = reference_energy(p.phase_dir)
        if h is not None:
            ref[p.tag] = (p.composition, h)
            labels.add((p.phase_dir.energy() or {}).get("calculator", "?"))
    flags = [refusal_note(tag, c) for tag, c in refused.items()]
    tl = (target_dir.energy() or {}).get("calculator", "?")
    if labels - {tl}:
        flags.append(f"energy-scale mismatch: target {tl} vs competitors {sorted(labels - {tl})}")
    try:
        r = hull_distance(target.composition, reference_energy(target_dir), ref)
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "message": f"0 K hull could not be built: {exc}", "flags": flags,
                "n_competing": len(ref)}
    return {"available": True, "e_hull_0K_eV": r["e_hull_eV"], "decomposition": r["decomposition"],
            "stable_phases": r["stable_phases"], "n_competing": len(ref), "flags": flags}
