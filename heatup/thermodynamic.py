"""heatup.thermodynamic
========================
Gate 3 - temperature-dependent convex hull.

Workflow
--------
1. discover competing phases (:mod:`heatup.phases`);
2. relax all of them (0 K, same calculator);
3. **0 K pruning**: build the 0 K hull from the relaxed enthalpies and drop every
   competitor more than ``hull.prune_window_eV`` above it - no phonon calculation
   is spent on phases that cannot reach the hull (the dominant cost in v0.2);
4. harmonic (or QHA) phonons for the survivors;
5. G(T) for every phase with the *same* method (:mod:`heatup.free_energy`):
   the target is **not** treated with a different method than its competitors
   (v0.2 compared an anharmonic target with harmonic competitors, which makes
   the hull distance depend on the method offset, not on chemistry);
6. hull at every temperature; the target's *signed* distance to the hull of its
   competitors (negative = stable with margin) at ``hull.operating_T``;
7. optional anharmonic overlay: dF_anh(T) = F_VDOS(T) - F_HA(T) of the target is
   added to its distance (an estimate, reported next to - never instead of - the
   consistent number).

Not included (documented limitations): gas-phase corrections for O2/N2/F2
references, spin/orbital contributions beyond ``magnetic/moments.json``, and
finite-size / charged defects.  See ``docs/AUDIT.md``.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from heatup.config import (STATUS_FAIL, STATUS_MISSING, STATUS_OK, STATUS_WARN, cfg, forced)
from heatup.free_energy import free_energy, reference_energy
from heatup.paths import PhaseDir, write_json
from heatup.phases import Phase, find_competing_phases, phase_from_dir

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


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------

def _prepare(phases: list[Phase], step: str, device: str | None, force: bool | None = None) -> list[Phase]:
    from heatup.runtime import run_step
    ok = []
    for p in phases:
        if step == "relax" and p.phase_dir.has_relaxation() and not forced("relax", force):
            ok.append(p)
            continue
        if run_step(step, p.phase_dir, device=device, force=force):
            ok.append(p)
        else:
            log.warning("%s: %s failed - phase dropped from the hull", p.tag, step)
    return ok


def assess_thermodynamic_stability(target_dir, operating_T: float | None = None,
                                   device: str | None = None, prepare: bool = True,
                                   force: bool | None = None, mp_docs: list | None = None) -> dict:
    target_dir = PhaseDir.of(target_dir)
    hc = cfg.hull
    T = np.asarray(hc.temperatures, float)
    op = float(hc.operating_T if operating_T is None else operating_T)
    if not (T.min() <= op <= T.max()):
        return {"available": False, "status": STATUS_MISSING,
                "message": f"operating T = {op:g} K outside the hull grid [{T.min():g}, {T.max():g}] K"}
    target = phase_from_dir(target_dir, "target")
    if target is None or not target_dir.has_relaxation():
        return {"available": False, "status": STATUS_MISSING,
                "message": "target has no converged relaxation (run step 'relax')."}

    stab = target_dir.stability_dir
    os.makedirs(stab, exist_ok=True)
    flags: list[str] = []
    method = hc.phonon_method.upper()

    # 1-2. discover + relax
    competitors = find_competing_phases(target_dir, mp_docs)
    log.info("%d competing phase(s) found", len(competitors))
    if prepare:
        competitors = _prepare(competitors, "relax", device, force)
    else:
        competitors = [p for p in competitors if p.phase_dir.has_relaxation()]

    # 3. 0 K pruning
    try:
        z = zero_k_screen(target, competitors)
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "status": STATUS_MISSING,
                "message": f"0 K hull could not be built: {exc}"}
    keep, dropped = [], []
    for p in competitors:
        d = z["e_above_hull_0K"].get(p.tag)
        ok = d is not None and (hc.prune_window_eV is None or d <= hc.prune_window_eV)
        (keep if ok else dropped).append(p)
    log.info("0 K pruning (%s): %d kept, %d dropped",
             "off" if hc.prune_window_eV is None else f"> {1e3 * hc.prune_window_eV:.0f} meV", len(keep), len(dropped))
    write_json(os.path.join(stab, "zero_K_hull.json"), {
        "prune_window_eV": hc.prune_window_eV, "e_above_hull_0K": z["e_above_hull_0K"],
        "kept": [p.tag for p in keep], "dropped": [p.tag for p in dropped]})

    # 4. phonons
    if prepare:
        if method in ("HA", "QHA"):
            keep = _prepare(keep, "phonons", device, force)
            if method == "QHA":
                keep = _prepare(keep, "qha", device, force)
            tgt_ok = _prepare([target], "phonons", device, force) and (
                method != "QHA" or _prepare([target], "qha", device, force))
            if not tgt_ok:
                return {"available": False, "status": STATUS_MISSING,
                        "message": "target phonons failed."}

    # 5. G(T) for everyone, same method
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
            flags.append(f"{p.tag}: {g.get('message')} - dropped")
            continue
        flags += [f"{p.tag}: {f}" for f in g["flags"]]
        comps[p.tag] = (p.composition, g["G_eV_per_atom"])
        labels[p.tag] = (p.phase_dir.energy() or {}).get("calculator", "?")
    tlabel = (target_dir.energy() or {}).get("calculator", "?")
    mism = sorted({v for v in labels.values() if v != tlabel})
    if mism:
        flags.append(f"energy-scale mismatch: target {tlabel} vs competitors {mism} "
                     "- hull distances are not comparable across calculators")

    # 6. hull
    try:
        scan = hull_scan(target.composition, gt["G_eV_per_atom"], comps)
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "status": STATUS_MISSING,
                "message": f"hull could not be built: {exc}", "flags": flags}
    e_T = float(np.interp(op, T, scan["e_hull_eV"]))
    i_op = int(np.argmin(np.abs(T - op)))
    decomp = scan["decomposition"][i_op]

    # 7. anharmonic overlay for the target
    overlay = None
    if hc.target_overlay.upper() == "VDOS" and method != "VDOS":
        try:
            gv = free_energy(target_dir, T, "VDOS")
            if gv.get("available"):
                # dF_anh (HA subtracted *with the same ZPE/T grid*)
                dF = gv["G_eV_per_atom"] - gt["G_eV_per_atom"]
                overlay = {"dF_anh_eV": dF, "e_hull_eV": scan["e_hull_eV"] + dF,
                           "e_hull_at_T_eV": float(np.interp(op, T, scan["e_hull_eV"] + dF)),
                           "md_temperatures": target_dir.md_temperatures()}
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

def zero_k_hull(target_dir, device: str | None = None, prepare: bool = True, force: bool | None = None,
                mp_docs: list | None = None) -> dict:
    """Signed distance of the target's relaxed enthalpy to the 0 K hull of its competitors.

    Needs only relaxations (cheap).  ``{"available": False, "message": ...}`` when the hull cannot
    be built (no competitors / missing elemental references).
    """
    target_dir = PhaseDir.of(target_dir)
    target = phase_from_dir(target_dir, "target")
    if target is None or not target_dir.has_relaxation():
        return {"available": False, "message": "target has no converged relaxation"}
    comps = find_competing_phases(target_dir, mp_docs)
    comps = _prepare(comps, "relax", device, force) if prepare else [
        p for p in comps if p.phase_dir.has_relaxation()]
    ref = {}
    labels = set()
    for p in comps:
        h = reference_energy(p.phase_dir)
        if h is not None:
            ref[p.tag] = (p.composition, h)
            labels.add((p.phase_dir.energy() or {}).get("calculator", "?"))
    flags = []
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
