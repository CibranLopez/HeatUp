"""heatup.thermodynamic
========================
Gate 3: Temperature-dependent thermodynamic stability via the convex hull.

This module:

1. **Discovers secondary phases** from up to four sources (in priority order):

   a. **Materials Project API** — downloads all structures in the target
      chemical space directly from MP.  Most reliable; requires an API key
      (``config.MP_API_KEY`` or ``MP_API_KEY`` env var).

   b. **Local simulation database** — scans ``database_root`` for structures
      whose element set ⊆ target elements.

   c. **Local candidates tree** — scans ``candidates_root`` for the same.

   d. **PyXtal random generation** — for every sub-composition not already
      covered, generates random crystal structures for all 230 ITA space
      groups.  Provides systematic coverage when MP and local data are
      incomplete, but is slow.

   Which sources are active is controlled by ``config.COMPETING_PHASE_SOURCES``
   (a list that can be freely reordered or trimmed).  Removing ``"mp-api"``
   skips the download; removing ``"pyxtal"`` skips generation entirely.

2. **Prepares competing phases** — ensures every phase has a MACE-relaxed
   energy (``relaxation/energy.json``) and a harmonic phonon DOS
   (``phonons/dos.json``), triggering the calculations if needed.

3. **Computes F(T)** for all phases:

   - **Target**: anharmonic F(T) from AIMD VACF/VDOS
     (via :mod:`heatup.anharmonic_phonons`), with transparent fallback to
     harmonic if no AIMD data are available.

   - **Competing phases**: harmonic F(T) from their stored DOS.  This
     asymmetric choice is intentional — competing phases rarely have AIMD
     trajectories, and the harmonic approximation introduces a consistent
     systematic offset that largely cancels in hull-distance differences.
     If AIMD VDOS is available for a competing phase, anharmonic F(T) is
     used automatically.

4. **Builds the pymatgen convex hull** at each temperature in
   ``config.HULL_TEMPERATURES`` and evaluates E_above_hull at ``operating_T``.

Outputs::

    <sym_dir>/stability/
        stability_report.json   ← written by heatup.pipeline
        secondary_phases.json   ← list of all competing phases used
        hull_vs_T.json          ← hull results at every requested temperature
"""

from __future__ import annotations

import json
import os
import re
import warnings
from itertools import chain, combinations

import numpy as np

from heatup import config


# ---------------------------------------------------------------------------
# Free-energy helpers
# ---------------------------------------------------------------------------

def _load_e0(sym_dir: str) -> float | None:
    """Return the MLIP ground-state energy per atom (eV), or None."""
    path = os.path.join(sym_dir, "relaxation", "energy.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return float(json.load(fh)["energy_eV_per_atom"])


def _load_harmonic_dos(sym_dir: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Load and normalise the harmonic phonon DOS from ``phonons/dos.json``."""
    path = os.path.join(sym_dir, "phonons", "dos.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        d = json.load(fh)
    en = np.array(d["energies_eV"], dtype=float)
    wt = np.array(d["weights"],     dtype=float)
    mask = en > config.OMEGA_MIN_MEV * config.MEV_TO_EV
    en, wt = en[mask], wt[mask]
    norm = np.trapezoid(wt, en)
    if norm > 0:
        wt /= norm
    return en, wt


def _harmonic_f_vib(en: np.ndarray, wt: np.ndarray, T: float) -> float:
    """Harmonic vibrational Helmholtz free energy per atom (eV) at temperature T."""
    if T <= 0.0:
        return float(np.trapezoid(0.5 * wt * en, en))
    kT = config.KB_EV * T
    x  = en / (2.0 * kT)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        integrand = wt * np.log(2.0 * np.sinh(np.clip(x, 1e-12, 500.0)))
    return float(kT * np.trapezoid(integrand, en))


def _harmonic_free_energy(sym_dir: str, temperatures: list[float]) -> dict | None:
    """Compute harmonic F(T) = E0 + F_vib(T).  Returns None if data absent."""
    e0 = _load_e0(sym_dir)
    if e0 is None:
        return None
    dos = _load_harmonic_dos(sym_dir)
    if dos is None:
        return None
    en, wt = dos
    return {
        "E0_eV_per_atom" : e0,
        "temperatures"   : temperatures,
        "F_eV_per_atom"  : [e0 + _harmonic_f_vib(en, wt, T) for T in temperatures],
    }


def _anharmonic_free_energy(
        sym_dir: str,
        temperatures: list[float],
        device: str,
) -> dict | None:
    """Compute anharmonic G(T) via the GibbsAssembler."""
    from heatup.free_energy import build_default_assembler
    asm    = build_default_assembler(phonon_mode="anharmonic", device=device)
    result = asm.compute(sym_dir, temperatures)
    return result if result.get("E0_eV_per_atom") is not None else None


# ---------------------------------------------------------------------------
# Structure parsing helpers
# ---------------------------------------------------------------------------

def _elements_of_poscar(poscar_path: str) -> frozenset[str]:
    """Return element symbols present in a POSCAR, or empty frozenset."""
    try:
        from pymatgen.io.vasp import Poscar
        s = Poscar.from_file(poscar_path).structure
        return frozenset(str(sp) for sp in s.composition.elements)
    except Exception:
        return frozenset()


def _formula_from_poscar(poscar_path: str) -> tuple[str, dict[str, float]]:
    """Return ``(reduced_formula, {element: mole_fraction})`` for a POSCAR."""
    from pymatgen.io.vasp import Poscar
    s    = Poscar.from_file(poscar_path).structure
    comp = s.composition
    tot  = comp.num_atoms
    fracs = {str(el): amt / tot for el, amt in comp.items()}
    return comp.reduced_formula, fracs


# ---------------------------------------------------------------------------
# Secondary-phase discovery
# ---------------------------------------------------------------------------

def _discover_from_local(
        target_sym_dir: str,
        target_elements: frozenset[str],
        roots: list[tuple[str, str]],
) -> list[dict]:
    """Scan local directory trees for competing phases.

    Args:
        target_sym_dir:   Target symmetry directory (excluded from results).
        target_elements:  Element set of the target.
        roots:            List of ``(root_path, source_label)`` pairs to scan.

    Returns:
        List of phase dicts with keys ``sym_dir``, ``poscar_path``,
        ``formula``, ``composition``, ``elements``, ``source``.
    """
    target_abs = os.path.abspath(target_sym_dir)
    phases: list[dict] = []
    seen:   set[str]   = set()

    for root, label in roots:
        if not os.path.isdir(root):
            continue
        for mat in sorted(os.listdir(root)):
            mat_dir = os.path.join(root, mat)
            if not os.path.isdir(mat_dir):
                continue
            for sym in sorted(os.listdir(mat_dir)):
                sd     = os.path.join(mat_dir, sym)
                poscar = os.path.join(sd, "POSCAR")
                if not os.path.exists(poscar):
                    continue
                abs_sd = os.path.abspath(sd)
                if abs_sd == target_abs or abs_sd in seen:
                    continue
                els = _elements_of_poscar(poscar)
                if not els or not els.issubset(target_elements):
                    continue
                try:
                    formula, comp = _formula_from_poscar(poscar)
                except Exception:
                    continue
                seen.add(abs_sd)
                phases.append({
                    "sym_dir"    : sd,
                    "poscar_path": poscar,
                    "formula"    : formula,
                    "composition": comp,
                    "elements"   : sorted(els),
                    "source"     : label,
                })
    return phases


def _discover_from_mp(
        target_elements: frozenset[str],
        candidates_root: str,
        existing_sym_dirs: set[str],
) -> list[dict]:
    """Download competing phases from the Materials Project API.

    Downloads all materials in the target chemical space (all sub-compositions)
    that are not already present locally.  Structures are saved to
    ``candidates_root/<formula>/<sg>/POSCAR`` for reuse.

    Args:
        target_elements:   Element set of the target phase.
        candidates_root:   Root of the candidates POSCAR tree.
        existing_sym_dirs: Absolute paths already discovered (to avoid duplicates).

    Returns:
        List of newly downloaded phase dicts.
    """
    api_key = config.MP_API_KEY.strip()
    if not api_key:
        print("  [mp-api] MP_API_KEY is not set — skipping Materials Project download.")
        print("           Set it with: import heatup.config as cfg; "
              "cfg.MP_API_KEY = 'YOUR_KEY'")
        return []

    try:
        from mp_api.client import MPRester
        from pymatgen.io.vasp import Poscar as PmgPoscar
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    except ImportError as exc:
        print(f"  [mp-api] mp_api or pymatgen not available ({exc}) — skipping MP download.")
        return []

    elements = sorted(target_elements)
    print(f"  [mp-api] Querying Materials Project for elements {elements} ...")

    phases: list[dict] = []
    try:
        with MPRester(api_key) as mpr:
            # Fetch all materials whose element set ⊆ target_elements.
            docs = mpr.materials.summary.search(
                elements   = elements,
                fields     = ["material_id", "formula_pretty", "structure",
                              "energy_per_atom", "band_gap"],
                all_fields = False,
            )
    except Exception as exc:
        print(f"  [mp-api] MP query failed: {exc}")
        return []

    print(f"  [mp-api] Received {len(docs)} structures from MP.")
    for doc in docs:
        if not doc.structure:
            continue
        # Verify the element subset constraint (MP may return supersets).
        doc_els = frozenset(str(el) for el in doc.structure.composition.elements)
        if not doc_els.issubset(target_elements):
            continue

        try:
            sga      = SpacegroupAnalyzer(doc.structure)
            sg_sym   = sga.get_space_group_symbol()
            formula  = doc.structure.composition.reduced_formula
            fracs    = {str(el): doc.structure.composition[el] / doc.structure.composition.num_atoms
                        for el in doc.structure.composition.elements}
        except Exception:
            continue

        out_dir = os.path.join(candidates_root, formula, sg_sym)
        abs_dir = os.path.abspath(out_dir)
        if abs_dir in existing_sym_dirs:
            continue

        poscar_out = os.path.join(out_dir, "POSCAR")
        if not os.path.exists(poscar_out):
            os.makedirs(out_dir, exist_ok=True)
            try:
                PmgPoscar(doc.structure).write_file(poscar_out)
                with open(os.path.join(out_dir, "metadata.json"), "w") as fh:
                    json.dump({
                        "material_id"    : str(doc.material_id),
                        "formula"        : formula,
                        "symmetry"       : sg_sym,
                        "energy_per_atom": doc.energy_per_atom,
                        "band_gap"       : doc.band_gap,
                        "source"         : "mp-api",
                    }, fh, indent=2)
            except Exception as exc:
                import shutil
                shutil.rmtree(out_dir, ignore_errors=True)
                continue

        existing_sym_dirs.add(abs_dir)
        phases.append({
            "sym_dir"    : out_dir,
            "poscar_path": poscar_out,
            "formula"    : formula,
            "composition": fracs,
            "elements"   : sorted(doc_els),
            "source"     : "mp-api",
        })

    print(f"  [mp-api] Added {len(phases)} new structure(s).")
    return phases


def _powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)))


def _discover_from_pyxtal(
        target_elements: frozenset[str],
        candidates_root: str,
        existing_keys: set[tuple[str, str]],
) -> list[dict]:
    """Generate missing secondary-phase structures with PyXtal.

    For every sub-composition of the target element set and for every ITA
    space group 1–230, generates a random crystal if no POSCAR already exists.
    This provides systematic coverage without requiring any experimental or
    database knowledge of the stoichiometry.

    This replaces the former ``stoichiometry_hints.json`` mechanism: instead
    of manually listing non-trivial stoichiometries, all sub-compositions
    are enumerated automatically from the element set.

    Args:
        target_elements:  Element set of the target phase.
        candidates_root:  Root of the candidates POSCAR tree.
        existing_keys:    Set of ``(reduced_formula, sg_symbol)`` pairs already
                          present (avoids overwriting existing structures).

    Returns:
        List of newly generated phase dicts.
    """
    try:
        from pyxtal import pyxtal
        from pymatgen.io.vasp import Poscar as PmgPoscar
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        from pymatgen.symmetry.groups import SpaceGroup
    except ImportError as exc:
        print(f"  [pyxtal] PyXtal not available ({exc}) — skipping generation.")
        return []

    # Build sg number → symbol table.
    sg_num2name = {}
    for n in range(1, 231):
        try:
            sg_num2name[n] = SpaceGroup.from_int_number(n).symbol
        except Exception:
            sg_num2name[n] = f"SG{n}"

    struc     = pyxtal()
    generated: list[dict] = []
    elements  = sorted(target_elements)

    for el_subset in _powerset(elements):
        atoms = list(el_subset)
        n     = len(atoms)

        for sg_num in range(1, 231):
            sg_name = sg_num2name[sg_num]
            gen_struc = None

            for mult in range(1, config.PYXTAL_MAX_ATOMS // max(n, 1) + 1):
                ions = [mult] * n
                for _ in range(config.PYXTAL_MAX_ATTEMPTS):
                    try:
                        struc.from_random(3, sg_num, atoms, ions)
                        if struc.valid:
                            gen_struc = struc
                            break
                    except Exception:
                        pass
                if gen_struc is not None:
                    break

            if gen_struc is None:
                continue

            try:
                pmg_s   = gen_struc.to_pymatgen()
                actual_sg = SpacegroupAnalyzer(pmg_s).get_space_group_symbol()
                formula   = pmg_s.composition.reduced_formula
                fracs     = {str(el): pmg_s.composition[el] / pmg_s.composition.num_atoms
                             for el in pmg_s.composition.elements}
            except Exception:
                actual_sg = sg_name
                formula   = "".join(f"{el}{1}" if 1 == 1 else el for el in atoms)
                fracs     = {el: 1.0 / n for el in atoms}

            if (formula, actual_sg) in existing_keys:
                continue

            out_dir    = os.path.join(candidates_root, formula, actual_sg)
            poscar_out = os.path.join(out_dir, "POSCAR")
            if os.path.exists(poscar_out):
                existing_keys.add((formula, actual_sg))
                continue

            os.makedirs(out_dir, exist_ok=True)
            try:
                PmgPoscar(pmg_s).write_file(poscar_out)
                with open(os.path.join(out_dir, "metadata.json"), "w") as fh:
                    json.dump({"material_id": "pyxtal-generated", "formula": formula,
                               "symmetry": actual_sg, "energy_per_atom": None,
                               "band_gap": None, "source": "pyxtal"}, fh, indent=2)
                existing_keys.add((formula, actual_sg))
                generated.append({
                    "sym_dir"    : out_dir,
                    "poscar_path": poscar_out,
                    "formula"    : formula,
                    "composition": fracs,
                    "elements"   : sorted(frozenset(atoms)),
                    "source"     : "pyxtal",
                })
            except Exception:
                import shutil
                shutil.rmtree(out_dir, ignore_errors=True)

    print(f"  [pyxtal] Generated {len(generated)} new structure(s).")
    return generated


def find_secondary_phases(
        target_sym_dir: str,
        candidates_root: str = config.CANDIDATES_ROOT,
        database_root: str = config.DATABASE_ROOT,
) -> list[dict]:
    """Discover all competing phases for the target material.

    Combines results from the sources listed in ``config.COMPETING_PHASE_SOURCES``
    (default: ``["mp-api", "database", "candidates", "pyxtal"]``).  Each source
    is checked in order; the combined, de-duplicated list is returned.

    Removing a source from ``config.COMPETING_PHASE_SOURCES`` disables it
    entirely without code changes.

    Args:
        target_sym_dir:  Path to the target symmetry directory.
        candidates_root: Root of the candidate POSCAR tree.
        database_root:   Root of the simulation database.

    Returns:
        De-duplicated list of phase dicts with keys ``sym_dir``,
        ``poscar_path``, ``formula``, ``composition``, ``elements``, ``source``.
    """
    target_poscar   = os.path.join(target_sym_dir, "POSCAR")
    target_elements = _elements_of_poscar(target_poscar)
    if not target_elements:
        return []

    sources = config.COMPETING_PHASE_SOURCES
    all_phases: list[dict] = []

    # --- Local sources first (fast, no network) ---
    local_roots = []
    if "database"   in sources:
        local_roots.append((database_root,   "database"))
    if "candidates" in sources:
        local_roots.append((candidates_root, "candidates"))

    local = _discover_from_local(target_sym_dir, target_elements, local_roots)
    all_phases.extend(local)

    # Track absolute dirs and (formula, sg) keys for de-duplication.
    seen_dirs = {os.path.abspath(target_sym_dir)} | {os.path.abspath(p["sym_dir"]) for p in local}
    seen_keys = {(p["formula"], os.path.basename(p["sym_dir"])) for p in local}

    # --- Materials Project API ---
    if "mp-api" in sources:
        mp_phases = _discover_from_mp(target_elements, candidates_root, seen_dirs)
        for ph in mp_phases:
            key = (ph["formula"], os.path.basename(ph["sym_dir"]))
            if key not in seen_keys:
                all_phases.append(ph)
                seen_keys.add(key)

    # --- PyXtal random generation ---
    if "pyxtal" in sources:
        pyxtal_phases = _discover_from_pyxtal(target_elements, candidates_root, seen_keys)
        all_phases.extend(pyxtal_phases)

    return all_phases


# ---------------------------------------------------------------------------
# Phase preparation (relaxation + phonons)
# ---------------------------------------------------------------------------

def _ensure_phase_prepared(ph: dict, device: str) -> bool:
    """Ensure a competing phase has ``relaxation/energy.json`` + ``phonons/dos.json``.

    Triggers CUDA-isolated subprocess runs if either file is missing.

    Args:
        ph:     Phase dict with key ``"sym_dir"``.
        device: Compute device string.

    Returns:
        ``True`` if both files exist after preparation attempts.
    """
    from heatup.structure_pipeline import run_relaxation_subprocess, run_phonons_subprocess

    sd = ph["sym_dir"]
    if not os.path.exists(os.path.join(sd, "relaxation", "energy.json")):
        run_relaxation_subprocess(sd, device=device)
    if not os.path.exists(os.path.join(sd, "phonons", "dos.json")):
        run_phonons_subprocess(sd, device=device)

    return (
        os.path.exists(os.path.join(sd, "relaxation", "energy.json"))
        and os.path.exists(os.path.join(sd, "phonons", "dos.json"))
    )


# ---------------------------------------------------------------------------
# Convex hull
# ---------------------------------------------------------------------------

def _build_hull_at_T(
        target_phase: dict,
        competing_phases: list[dict],
        free_energies: dict[str, dict],
        T: float,
) -> dict:
    """Build the pymatgen convex hull at a single temperature T.

    Args:
        target_phase:     Phase dict for the target material.
        competing_phases: List of phase dicts for competitors.
        free_energies:    Mapping from ``sym_dir`` → free-energy dict.
        T:                Temperature in Kelvin.

    Returns:
        Dict with keys ``T``, ``e_above_hull_eV_atom``, ``stable``,
        ``hull_phases``, ``n_entries``.
    """
    from pymatgen.core import Composition
    from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram

    entries = []
    for ph in competing_phases:
        fe = free_energies.get(ph["sym_dir"])
        if fe is None:
            continue
        F_at_T = float(np.interp(T, fe["temperatures"], fe["F_eV_per_atom"]))
        comp   = Composition(ph["formula"])
        entries.append(PDEntry(comp, F_at_T * comp.num_atoms, name=ph["formula"]))

    if not entries:
        return {"T": T, "e_above_hull_eV_atom": None, "stable": None}

    try:
        pd      = PhaseDiagram(entries)
        tgt_fe  = free_energies[target_phase["sym_dir"]]
        F_tgt   = float(np.interp(T, tgt_fe["temperatures"], tgt_fe["F_eV_per_atom"]))
        tgt_comp = Composition(target_phase["formula"])
        tgt_entry = PDEntry(tgt_comp, F_tgt * tgt_comp.num_atoms, name="TARGET")
        e_hull  = pd.get_e_above_hull(tgt_entry)
        return {
            "T"                    : T,
            "e_above_hull_eV_atom" : float(e_hull),
            "stable"               : bool(e_hull < 1e-4),
            "hull_phases"          : [e.name for e in pd.stable_entries],
            "n_entries"            : len(entries),
        }
    except Exception as exc:
        return {"T": T, "e_above_hull_eV_atom": None, "stable": None,
                "error": str(exc)}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def assess_thermodynamic_stability(
        sym_dir: str,
        operating_T: float,
        candidates_root: str = config.CANDIDATES_ROOT,
        database_root: str = config.DATABASE_ROOT,
        temperatures: list[float] | None = None,
        device: str = config.DEFAULT_DEVICE,
        generate_missing: bool = True,
) -> dict:
    """Build the T-dependent convex hull and assess stability at *operating_T*.

    Steps:
    1. Discover secondary phases from all enabled sources
       (``config.COMPETING_PHASE_SOURCES``).
    2. Prepare each competitor (relaxation + harmonic phonons).
    3. Compute F(T) for the target (anharmonic) and all competitors (harmonic,
       or anharmonic if VDOS is available).
    4. Build the convex hull at every temperature in *temperatures*.
    5. Evaluate E_above_hull at *operating_T*.

    Args:
        sym_dir:          Target symmetry directory.
        operating_T:      Temperature (K) at which to evaluate the hull.
        candidates_root:  Root of the candidates tree.
        database_root:    Root of the simulation database.
        temperatures:     Temperature grid (K).  Defaults to
                          ``config.HULL_TEMPERATURES``.
        device:           Compute device for MACE if calculations are needed.
        generate_missing: If True, enable PyXtal and MP-API phase generation
                          (as permitted by ``config.COMPETING_PHASE_SOURCES``).

    Returns:
        Dict with keys ``available``, ``e_above_hull_at_T_eV``,
        ``operating_T_K``, ``hull_results``, ``n_competing``,
        ``n_generated``, ``status``, ``message``.
    """
    if temperatures is None:
        temperatures = [float(t) for t in config.HULL_TEMPERATURES]

    target_poscar = os.path.join(sym_dir, "POSCAR")
    if not os.path.exists(target_poscar):
        return {"available": False, "status": config.STATUS_MISSING,
                "message": "Target POSCAR not found."}
    try:
        tgt_formula, tgt_comp = _formula_from_poscar(target_poscar)
    except Exception as exc:
        return {"available": False, "status": config.STATUS_MISSING,
                "message": f"Cannot read target POSCAR: {exc}"}

    if _load_e0(sym_dir) is None:
        return {"available": False, "status": config.STATUS_MISSING,
                "message": "relaxation/energy.json missing — run relaxation first."}

    target_elements = _elements_of_poscar(target_poscar)
    target_phase = {
        "sym_dir"    : sym_dir,
        "poscar_path": target_poscar,
        "formula"    : tgt_formula,
        "composition": tgt_comp,
        "elements"   : sorted(target_elements),
        "source"     : "target",
    }

    # ── Discover competing phases ─────────────────────────────────────────
    print(f"    Scanning competing phases (elements ⊆ {set(target_phase['elements'])}) ...")

    # Temporarily restrict sources if generate_missing=False.
    _orig_sources = config.COMPETING_PHASE_SOURCES
    if not generate_missing:
        config.COMPETING_PHASE_SOURCES = [s for s in _orig_sources
                                          if s not in ("mp-api", "pyxtal")]
    try:
        competing = find_secondary_phases(sym_dir, candidates_root, database_root)
    finally:
        config.COMPETING_PHASE_SOURCES = _orig_sources

    n_generated = sum(1 for p in competing if p["source"] in ("pyxtal", "mp-api"))
    print(f"    Found {len(competing)} competing phase(s) "
          f"({n_generated} newly added).")

    # Persist secondary-phases list.
    stab_dir = os.path.join(sym_dir, "stability")
    os.makedirs(stab_dir, exist_ok=True)
    with open(os.path.join(stab_dir, "secondary_phases.json"), "w") as fh:
        json.dump(
            [{k: v for k, v in ph.items() if k != "composition"} for ph in competing],
            fh, indent=4,
        )

    # ── Prepare competing phases ──────────────────────────────────────────
    print(f"    Preparing {len(competing)} competing phase(s) ...")
    usable = [ph for ph in competing if _ensure_phase_prepared(ph, device)]
    print(f"    {len(usable)} phase(s) ready for hull.")

    # ── Compute F(T) ─────────────────────────────────────────────────────
    free_energies: dict[str, dict] = {}

    # Target — anharmonic (with harmonic fallback).
    tgt_fe = _anharmonic_free_energy(sym_dir, temperatures, device)
    if tgt_fe is None:
        return {"available": False, "status": config.STATUS_MISSING,
                "message": "Cannot compute free energy for target."}
    free_energies[sym_dir] = tgt_fe

    # Competitors — upgrade to anharmonic if VDOS is cached.
    for ph in usable:
        aimd_d   = os.path.join(ph["sym_dir"], "aimd")
        has_vdos = (
            os.path.isdir(aimd_d) and any(
                os.path.exists(os.path.join(aimd_d, tf, "anharmonic_phonons", "vdos.json"))
                for tf in os.listdir(aimd_d) if tf.endswith("K")
            )
        )
        fe = (_anharmonic_free_energy(ph["sym_dir"], temperatures, device)
              if has_vdos
              else _harmonic_free_energy(ph["sym_dir"], temperatures))
        if fe is not None:
            free_energies[ph["sym_dir"]] = fe
            src = "anharmonic" if has_vdos else "harmonic"
            print(f"      [{src}] {ph['formula']} ({os.path.basename(ph['sym_dir'])})")

    # ── Build hull ────────────────────────────────────────────────────────
    print(f"    Building convex hull at {len(temperatures)} temperatures ...")
    hull_results = [
        _build_hull_at_T(target_phase, usable, free_energies, T)
        for T in temperatures
    ]

    with open(os.path.join(stab_dir, "hull_vs_T.json"), "w") as fh:
        json.dump({
            "target"          : {k: v for k, v in target_phase.items()
                                 if k != "composition"},
            "competing_phases": [{k: v for k, v in ph.items() if k != "composition"}
                                 for ph in usable],
            "temperatures"    : temperatures,
            "hull_results"    : hull_results,
        }, fh, indent=4)

    # ── Evaluate at operating_T ───────────────────────────────────────────
    valid = [(r["T"], r["e_above_hull_eV_atom"]) for r in hull_results
             if r.get("e_above_hull_eV_atom") is not None]
    e_at_T = None
    if valid:
        Ts_arr = np.array([v[0] for v in valid])
        Es_arr = np.array([v[1] for v in valid])
        e_at_T = float(np.interp(operating_T, Ts_arr, Es_arr))

    if e_at_T is None:
        status  = config.STATUS_MISSING
        message = "Could not evaluate hull (insufficient competing phases)."
    elif e_at_T > config.THERMO_HULL_WARN_EV:
        status  = config.STATUS_FAIL
        message = (f"E_above_hull = {e_at_T * 1000:.1f} meV/atom at {operating_T:.0f} K "
                   f"(threshold {config.THERMO_HULL_WARN_EV * 1000:.0f} meV/atom) — unstable.")
    elif e_at_T > config.THERMO_HULL_STABLE_EV + 1e-4:
        status  = config.STATUS_WARN
        message = (f"E_above_hull = {e_at_T * 1000:.1f} meV/atom at {operating_T:.0f} K "
                   f"— metastable.")
    else:
        status  = config.STATUS_OK
        message = (f"On the convex hull at {operating_T:.0f} K "
                   f"(E_above_hull = {e_at_T * 1000:.1f} meV/atom).")

    return {
        "available"            : True,
        "e_above_hull_at_T_eV" : e_at_T,
        "operating_T_K"        : operating_T,
        "hull_results"         : hull_results,
        "n_competing"          : len(usable),
        "n_generated"          : n_generated,
        "status"               : status,
        "message"              : message,
    }
