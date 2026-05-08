"""heatup.thermodynamic
==============================
Gate 3: Temperature-dependent thermodynamic stability via the convex hull.

This module:

1. **Discovers secondary phases** — scans both the simulation database and
   the candidates tree for all structures whose element set is a subset of
   the target's elements.

2. **Generates missing polymorphs** — for each sub-composition of the target
   formula (including pure elements and all binary/ternary sub-systems),
   iterates over all 230 ITA space groups and uses PyXtal to generate a
   random crystal for any (formula, space-group) pair not yet present in the
   candidates tree.  New structures are saved as ``POSCAR + metadata.json``.

3. **Prepares competing phases** — ensures every competing phase has a
   MACE-relaxed energy (``relaxation/energy.json``) and a harmonic phonon
   DOS (``phonons/dos.json``), triggering the calculations via isolated
   subprocesses if needed.

4. **Computes F(T)** for all phases:
   - **Target**: anharmonic F(T) from AIMD VACF/VDOS (via
     :mod:`heatup.vdos`), with transparent fallback to harmonic
     if no AIMD data are available.
   - **Competing phases**: harmonic F(T) from the stored DOS.  This
     asymmetric choice is intentional: competing phases rarely have AIMD
     trajectories, and the harmonic approximation introduces a consistent
     systematic offset that largely cancels in hull-distance differences.

5. **Builds the pymatgen convex hull** at each temperature in
   ``config.HULL_TEMPERATURES`` and evaluates E_above_hull at
   ``operating_T``.

The hull results are written to ``<sym_dir>/stability/hull_vs_T.json`` and
the list of discovered secondary phases to
``<sym_dir>/stability/secondary_phases.json``.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import warnings
from itertools import chain, combinations

import numpy as np

from heatup import config


# ---------------------------------------------------------------------------
# Free-energy helpers
# ---------------------------------------------------------------------------

def _load_e0(sym_dir: str) -> float | None:
    """Return MACE ground-state energy per atom, or None if missing."""
    path = os.path.join(sym_dir, "relaxation", "energy.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return float(json.load(fh)["energy_eV_per_atom"])


def _load_harmonic_dos(
        sym_dir: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load and normalise the harmonic phonon DOS."""
    path = os.path.join(sym_dir, "phonons", "dos.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        d = json.load(fh)
    en = np.array(d["energies_eV"], dtype=float)
    wt = np.array(d["weights"],     dtype=float)
    mask = en > config.OMEGA_MIN_MEV * config.MEV_TO_EV
    en, wt = en[mask], wt[mask]
    norm = np.trapz(wt, en)
    if norm > 0:
        wt /= norm
    return en, wt


def _harmonic_f_vib(
        energies: np.ndarray,
        weights: np.ndarray,
        T: float,
) -> float:
    """Harmonic vibrational Helmholtz free energy per atom in eV."""
    if T <= 0.0:
        return float(np.trapz(0.5 * weights * energies, energies))
    kT = config.KB_EV * T
    x  = energies / (2.0 * kT)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        integrand = weights * np.log(
            2.0 * np.sinh(np.clip(x, 1e-12, 500.0))
        )
    return float(kT * np.trapz(integrand, energies))


def _harmonic_free_energy(
        sym_dir: str,
        temperatures: list[float],
) -> dict | None:
    """Compute harmonic F(T) = E0 + F_vib(T).  Returns None if data absent."""
    e0 = _load_e0(sym_dir)
    if e0 is None:
        return None
    dos = _load_harmonic_dos(sym_dir)
    if dos is None:
        return None
    en, wt = dos
    Fs = [e0 + _harmonic_f_vib(en, wt, T) for T in temperatures]
    return {"E0_eV_per_atom": e0, "temperatures": temperatures, "F_eV_per_atom": Fs}


def _gibbs_free_energy(
        sym_dir: str,
        temperatures: list[float],
        device: str,
        phonon_mode: str = "anharmonic",
        assembler=None,
) -> dict | None:
    """Compute G(T) using the GibbsAssembler.

    Uses the full generalised free-energy assembler which can include
    vibrational, electronic, magnetic, configurational, and PV contributions.
    Falls back to harmonic-only if the assembler is not provided.

    Args:
        sym_dir:      Symmetry directory.
        temperatures: Temperature grid (K).
        device:       Compute device for AIMD.
        phonon_mode:  "anharmonic" or "harmonic".
        assembler:    Optional pre-configured GibbsAssembler.  If None, the
                      default assembler (all contributions, anharmonic vib) is
                      built automatically.

    Returns:
        Free-energy dict with "E0_eV_per_atom", "temperatures",
        "F_eV_per_atom", or None if E0 is unavailable.
    """
    if assembler is None:
        from heatup.free_energy import build_default_assembler
        assembler = build_default_assembler(phonon_mode=phonon_mode, device=device)
    result = assembler.compute(sym_dir, temperatures)
    if result.get("E0_eV_per_atom") is None:
        return None
    return result


def _anharmonic_free_energy(
        sym_dir: str,
        temperatures: list[float],
        device: str,
) -> dict | None:
    """Convenience wrapper: anharmonic Gibbs free energy via GibbsAssembler."""
    return _gibbs_free_energy(sym_dir, temperatures, device, phonon_mode="anharmonic")


# ---------------------------------------------------------------------------
# Secondary-phase discovery
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
    """Return (reduced_formula, {element: mole_fraction}) for a POSCAR."""
    from pymatgen.io.vasp import Poscar
    s    = Poscar.from_file(poscar_path).structure
    comp = s.composition
    tot  = comp.num_atoms
    fracs = {str(el): amt / tot for el, amt in comp.items()}
    return comp.reduced_formula, fracs


def find_secondary_phases(
        target_sym_dir: str,
        candidates_root: str = config.CANDIDATES_ROOT,
        database_root: str = config.DATABASE_ROOT,
) -> list[dict]:
    """Return all competing phases whose elements ⊆ target elements.

    Scans both ``database_root`` and ``candidates_root``.  The target itself
    is excluded.

    Args:
        target_sym_dir:  Path to the target symmetry directory.
        candidates_root: Root of the candidate POSCAR tree.
        database_root:   Root of the simulation database.

    Returns:
        List of phase dicts with keys ``'sym_dir'``, ``'poscar_path'``,
        ``'formula'``, ``'composition'``, ``'elements'``, ``'source'``.
    """
    target_poscar   = os.path.join(target_sym_dir, "POSCAR")
    target_elements = _elements_of_poscar(target_poscar)
    target_abs      = os.path.abspath(target_sym_dir)

    phases: list[dict] = []
    seen:   set[str]   = set()

    def _try_add(sym_dir: str, poscar: str, source: str) -> None:
        abs_dir = os.path.abspath(sym_dir)
        if abs_dir == target_abs or abs_dir in seen:
            return
        els = _elements_of_poscar(poscar)
        if not els or not els.issubset(target_elements):
            return
        try:
            formula, comp = _formula_from_poscar(poscar)
        except Exception:
            return
        seen.add(abs_dir)
        phases.append({
            "sym_dir"    : sym_dir,
            "poscar_path": poscar,
            "formula"    : formula,
            "composition": comp,
            "elements"   : sorted(els),
            "source"     : source,
        })

    for root, src_label in (
        (database_root,   "database"),
        (candidates_root, "candidates"),
    ):
        if not os.path.isdir(root):
            continue
        for mat in sorted(os.listdir(root)):
            mat_dir = os.path.join(root, mat)
            if not os.path.isdir(mat_dir):
                continue
            for sym in sorted(os.listdir(mat_dir)):
                sd     = os.path.join(mat_dir, sym)
                poscar = os.path.join(sd, "POSCAR")
                if os.path.exists(poscar):
                    _try_add(sd, poscar, src_label)

    return phases


# ---------------------------------------------------------------------------
# Secondary-phase generation (PyXtal)
# ---------------------------------------------------------------------------

def _parse_formula(formula: str) -> dict[str, int]:
    """Parse a simple chemical formula string into {element: count}."""
    tokens = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    result: dict[str, int] = {}
    for element, count_str in tokens:
        if not element:
            continue
        result[element] = result.get(element, 0) + (int(count_str) if count_str else 1)
    if not result:
        raise ValueError(f"Could not parse formula: {formula!r}")
    return result


def _powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)))


def generate_missing_secondary_phases(
        target_sym_dir: str,
        candidates_root: str,
        existing_phases: list[dict],
) -> list[dict]:
    """Generate POSCAR + metadata.json for missing secondary-phase polymorphs.

    For every sub-composition of the target (all non-empty subsets of the
    target's element set, with 1:1:… stoichiometry) and for every ITA space
    group 1–230, generates a random crystal with PyXtal if no POSCAR exists.

    Args:
        target_sym_dir:  Target symmetry directory.
        candidates_root: Root of the candidates tree.
        existing_phases: Already-known competing phases (avoids duplicates).

    Returns:
        List of newly generated phase dicts (same schema as ``find_secondary_phases``).
    """
    try:
        from pyxtal import pyxtal
        from pymatgen.io.vasp import Poscar as PmgPoscar
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        from pymatgen.symmetry.groups import SpaceGroup
    except ImportError as exc:
        print(f"  [warn] PyXtal not available ({exc}) — skipping phase generation.")
        return []

    target_poscar = os.path.join(target_sym_dir, "POSCAR")
    try:
        tgt_formula, _ = _formula_from_poscar(target_poscar)
        tgt_elements   = sorted(_elements_of_poscar(target_poscar))
    except Exception as exc:
        print(f"  [warn] Cannot parse target POSCAR: {exc}")
        return []

    # Build space-group lookup table.
    sg_num2name: dict[int, str] = {}
    for n in range(1, 231):
        try:
            sg_num2name[n] = SpaceGroup.from_int_number(n).symbol
        except Exception:
            sg_num2name[n] = f"SG{n}"

    # Set of (formula, sg_name) pairs already in the candidates tree.
    existing_keys: set[tuple[str, str]] = {
        (ph["formula"], os.path.basename(ph["sym_dir"]))
        for ph in existing_phases
    }

    # Load stoichiometry hints for this element set.
    # Format: {"Li-P-S": [[3,1,4],[1,1,3]], "Li-Zr-O": [[7,2,12]]}
    extra_stoichs: dict[frozenset, list[list[int]]] = {}
    hints_path = config.STOICHIOMETRY_HINTS_PATH
    if os.path.exists(hints_path):
        try:
            with open(hints_path) as _fh:
                hints_raw = json.load(_fh)
            for key_str, stoich_list in hints_raw.items():
                key_els = frozenset(key_str.split("-"))
                extra_stoichs[key_els] = stoich_list
        except Exception as _exc:
            pass

    struc = pyxtal()
    generated: list[dict] = []

    for el_subset in _powerset(tgt_elements):
        atoms    = list(el_subset)
        n_base   = len(atoms)
        # Stoichiometry list: always include 1:1:... plus any hints
        stoich_list = [[1] * n_base]
        hint_key = frozenset(el_subset)
        if hint_key in extra_stoichs:
            stoich_list += extra_stoichs[hint_key]

        for numIons in stoich_list:
          if len(numIons) != n_base:
              numIons = [1] * n_base  # safety
          for sg_num in range(1, 231):
            sg_name = sg_num2name[sg_num]
            gen_struc = None

            for mult in range(1, max(1, config.PYXTAL_MAX_ATOMS // max(n_base,1)) + 1):
                actual_ions = [n * mult for n in numIons]
                for _ in range(config.PYXTAL_MAX_ATTEMPTS):
                    try:
                        struc.from_random(3, sg_num, atoms, actual_ions)
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
                pmg_struct = gen_struc.to_pymatgen()
                actual_sg  = SpacegroupAnalyzer(pmg_struct).get_space_group_symbol()
                formula    = pmg_struct.composition.reduced_formula
            except Exception:
                actual_sg = sg_name
                formula   = "".join(
                    f"{el}{n}" if n > 1 else el
                    for el, n in zip(atoms, numIons)
                )

            if (formula, actual_sg) in existing_keys:
                continue

            out_dir    = os.path.join(candidates_root, formula, actual_sg)
            poscar_out = os.path.join(out_dir, "POSCAR")
            meta_out   = os.path.join(out_dir, "metadata.json")

            if os.path.exists(poscar_out):
                existing_keys.add((formula, actual_sg))
                continue

            os.makedirs(out_dir, exist_ok=True)
            try:
                PmgPoscar(pmg_struct).write_file(poscar_out)
                with open(meta_out, "w") as fh:
                    json.dump({
                        "material_id"    : "coverage",
                        "formula"        : formula,
                        "symmetry"       : actual_sg,
                        "energy_per_atom": None,
                        "band_gap"       : None,
                    }, fh, indent=2)
                existing_keys.add((formula, actual_sg))
                generated.append({
                    "sym_dir"    : out_dir,
                    "poscar_path": poscar_out,
                    "formula"    : formula,
                    "composition": {el: 1.0 / len(atoms) for el in atoms},
                    "elements"   : atoms,
                    "source"     : "generated",
                })
            except Exception:
                import shutil
                shutil.rmtree(out_dir, ignore_errors=True)

    print(f"    Generated {len(generated)} new secondary-phase POSCAR(s).")
    return generated


# ---------------------------------------------------------------------------
# Subprocess helpers (CUDA-isolated)
# ---------------------------------------------------------------------------

def _project_root() -> str:
    """Project root: parent of library/ or this file's directory."""
    this_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(this_dir) == "library":
        return os.path.dirname(this_dir)
    # Walk up until we find a library/ sibling.
    candidate = this_dir
    for _ in range(4):
        if os.path.isdir(os.path.join(candidate, "library")):
            return candidate
        candidate = os.path.dirname(candidate)
    return this_dir


def _cuda_env() -> dict:
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    return env


def _run_structure_step(sym_dir: str, step: str, device: str) -> int:
    root   = _project_root()
    script = os.path.join(root, "library", "run_single_structure.py")
    if not os.path.exists(script):
        script = os.path.join(root, "run_single_structure.py")
    cmd = [sys.executable, script, sym_dir, "--step", step, "--device", device]
    return subprocess.run(cmd, env=_cuda_env()).returncode


def _ensure_phase_prepared(ph: dict, device: str) -> bool:
    """Ensure a competing phase has relaxation/energy.json + phonons/dos.json."""
    sd = ph["sym_dir"]
    if not os.path.exists(os.path.join(sd, "relaxation", "energy.json")):
        _run_structure_step(sd, "relaxation", device)
    if not os.path.exists(os.path.join(sd, "phonons", "dos.json")):
        _run_structure_step(sd, "phonons", device)
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
    """Build the pymatgen convex hull at a single temperature T."""
    from pymatgen.core import Composition
    from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram

    entries = []
    for ph in competing_phases:
        fe = free_energies.get(ph["sym_dir"])
        if fe is None:
            continue
        Ts_i = np.array(fe["temperatures"])
        Fs_i = np.array(fe["F_eV_per_atom"])
        F_at_T = float(np.interp(T, Ts_i, Fs_i))
        comp   = Composition(ph["formula"])
        entries.append(PDEntry(comp, F_at_T * comp.num_atoms, name=ph["formula"]))

    if not entries:
        return {"T": T, "e_above_hull_eV_atom": None, "stable": None}

    try:
        pd = PhaseDiagram(entries)
        tgt_fe = free_energies[target_phase["sym_dir"]]
        F_tgt  = float(np.interp(
            T,
            np.array(tgt_fe["temperatures"]),
            np.array(tgt_fe["F_eV_per_atom"]),
        ))
        tgt_comp  = Composition(target_phase["formula"])
        tgt_entry = PDEntry(tgt_comp, F_tgt * tgt_comp.num_atoms, name="TARGET")
        e_hull    = pd.get_e_above_hull(tgt_entry)
        return {
            "T"                    : T,
            "e_above_hull_eV_atom" : float(e_hull),
            "stable"               : bool(e_hull < 1e-4),
            "hull_phases"          : [e.name for e in pd.stable_entries],
            "n_entries"            : len(entries),
        }
    except Exception as exc:
        return {
            "T"                   : T,
            "e_above_hull_eV_atom": None,
            "stable"              : None,
            "error"               : str(exc),
        }


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
    """Build the T-dependent convex hull and assess stability at operating_T.

    Args:
        sym_dir:          Target symmetry directory.
        operating_T:      Temperature (K) at which to evaluate the hull.
        candidates_root:  Root of the candidates tree.
        database_root:    Root of the simulation database.
        temperatures:     Temperature grid (K).  Defaults to
                          ``config.HULL_TEMPERATURES``.
        device:           Compute device for MACE if calculations are needed.
        generate_missing: If True, run PyXtal to generate missing polymorphs
                          for all sub-compositions before building the hull.

    Returns:
        Dict with keys ``'available'``, ``'e_above_hull_at_T_eV'``,
        ``'operating_T_K'``, ``'hull_results'``, ``'n_competing'``,
        ``'n_generated'``, ``'status'``, ``'message'``.
    """
    if temperatures is None:
        temperatures = [float(t) for t in config.HULL_TEMPERATURES]

    # ── Target prerequisites ──────────────────────────────────────────────
    target_poscar = os.path.join(sym_dir, "POSCAR")
    if not os.path.exists(target_poscar):
        return {
            "available": False,
            "status"   : config.STATUS_MISSING,
            "message"  : "Target POSCAR not found.",
        }
    try:
        tgt_formula, tgt_comp = _formula_from_poscar(target_poscar)
    except Exception as exc:
        return {
            "available": False,
            "status"   : config.STATUS_MISSING,
            "message"  : f"Cannot read target POSCAR: {exc}",
        }

    if _load_e0(sym_dir) is None:
        return {
            "available": False,
            "status"   : config.STATUS_MISSING,
            "message"  : "relaxation/energy.json missing — run relaxation first.",
        }

    target_phase = {
        "sym_dir"    : sym_dir,
        "poscar_path": target_poscar,
        "formula"    : tgt_formula,
        "composition": tgt_comp,
        "elements"   : sorted(_elements_of_poscar(target_poscar)),
        "source"     : "target",
    }

    # ── Discover secondary phases ─────────────────────────────────────────
    print(f"    Scanning secondary phases "
          f"(elements ⊆ {set(target_phase['elements'])})...")
    existing  = find_secondary_phases(sym_dir, candidates_root, database_root)
    print(f"    Found {len(existing)} existing competing phase(s).")

    generated: list[dict] = []
    if generate_missing:
        print("    Generating missing polymorphs with PyXtal...")
        generated = generate_missing_secondary_phases(
            sym_dir, candidates_root, existing
        )
    competing = existing + generated

    # Persist secondary-phases list.
    stab_dir = os.path.join(sym_dir, "stability")
    os.makedirs(stab_dir, exist_ok=True)
    with open(os.path.join(stab_dir, "secondary_phases.json"), "w") as fh:
        json.dump(
            [{k: v for k, v in ph.items() if k != "composition"} for ph in competing],
            fh, indent=4,
        )

    # ── Prepare competing phases ──────────────────────────────────────────
    print(f"    Preparing {len(competing)} competing phase(s)...")
    usable = [ph for ph in competing if _ensure_phase_prepared(ph, device)]
    print(f"    {len(usable)} phase(s) ready for hull.")

    # ── Compute F(T) ─────────────────────────────────────────────────────
    free_energies: dict[str, dict] = {}

    # Target — anharmonic (with harmonic fallback).
    tgt_fe = _anharmonic_free_energy(sym_dir, temperatures, device)
    if tgt_fe is None:
        return {
            "available": False,
            "status"   : config.STATUS_MISSING,
            "message"  : "Cannot compute free energy for target.",
        }
    free_energies[sym_dir] = tgt_fe

    # Competing phases — upgrade to anharmonic if AIMD data available, 
    # otherwise use harmonic.  Log which source was used for transparency.
    fe_sources: dict[str, str] = {}
    for ph in usable:
        aimd_dir_ph = os.path.join(ph["sym_dir"], "aimd")
        has_vdos = False
        if os.path.isdir(aimd_dir_ph):
            for tf in os.listdir(aimd_dir_ph):
                if tf.endswith("K") and os.path.exists(
                    os.path.join(aimd_dir_ph, tf, "anharmonic_phonons", "vdos.json")
                ):
                    has_vdos = True
                    break
        if has_vdos:
            fe = _anharmonic_free_energy(ph["sym_dir"], temperatures, device)
            source = "anharmonic"
        else:
            fe = _harmonic_free_energy(ph["sym_dir"], temperatures)
            source = "harmonic"
        if fe is not None:
            free_energies[ph["sym_dir"]] = fe
            fe_sources[ph["sym_dir"]] = source
            print(f"      [{source}] {ph['formula']} ({os.path.basename(ph['sym_dir'])})")

    # ── Build hull ────────────────────────────────────────────────────────
    print(f"    Building convex hull at {len(temperatures)} temperatures...")
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
    valid = [
        (r["T"], r["e_above_hull_eV_atom"])
        for r in hull_results
        if r.get("e_above_hull_eV_atom") is not None
    ]
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
        message = (
            f"E_above_hull = {e_at_T * 1000:.1f} meV/atom at {operating_T:.0f} K "
            f"(threshold {config.THERMO_HULL_WARN_EV * 1000:.0f} meV/atom) — unstable."
        )
    elif e_at_T > config.THERMO_HULL_STABLE_EV + 1e-4:
        status  = config.STATUS_WARN
        message = (
            f"E_above_hull = {e_at_T * 1000:.1f} meV/atom at {operating_T:.0f} K "
            f"— metastable."
        )
    else:
        status  = config.STATUS_OK
        message = (
            f"On the convex hull at {operating_T:.0f} K "
            f"(E_above_hull = {e_at_T * 1000:.1f} meV/atom)."
        )

    return {
        "available"            : True,
        "e_above_hull_at_T_eV" : e_at_T,
        "operating_T_K"        : operating_T,
        "hull_results"         : hull_results,
        "n_competing"          : len(usable),
        "n_generated"          : len(generated),
        "status"               : status,
        "message"              : message,
    }
