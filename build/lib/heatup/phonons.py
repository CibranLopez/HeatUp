"""heatup.phonons
==================
Unified vibrational free-energy module supporting three phonon methods.

This module is the single entry point for **all** phonon calculations in
HeatUp.  The active method is selected by ``config.PHONON_MODE``:

+---------+-------------------------------------------------------------+
| Mode    | Description                                                 |
+=========+=============================================================+
| ``HA``  | **Harmonic Approximation** — finite-displacement phonon     |
|         | DOS at the relaxed volume V₀.  Fast; accurate at low T.    |
|         | Misses thermal expansion and anharmonic softening.          |
+---------+-------------------------------------------------------------+
| ``QHA`` | **Quasi-Harmonic Approximation** — phonon DOS at            |
|         | ``config.QHA_N_VOLUMES`` volumes around V₀.  Fits E(V,T)   |
|         | to extract thermal expansion α(T) and F(T,P) at the         |
|         | equilibrium volume.  Requires phonopy.                      |
+---------+-------------------------------------------------------------+
| ``VDOS``| **AIMD VDOS** (default) — vibrational density of states     |
|         | extracted from the velocity autocorrelation function of an  |
|         | AIMD trajectory.  Full anharmonicity at finite T.           |
+---------+-------------------------------------------------------------+

Switching modes
---------------
Change ``config.PHONON_MODE`` before running the pipeline::

    import heatup.config as cfg
    cfg.PHONON_MODE = "QHA"          # switch to quasi-harmonic
    cfg.QHA_N_VOLUMES = 9            # use 9 volume points
    cfg.PHONON_BACKEND = "phonopy"   # required for QHA

    from heatup.phonons import run_phonons
    run_phonons("database/LGPS/P42-nmc", device="cuda")

Output layout
-------------
All modes write to sub-directories of ``<sym_dir>/phonons/``:

HA::

    phonons/
        phonon-input.json          ← parameters + manifest
        POSCAR                     ← copy of relaxation/CONTCAR
        dos.json                   ← {"energies_eV": [...], "weights": [...]}
        band_structure.json        ← serialised band structure
        phonon.pdf                 ← band structure + DOS figure
        dos_manifest.json          ← full config snapshot

QHA::

    phonons/
        qha/
            qha-input.json         ← QHA parameters + manifest
            volume_*/
                POSCAR             ← strained structure
                dos.json           ← phonon DOS at this volume
            qha_result.json        ← F(V,T), α(T), V_eq(T), Grüneisen
            qha_manifest.json

VDOS::

    aimd/<T>K/
        anharmonic_phonons/
            vdos.json              ← {"omega_mev": [...], "vdos": [...]}
            thermo.json            ← Cv, S, F at MD temperature
            free_energy.json       ← F(T) over full temperature grid
            vdos_manifest.json     ← full config snapshot

Free-energy interface
---------------------
All three modes produce a dict compatible with :class:`heatup.free_energy.GibbsAssembler`::

    {
        "E0_eV_per_atom" : float,
        "temperatures"   : [float, ...],
        "F_eV_per_atom"  : [float, ...],
        "phonon_mode"    : "HA" | "QHA" | "VDOS",
        "source_files"   : [str, ...],     # paths used to compute this
    }
"""

from __future__ import annotations

import json
import os
import shutil
import traceback
import warnings
from typing import Sequence

import numpy as np

from heatup import config
from heatup.manifest import write_manifest


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------

def run_phonons(
        sym_dir: str,
        device: str = "cpu",
        force_rerun: bool = False,
        mode: str | None = None,
) -> bool:
    """Run the phonon calculation for *sym_dir* in the configured mode.

    Dispatches to :func:`run_ha`, :func:`run_qha`, or — for VDOS — does
    nothing here (VDOS is computed from the AIMD trajectory in
    :func:`run_vdos_for_sim`).

    Args:
        sym_dir:     Path to ``database/<material>/<symmetry>/``.
        device:      Compute device for the calculator.
        force_rerun: Redo even if output already exists.
        mode:        Override ``config.PHONON_MODE`` for this call.
                     One of ``"HA"``, ``"QHA"``, ``"VDOS"``.

    Returns:
        ``True`` on success.  For VDOS mode, always returns ``True`` because
        the VDOS computation depends on the AIMD trajectory and is triggered
        separately via :func:`run_vdos_for_sim`.
    """
    m = (mode or config.PHONON_MODE).upper()

    if m == "HA":
        return run_ha(sym_dir, device=device, force_rerun=force_rerun)
    if m == "QHA":
        # HA is needed as baseline even in QHA (provides the V₀ DOS fallback).
        ha_ok = run_ha(sym_dir, device=device, force_rerun=False)
        return run_qha(sym_dir, device=device, force_rerun=force_rerun) and ha_ok
    if m == "VDOS":
        # HA is always computed as a cheap baseline + needed for competing phases.
        return run_ha(sym_dir, device=device, force_rerun=False)

    raise ValueError(
        f"Unknown PHONON_MODE: {m!r}.  Supported: 'HA', 'QHA', 'VDOS'."
    )


def get_free_energy(
        sym_dir: str,
        temperatures: Sequence[float],
        mode: str | None = None,
        device: str = "cpu",
) -> dict | None:
    """Return F(T) for *sym_dir* using the configured (or specified) mode.

    This is the universal free-energy accessor used by
    :mod:`heatup.thermodynamic` and :mod:`heatup.free_energy`.

    Args:
        sym_dir:      Symmetry directory.
        temperatures: Temperature grid (K).
        mode:         Override ``config.PHONON_MODE``.
        device:       Compute device (for VDOS/AIMD triggering).

    Returns:
        Free-energy dict with keys ``E0_eV_per_atom``, ``temperatures``,
        ``F_eV_per_atom``, ``phonon_mode``, ``source_files``, or None on failure.
    """
    m = (mode or config.PHONON_MODE).upper()

    if m == "HA":
        return _ha_free_energy(sym_dir, list(temperatures))
    if m == "QHA":
        fe = _qha_free_energy(sym_dir, list(temperatures))
        return fe if fe is not None else _ha_free_energy(sym_dir, list(temperatures))
    if m == "VDOS":
        from heatup.anharmonic_phonons import get_anharmonic_free_energy
        fe = get_anharmonic_free_energy(
            sym_dir, list(temperatures), device=device, run_aimd_if_missing=False)
        if fe is not None:
            fe["phonon_mode"]  = "VDOS"
            fe["source_files"] = fe.get("anharmonic_sources", [])
        return fe

    raise ValueError(f"Unknown phonon mode: {m!r}")


# =============================================================================
# HA — Harmonic Approximation
# =============================================================================

def run_ha(
        sym_dir: str,
        device: str = "cpu",
        force_rerun: bool = False,
) -> bool:
    """Compute the harmonic phonon DOS and band structure.

    Reads ``relaxation/CONTCAR``, copies it to ``phonons/POSCAR``, and runs
    the finite-displacement phonon calculation using the backend set by
    ``config.PHONON_BACKEND``:

    - ``"ase"``     — :class:`ase.phonons.Phonons`  (2nd order only)
    - ``"phonopy"`` — phonopy Python API (2nd or 3rd order via phono3py)

    The force-constant order is set by ``config.FORCE_CONSTANT_ORDER``:

    - ``2`` — standard harmonic IFCs (both backends)
    - ``3`` — anharmonic IFCs via phono3py (phonopy backend only)

    Outputs::

        phonons/
            POSCAR                 ← copy of CONTCAR
            phonon-input.json      ← all parameters
            dos.json               ← {"energies_eV": [...], "weights": [...]}
            band_structure.json
            phonon.pdf
            dos_manifest.json

    Args:
        sym_dir:     Path to the symmetry directory.
        device:      Compute device for the calculator.
        force_rerun: Redo even if ``phonons/dos.json`` already exists.

    Returns:
        ``True`` on success.
    """
    ph_dir  = os.path.join(sym_dir, "phonons")
    dos_out = os.path.join(ph_dir, "dos.json")
    contcar = os.path.join(sym_dir, "relaxation", "CONTCAR")

    tag = _tag(sym_dir)
    if not force_rerun and os.path.exists(dos_out):
        print(f"[done] {tag}/phonons — dos.json exists (use force_rerun=True to redo).")
        return True
    if not os.path.exists(contcar):
        print(f"[skip] {tag}/phonons: relaxation/CONTCAR not found.")
        return False

    os.makedirs(ph_dir, exist_ok=True)
    dest = os.path.join(ph_dir, "POSCAR")
    shutil.copy2(contcar, dest)

    params = _ha_params(device)
    with open(os.path.join(ph_dir, "phonon-input.json"), "w") as fh:
        json.dump(params, fh, indent=4)

    backend = config.PHONON_BACKEND.lower()
    order   = config.FORCE_CONSTANT_ORDER

    print(f"HA phonons for {tag}  "
          f"(backend={backend}, IFC order={order}, "
          f"supercell={config.PHONON_SUPERCELL}, δ={config.PHONON_DELTA} Å) ...")

    try:
        if backend == "ase":
            if order != 2:
                raise ValueError(
                    "ASE Phonons only supports FORCE_CONSTANT_ORDER = 2.  "
                    "Set PHONON_BACKEND = 'phonopy' for 3rd-order IFCs."
                )
            ok = _run_ha_ase(sym_dir, dest, ph_dir, device)
        elif backend == "phonopy":
            ok = _run_ha_phonopy(sym_dir, dest, ph_dir, device, order)
        else:
            raise ValueError(f"Unknown PHONON_BACKEND: {backend!r}. "
                             "Supported: 'ase', 'phonopy'.")
    except Exception as exc:
        print(f"  [error] {tag}/phonons: {exc}")
        traceback.print_exc()
        return False

    if ok:
        write_manifest(dos_out, step="phonons_ha", extra=params)
        print(f"  HA phonons done → {dos_out}")
    return ok


def _run_ha_ase(sym_dir: str, poscar: str, ph_dir: str, device: str) -> bool:
    """Run HA phonons with ASE's built-in Phonons class (2nd-order only)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ase.io.vasp import read_vasp
    from ase.phonons import Phonons
    from heatup.calculator import build_calculator, release_calculator

    atoms = read_vasp(file=poscar)
    calc  = build_calculator(device=device)
    try:
        ph = Phonons(atoms, calc,
                     supercell = config.PHONON_SUPERCELL,
                     delta     = config.PHONON_DELTA,
                     name      = os.path.join(ph_dir, "phonon"))
        ph.run()
        ph.read(acoustic=True)

        path = atoms.cell.bandpath(npoints=config.PHONON_NPOINTS)
        bs   = ph.get_band_structure(path)
        with open(os.path.join(ph_dir, "band_structure.json"), "w") as fh:
            json.dump({"kpoints": path.kpts.tolist(),
                       "path": path.path,
                       "energies": bs.energies.tolist()}, fh, indent=2)

        dos = ph.get_dos().sample_grid(npts  = config.PHONON_DOS_NPTS,
                                       width = config.PHONON_DOS_WIDTH)
        with open(os.path.join(ph_dir, "dos.json"), "w") as fh:
            json.dump({"energies_eV": dos.get_energies().tolist(),
                       "weights":     dos.get_weights().tolist()}, fh, indent=2)

        _plot_phonon_ase(bs, dos, os.path.join(ph_dir, "phonon.pdf"))
        ph.clean()
        return True
    finally:
        if atoms is not None:
            atoms.calc = None
        release_calculator(calc)


def _run_ha_phonopy(
        sym_dir: str, poscar: str, ph_dir: str, device: str, order: int,
) -> bool:
    """Run HA phonons with phonopy (supports 2nd and 3rd-order via phono3py)."""
    try:
        import phonopy
    except ImportError as exc:
        raise ImportError(
            "phonopy is required for PHONON_BACKEND = 'phonopy'. "
            "Install with:  pip install phonopy"
        ) from exc

    from ase.io.vasp import read_vasp, write_vasp
    from heatup.calculator import build_calculator, release_calculator
    import numpy as _np

    if order == 3:
        return _run_ha_phono3py(sym_dir, poscar, ph_dir, device)

    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    atoms = read_vasp(file=poscar)
    cell  = PhonopyAtoms(
        symbols           = atoms.get_chemical_symbols(),
        scaled_positions  = atoms.get_scaled_positions(),
        cell              = atoms.get_cell(),
    )
    sc    = list(config.PHONON_SUPERCELL)
    ph    = Phonopy(cell, supercell_matrix=_np.diag(sc))
    ph.generate_displacements(distance=config.PHONON_DELTA)
    ph.save(os.path.join(ph_dir, "phonopy_disp.yaml"))

    calc = build_calculator(device=device)
    try:
        force_sets = []
        for sc_atoms in ph.supercells_with_displacements:
            from phonopy.interface.vasp import get_vasp_structure
            from ase import Atoms as AseAtoms
            ase_sc = AseAtoms(
                symbols          = sc_atoms.get_chemical_symbols(),
                scaled_positions = sc_atoms.get_scaled_positions(),
                cell             = sc_atoms.get_cell(),
                pbc              = True,
            )
            ase_sc.calc = calc
            forces = ase_sc.get_forces()
            ase_sc.calc = None
            force_sets.append(forces)
    finally:
        release_calculator(calc)

    ph.forces = force_sets
    ph.produce_force_constants()

    # DOS
    ph.run_mesh(list(config.PHONON_DOS_KPTS))
    ph.run_total_dos(freq_pitch=config.PHONON_DOS_WIDTH * 33.356)  # meV → THz approx
    dos_obj = ph.get_total_dos_dict()
    # phonopy gives THz; convert to eV:  1 THz = 4.135667696e-3 eV
    freq_eV = _np.array(dos_obj["frequency_points"]) * 4.135667696e-3
    weights  = _np.array(dos_obj["total_dos"])
    with open(os.path.join(ph_dir, "dos.json"), "w") as fh:
        json.dump({"energies_eV": freq_eV.tolist(),
                   "weights":     weights.tolist()}, fh, indent=2)

    # Band structure approximation via mesh (no special path in phonopy without seekpath).
    with open(os.path.join(ph_dir, "band_structure.json"), "w") as fh:
        json.dump({"note": "phonopy backend — use phonopy CLI for full band structure",
                   "energies": [[]], "kpoints": [], "path": ""}, fh, indent=2)

    return True


def _run_ha_phono3py(sym_dir: str, poscar: str, ph_dir: str, device: str) -> bool:
    """Run 3rd-order IFCs via phono3py."""
    try:
        import phono3py
    except ImportError as exc:
        raise ImportError(
            "phono3py is required for FORCE_CONSTANT_ORDER = 3. "
            "Install with:  pip install phono3py"
        ) from exc

    from ase.io.vasp import read_vasp
    from phono3py import Phono3py
    from phonopy.structure.atoms import PhonopyAtoms
    from heatup.calculator import build_calculator, release_calculator
    from ase import Atoms as AseAtoms
    import numpy as _np

    atoms = read_vasp(file=poscar)
    cell  = PhonopyAtoms(
        symbols           = atoms.get_chemical_symbols(),
        scaled_positions  = atoms.get_scaled_positions(),
        cell              = atoms.get_cell(),
    )
    sc = list(config.PHONON_SUPERCELL)
    ph3 = Phono3py(cell, supercell_matrix=_np.diag(sc))
    ph3.generate_displacements(distance=config.PHONON_DELTA)

    calc = build_calculator(device=device)
    try:
        force_sets = []
        for sc_atoms in ph3.supercells_with_displacements:
            if sc_atoms is None:
                force_sets.append(None)
                continue
            ase_sc = AseAtoms(
                symbols          = sc_atoms.get_chemical_symbols(),
                scaled_positions = sc_atoms.get_scaled_positions(),
                cell             = sc_atoms.get_cell(),
                pbc              = True,
            )
            ase_sc.calc = calc
            force_sets.append(ase_sc.get_forces())
            ase_sc.calc = None
    finally:
        release_calculator(calc)

    ph3.forces = force_sets
    ph3.produce_fc3()
    ph3.produce_fc2()

    # Extract 2nd-order DOS via the embedded Phonopy object.
    ph3.run_mesh(list(config.PHONON_DOS_KPTS))
    ph3.run_total_dos()
    dos_obj = ph3.get_total_dos_dict()
    freq_eV = _np.array(dos_obj["frequency_points"]) * 4.135667696e-3
    weights  = _np.array(dos_obj["total_dos"])
    with open(os.path.join(ph_dir, "dos.json"), "w") as fh:
        json.dump({"energies_eV": freq_eV.tolist(),
                   "weights":     weights.tolist(),
                   "note": "from 3rd-order IFCs via phono3py"}, fh, indent=2)

    # Save fc3 for downstream lattice-thermal-conductivity calculations.
    ph3.save(os.path.join(ph_dir, "phono3py_params.hdf5"))
    print(f"  3rd-order IFCs saved → {ph_dir}/phono3py_params.hdf5")
    return True


# =============================================================================
# QHA — Quasi-Harmonic Approximation
# =============================================================================

def run_qha(
        sym_dir: str,
        device: str = "cpu",
        force_rerun: bool = False,
) -> bool:
    """Compute the quasi-harmonic free energy F(T, P=0).

    Samples ``config.QHA_N_VOLUMES`` volumes around V₀ in the range
    ``V₀ × (1 ± QHA_VOLUME_RANGE)``, computes the phonon DOS at each volume
    using ``config.PHONON_BACKEND``, and fits the E(V, T) surface with the
    EOS model ``config.QHA_EOS`` to obtain:

    - ``F_eq(T)`` — free energy at the equilibrium volume V(T).
    - ``V_eq(T)`` — equilibrium volume vs temperature (thermal expansion).
    - ``alpha(T)`` — volumetric thermal expansion coefficient.
    - ``Gruneisen`` — Grüneisen parameter (average).

    Requires phonopy.  Outputs are written to ``phonons/qha/``.

    Args:
        sym_dir:     Path to the symmetry directory.
        device:      Compute device.
        force_rerun: Redo even if ``phonons/qha/qha_result.json`` exists.

    Returns:
        ``True`` on success.
    """
    try:
        from phonopy.qha import PhonopyQHA
    except ImportError as exc:
        raise ImportError(
            "phonopy is required for QHA.  Install with:  pip install phonopy"
        ) from exc

    from ase.io.vasp import read_vasp
    from heatup.calculator import build_calculator, release_calculator
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    from ase import Atoms as AseAtoms
    import numpy as _np

    tag     = _tag(sym_dir)
    qha_dir = os.path.join(sym_dir, "phonons", "qha")
    result_path = os.path.join(qha_dir, "qha_result.json")

    if not force_rerun and os.path.exists(result_path):
        print(f"[done] {tag}/phonons/qha — qha_result.json exists.")
        return True

    contcar = os.path.join(sym_dir, "relaxation", "CONTCAR")
    if not os.path.exists(contcar):
        print(f"[skip] {tag}/phonons/qha: relaxation/CONTCAR not found.")
        return False

    os.makedirs(qha_dir, exist_ok=True)

    n_vol  = config.QHA_N_VOLUMES
    v_range = config.QHA_VOLUME_RANGE

    if n_vol < 3 or n_vol % 2 == 0:
        raise ValueError(
            f"QHA_N_VOLUMES must be an odd integer ≥ 3 (got {n_vol}).  "
            f"Typical choices: 5, 7, 9."
        )

    atoms_ref = read_vasp(file=contcar)
    V0        = atoms_ref.get_volume()
    # Symmetric set of volume scale factors around 1.0
    scales    = _np.linspace(1.0 - v_range, 1.0 + v_range, n_vol)
    volumes   = V0 * scales

    params = _qha_params(device)
    with open(os.path.join(qha_dir, "qha-input.json"), "w") as fh:
        json.dump(params, fh, indent=4)

    print(f"QHA for {tag}: {n_vol} volumes "
          f"[{volumes.min():.2f} … {volumes.max():.2f} Å³], "
          f"EOS={config.QHA_EOS}, backend={config.PHONON_BACKEND} ...")

    calc = build_calculator(device=device)
    energies_eV:    list[float]       = []
    dos_freq_list:  list[np.ndarray]  = []
    dos_weight_list: list[np.ndarray] = []

    try:
        for i, (scale, vol) in enumerate(zip(scales, volumes)):
            v_dir = os.path.join(qha_dir, f"volume_{i:02d}")
            os.makedirs(v_dir, exist_ok=True)

            # Scale cell at constant fractional coordinates.
            sc_atoms = atoms_ref.copy()
            sc_atoms.set_cell(atoms_ref.get_cell() * (scale ** (1/3)),
                              scale_atoms=True)

            from ase.io.vasp import write_vasp
            write_vasp(os.path.join(v_dir, "POSCAR"), sc_atoms, direct=True)

            # Energy at this volume.
            sc_atoms.calc = calc
            e = float(sc_atoms.get_potential_energy()) / len(sc_atoms)
            sc_atoms.calc = None
            energies_eV.append(e)

            # Phonon DOS at this volume.
            phcell = PhonopyAtoms(
                symbols           = sc_atoms.get_chemical_symbols(),
                scaled_positions  = sc_atoms.get_scaled_positions(),
                cell              = sc_atoms.get_cell(),
            )
            sc_mat = _np.diag(list(config.PHONON_SUPERCELL))
            ph     = Phonopy(phcell, supercell_matrix=sc_mat)
            ph.generate_displacements(distance=config.PHONON_DELTA)

            force_sets_v: list[np.ndarray] = []
            for disp_sc in ph.supercells_with_displacements:
                ase_d = AseAtoms(
                    symbols          = disp_sc.get_chemical_symbols(),
                    scaled_positions = disp_sc.get_scaled_positions(),
                    cell             = disp_sc.get_cell(),
                    pbc              = True,
                )
                ase_d.calc = calc
                force_sets_v.append(ase_d.get_forces())
                ase_d.calc = None

            ph.forces = force_sets_v
            ph.produce_force_constants()
            ph.run_mesh(list(config.PHONON_DOS_KPTS))
            ph.run_total_dos(freq_pitch=config.PHONON_DOS_WIDTH * 33.356)
            dos_d = ph.get_total_dos_dict()

            freq_thz = _np.array(dos_d["frequency_points"])
            weights  = _np.array(dos_d["total_dos"])
            dos_freq_list.append(freq_thz)
            dos_weight_list.append(weights)

            # Save per-volume DOS for auditability.
            freq_eV = freq_thz * 4.135667696e-3
            with open(os.path.join(v_dir, "dos.json"), "w") as fh:
                json.dump({"energies_eV": freq_eV.tolist(),
                           "weights":     weights.tolist(),
                           "volume_A3":   float(vol),
                           "scale":       float(scale)}, fh, indent=2)
            print(f"  vol {i+1}/{n_vol}  scale={scale:.4f}  "
                  f"V={vol:.2f} Å³  E={e:.6f} eV/atom")
    finally:
        release_calculator(calc)

    # ── QHA fit via phonopy ───────────────────────────────────────────────
    T_arr = _np.array(sorted(set(config.HULL_TEMPERATURES)), dtype=float)
    T_arr = T_arr[T_arr >= 0]

    # PhonopyQHA needs equal-length frequency/dos arrays.
    # Use the frequency grid of the first volume (interpolate others).
    freq_ref = dos_freq_list[0]
    dos_interp = []
    for fq, wt in zip(dos_freq_list, dos_weight_list):
        dos_interp.append(_np.interp(freq_ref, fq, wt, left=0, right=0))

    try:
        qha = PhonopyQHA(
            volumes          = _np.array(volumes) / len(atoms_ref),  # per atom
            electronic_energies = _np.array(energies_eV),
            temperatures     = T_arr,
            free_energy      = None,   # computed internally
            cv               = None,
            entropy          = None,
            dos_frequency    = freq_ref,
            dos              = _np.array(dos_interp),
            eos              = config.QHA_EOS,
        )

        F_eq  = qha.get_gibbs_temperature()            # eV/atom vs T
        V_eq  = qha.get_volume_temperature()           # Å³/atom vs T
        alpha = qha.get_thermal_expansion()            # 1/K vs T
        bulk  = qha.get_bulk_modulus_temperature()     # GPa vs T
        grun  = float(_np.mean(alpha)) / (float(_np.mean(bulk)) * 1e9
                                           / 160.21766208)   # approximate Grüneisen

    except Exception as exc:
        print(f"  [error] QHA fit failed: {exc}  — falling back to HA.")
        traceback.print_exc()
        return False

    result = {
        "phonon_mode"    : "QHA",
        "E0_eV_per_atom" : float(energies_eV[n_vol // 2]),
        "temperatures"   : T_arr.tolist(),
        "F_eV_per_atom"  : F_eq.tolist(),
        "V_eq_A3_atom"   : V_eq.tolist(),
        "alpha_1_K"      : alpha.tolist(),
        "bulk_GPa"       : bulk.tolist(),
        "gruneisen_mean" : float(grun),
        "volumes_A3"     : (volumes / len(atoms_ref)).tolist(),
        "energies_eV_atom": energies_eV,
        "eos"            : config.QHA_EOS,
        "n_volumes"      : n_vol,
        "volume_range"   : v_range,
    }
    with open(result_path, "w") as fh:
        json.dump(result, fh, indent=4)

    write_manifest(result_path, step="phonons_qha", extra=params)
    print(f"  QHA done → {result_path}")
    return True


# =============================================================================
# VDOS — Anharmonic phonons from AIMD
# =============================================================================

def run_vdos_for_sim(
        sim_dir: str,
        temperatures: Sequence[float],
        force_recompute: bool = False,
) -> dict | None:
    """Compute VDOS and F(T) from an AIMD trajectory.

    This is a thin, well-documented wrapper around
    :func:`heatup.anharmonic_phonons.compute_anharmonic_phonons_for_sim`
    that writes the manifest alongside the VDOS cache.

    Args:
        sim_dir:         Path to the temperature sub-folder
                         (``aimd/<T>K``).
        temperatures:    Temperature grid for F(T) (K).
        force_recompute: Ignore cache and recompute.

    Returns:
        Free-energy dict ``{E0_eV_per_atom, temperatures, F_eV_per_atom,
        phonon_mode, source_files}``, or None on failure.
    """
    from heatup.anharmonic_phonons import compute_anharmonic_phonons_for_sim, _anharmonic_dir

    fe = compute_anharmonic_phonons_for_sim(
        sim_dir, temperatures, force_recompute=force_recompute)

    if fe is not None:
        vdos_path = os.path.join(_anharmonic_dir(sim_dir), "vdos.json")
        write_manifest(vdos_path, step="phonons_vdos",
                       extra={"sim_dir": sim_dir,
                              "n_temperatures": len(temperatures)})
        fe["phonon_mode"]  = "VDOS"
        fe["source_files"] = [sim_dir]
    return fe


# =============================================================================
# Free-energy helpers  (read cached output files)
# =============================================================================

def _ha_free_energy(sym_dir: str, temperatures: list[float]) -> dict | None:
    """Load cached HA F(T) from ``phonons/dos.json``."""
    e0 = _load_e0(sym_dir)
    if e0 is None:
        return None
    dos = _load_ha_dos(sym_dir)
    if dos is None:
        return None
    en, wt = dos
    Fs = [e0 + _ha_f_vib(en, wt, T) for T in temperatures]
    return {
        "E0_eV_per_atom": e0,
        "temperatures"  : temperatures,
        "F_eV_per_atom" : Fs,
        "phonon_mode"   : "HA",
        "source_files"  : [os.path.join(sym_dir, "phonons", "dos.json")],
    }


def _qha_free_energy(sym_dir: str, temperatures: list[float]) -> dict | None:
    """Load cached QHA F(T) from ``phonons/qha/qha_result.json``."""
    result_path = os.path.join(sym_dir, "phonons", "qha", "qha_result.json")
    if not os.path.exists(result_path):
        return None
    with open(result_path) as fh:
        r = json.load(fh)

    T_cached = _np.array(r["temperatures"])
    F_cached = _np.array(r["F_eV_per_atom"])
    F_interp = [float(_np.interp(T, T_cached, F_cached)) for T in temperatures]
    return {
        "E0_eV_per_atom": float(r["E0_eV_per_atom"]),
        "temperatures"  : temperatures,
        "F_eV_per_atom" : F_interp,
        "phonon_mode"   : "QHA",
        "qha_extras"    : {k: r[k] for k in ("V_eq_A3_atom","alpha_1_K","gruneisen_mean")
                           if k in r},
        "source_files"  : [result_path],
    }


def _load_e0(sym_dir: str) -> float | None:
    path = os.path.join(sym_dir, "relaxation", "energy.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return float(json.load(fh)["energy_eV_per_atom"])


def _load_ha_dos(sym_dir: str) -> tuple[np.ndarray, np.ndarray] | None:
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


def _ha_f_vib(en: np.ndarray, wt: np.ndarray, T: float) -> float:
    """Harmonic F_vib per atom (eV) via QHO integral."""
    if T <= 0.0:
        return float(np.trapezoid(0.5 * wt * en, en))
    kT = config.KB_EV * T
    x  = en / (2.0 * kT)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        integrand = wt * np.log(2.0 * np.sinh(np.clip(x, 1e-12, 500.0)))
    return float(kT * np.trapezoid(integrand, en))


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_phonon_ase(bs, dos, out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 5))
    ax_bs  = fig.add_axes([0.08, 0.10, 0.60, 0.82])
    ax_dos = fig.add_axes([0.70, 0.10, 0.22, 0.82])

    emax = max(float(np.max(bs.energies)) * 1.05, 1e-6)
    emin = min(float(np.min(bs.energies)) * 1.05, -0.1)
    bs.plot(ax=ax_bs, emin=0.0, emax=emax)

    ax_dos.fill_between(dos.get_weights(), dos.get_energies(), y2=0,
                        color="steelblue", edgecolor="k", alpha=0.75)
    ax_dos.set_ylim(emin, emax)
    ax_dos.set_yticks([]); ax_dos.set_xticks([])
    ax_dos.set_xlabel("DOS")
    fig.tight_layout(pad=0.5)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=80, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Parameter record builders (written to *-input.json before computation)
# ---------------------------------------------------------------------------

def _ha_params(device: str) -> dict:
    from heatup.calculator import calculator_label
    return {
        "phonon_mode"          : "HA",
        "force_constant_order" : config.FORCE_CONSTANT_ORDER,
        "phonon_backend"       : config.PHONON_BACKEND,
        "calculator"           : calculator_label(),
        "device"               : device,
        "supercell"            : list(config.PHONON_SUPERCELL),
        "delta_A"              : config.PHONON_DELTA,
        "dos_kpts"             : list(config.PHONON_DOS_KPTS),
        "dos_npts"             : config.PHONON_DOS_NPTS,
        "dos_width_eV"         : config.PHONON_DOS_WIDTH,
    }


def _qha_params(device: str) -> dict:
    from heatup.calculator import calculator_label
    return {
        "phonon_mode"   : "QHA",
        "phonon_backend": config.PHONON_BACKEND,
        "calculator"    : calculator_label(),
        "device"        : device,
        "supercell"     : list(config.PHONON_SUPERCELL),
        "delta_A"       : config.PHONON_DELTA,
        "n_volumes"     : config.QHA_N_VOLUMES,
        "volume_range"  : config.QHA_VOLUME_RANGE,
        "eos"           : config.QHA_EOS,
        "dos_kpts"      : list(config.PHONON_DOS_KPTS),
    }


# ---------------------------------------------------------------------------
# Private utility
# ---------------------------------------------------------------------------

def _tag(sym_dir: str) -> str:
    symmetry = os.path.basename(os.path.abspath(sym_dir))
    material = os.path.basename(os.path.dirname(os.path.abspath(sym_dir)))
    return f"{material}/{symmetry}"


_np = np  # alias to avoid shadowing in inner functions
