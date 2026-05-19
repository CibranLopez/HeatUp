"""heatup.structure_pipeline
==============================
Pre-MD structure preparation and property calculations.

This module handles the four workflows that operate on each
``database/<material>/<symmetry>/`` directory **before** the AIMD simulations:

1. **Relaxation** — geometry-optimise the reference structure with the
   configured calculator (see :mod:`heatup.calculator`).
2. **Phonons** — finite-displacement harmonic phonon calculation on the
   relaxed cell.  The DOS is used as the ``harmonic`` free-energy source.
3. **Elastic tensor** — full 6×6 Voigt stiffness tensor via central-difference
   stress–strain with the configured calculator.
4. **AIMD supercell preparation** — build a supercell where every lattice
   parameter is at least :data:`heatup.config.AIMD_MIN_CELL_ANG` Å and
   distribute it to the temperature sub-folders.

Expected directory layout produced::

    database/
      <material>/
        <symmetry>/
          POSCAR              ← reference structure (never modified)
          space_group         ← ITA number, written once at relaxation
          relaxation/
            POSCAR                ← copy of POSCAR used for this run
            relaxation-input.json ← all parameters, written before computation
            run.traj              ← BFGS optimisation trajectory
            CONTCAR               ← relaxed structure (VASP format)
            energy.json           ← {"energy_eV_per_atom": float, ...}
          phonons/
            POSCAR                ← copy of relaxation/CONTCAR
            phonon-input.json     ← all phonon parameters
            dos.json              ← {"energies_eV": [...], "weights": [...]}
            band_structure.json   ← serialised phonon band structure
            phonon.pdf            ← band structure + DOS figure
          elastic/
            POSCAR                ← copy of relaxation/CONTCAR
            elastic-input.json    ← all elastic parameters
            elastic_tensor.json   ← 6×6 Voigt tensor (GPa) + derived moduli
          aimd/
            POSCAR                ← master supercell (upper-triangular cell)
            POSCAR-unitcell       ← relaxed unit cell for reference
            aimd-supercell.json   ← repetitions, sizes, atom counts
            <T>K/
              POSCAR              ← copy of aimd/POSCAR for this temperature

All fixed calculation parameters are read from :mod:`heatup.config`; no
``import``-time constants are redefined here.  All parameters are serialised
into ``*-input.json`` files *before* any computation.

GPU memory isolation
--------------------
CUDA memory is **not** reliably freed by ``del`` / ``gc`` / ``empty_cache()``
when the same Python process runs multiple materials sequentially.  The
:func:`run_relaxation_subprocess`, :func:`run_phonons_subprocess`, and
:func:`run_elastic_subprocess` wrappers spawn isolated child processes so the
OS releases the CUDA context unconditionally on exit.  These wrappers are
recommended for batch runs.  The bare :func:`run_relaxation` etc. functions
are available for single-shot use or CPU runs.

Typical usage::

    from heatup.structure_pipeline import (
        run_relaxation_subprocess, run_phonons_subprocess,
        run_elastic_subprocess, prepare_aimd_folders,
    )

    sym_dir = "database/AgI/P6_3mc"
    run_relaxation_subprocess(sym_dir, device="cuda")
    run_phonons_subprocess   (sym_dir, device="cuda")
    run_elastic_subprocess   (sym_dir, device="cuda")
    prepare_aimd_folders(sym_dir, temperatures=[600, 900])
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ase.io.vasp import read_vasp, write_vasp
from ase.optimize import BFGS
from ase.filters import ExpCellFilter
from ase.phonons import Phonons

from heatup import config
from heatup.calculator import build_calculator, release_calculator, calculator_label


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _tag(sym_dir: str) -> str:
    """Return ``material/symmetry`` tag string for log messages."""
    symmetry = os.path.basename(os.path.abspath(sym_dir))
    material = os.path.basename(os.path.dirname(os.path.abspath(sym_dir)))
    return f"{material}/{symmetry}"


def _cuda_env() -> dict[str, str]:
    """Return a copy of the environment with CUDA allocator options set."""
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    return env


def _make_cell_upper_triangular(atoms):
    """Return a copy of *atoms* with an upper-triangular simulation cell.

    ASE's NPT integrator requires ``h[1,0] == h[2,0] == h[2,1] == 0``.

    The unique upper-triangular cell with the same metric tensor is
    constructed analytically from ``G = A @ A.T`` (Gram matrix)::

        f = √G₂₂
        e = G₁₂ / f
        d = √(G₁₁ − e²)
        c = G₀₂ / f
        b = (G₀₁ − c·e) / d
        a = √(G₀₀ − b² − c²)
        h_new = [[a, b, c],
                 [0, d, e],
                 [0, 0, f]]

    All inter-atomic distances, cell lengths, and angles are exactly
    preserved.  The transform is idempotent on already upper-triangular cells.
    Atomic positions are updated via ``scale_atoms=True`` (fractional
    coordinates fixed, Cartesian positions recomputed as ``frac @ h_new``).

    Args:
        atoms: ASE :class:`~ase.Atoms` object (not modified in-place).

    Returns:
        New :class:`~ase.Atoms` with an upper-triangular cell.
    """
    A = atoms.cell.array.copy()
    G = A @ A.T   # metric tensor

    f = np.sqrt(G[2, 2])
    e = G[1, 2] / f
    d = np.sqrt(np.maximum(G[1, 1] - e**2, 0.0))
    c = G[0, 2] / f
    b = (G[0, 1] - c * e) / (d if d > 0 else 1.0)
    a = np.sqrt(np.maximum(G[0, 0] - b**2 - c**2, 0.0))

    new = atoms.copy()
    new.set_cell(np.array([[a, b, c], [0, d, e], [0, 0, f]]), scale_atoms=True)
    return new


def _compute_supercell_repetitions(
        cell: np.ndarray,
        min_length: float,
) -> tuple[int, int, int]:
    """Compute the smallest supercell that satisfies the minimum-length constraint.

    Args:
        cell:       3×3 cell matrix in Å (rows are lattice vectors).
        min_length: Minimum required length of each lattice parameter in Å.

    Returns:
        Tuple ``(na, nb, nc)`` of integer repetitions along each axis.
    """
    lengths = np.linalg.norm(cell, axis=1)
    reps    = np.maximum(np.ceil(min_length / lengths).astype(int), 1)
    return int(reps[0]), int(reps[1]), int(reps[2])


def _write_space_group(sym_dir: str, atoms) -> int | None:
    """Write ITA space group number to ``<sym_dir>/space_group``.

    Args:
        sym_dir: Symmetry directory path.
        atoms:   ASE Atoms object for the reference structure.

    Returns:
        Space group number, or None if pymatgen is unavailable.
    """
    try:
        from pymatgen.io.ase import AseAtomsAdaptor
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        structure = AseAtomsAdaptor.get_structure(atoms)
        n = SpacegroupAnalyzer(structure).get_space_group_number()
        with open(os.path.join(sym_dir, "space_group"), "w") as fh:
            fh.write(f"{n}\n")
        return n
    except Exception as exc:
        print(f"  [warn] Could not write space_group file: {exc}")
        return None


def _relaxation_params(device: str) -> dict:
    return {
        "calculator"      : calculator_label(),
        "device"          : device,
        "fmax"            : config.RELAX_FMAX,
        "max_steps"       : config.RELAX_MAX_STEPS,
        "relax_cell"      : config.RELAX_CELL,
        "constant_volume" : config.RELAX_CONSTANT_VOLUME,
    }


def _phonon_params(device: str) -> dict:
    return {
        "calculator"  : calculator_label(),
        "device"      : device,
        "supercell"   : list(config.PHONON_SUPERCELL),
        "delta"       : config.PHONON_DELTA,
        "npoints"     : config.PHONON_NPOINTS,
        "dos_kpts"    : list(config.PHONON_DOS_KPTS),
        "dos_npts"    : config.PHONON_DOS_NPTS,
        "dos_width"   : config.PHONON_DOS_WIDTH,
    }


def _elastic_params(device: str) -> dict:
    return {
        "calculator" : calculator_label(),
        "device"     : device,
        "delta"      : config.ELASTIC_DELTA,
    }


def _plot_phonon(bs, dos, out_path: str) -> None:
    """Save a combined phonon band structure + DOS figure.

    Args:
        bs:       ASE :class:`~ase.spectrum.band_structure.BandStructure`.
        dos:      ASE DOS object returned by ``Phonons.get_dos().sample_grid()``.
        out_path: Full output path for the PDF figure.
    """
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
# Public calculation functions
# ---------------------------------------------------------------------------

def run_relaxation(
        sym_dir: str,
        device: str = "cpu",
        force_rerun: bool = False,
) -> bool:
    """Geometry-optimise the reference structure.

    Reads ``POSCAR`` from *sym_dir*, copies it to ``relaxation/POSCAR``, runs
    a BFGS optimisation using the calculator configured in
    :mod:`heatup.config`, and writes the relaxed structure as
    ``relaxation/CONTCAR``.

    The MACE ground-state energy is stored in ``relaxation/energy.json`` for
    use by the thermodynamic hull without re-running the calculator.

    All parameters (calculator, fmax, relax_cell …) are read from
    :mod:`heatup.config` and recorded in ``relaxation/relaxation-input.json``
    before any computation begins.

    Args:
        sym_dir:     Path to the symmetry directory, e.g.
                     ``"database/AgI/P6_3mc"``.
        device:      Compute device for the calculator (``"cpu"`` or ``"cuda"``).
        force_rerun: Redo even if ``relaxation/CONTCAR`` already exists.

    Returns:
        ``True`` on success, ``False`` if ``POSCAR`` is missing or the
        optimisation fails.
    """
    tag     = _tag(sym_dir)
    rel_dir = os.path.join(sym_dir, "relaxation")
    contcar = os.path.join(rel_dir, "CONTCAR")

    if not force_rerun and os.path.exists(contcar):
        print(f"[done] {tag}/relaxation — CONTCAR exists (use force_rerun=True to redo).")
        return True

    poscar = os.path.join(sym_dir, "POSCAR")
    if not os.path.exists(poscar):
        print(f"[skip] {tag}: POSCAR not found.")
        return False

    os.makedirs(rel_dir, exist_ok=True)
    dest = os.path.join(rel_dir, "POSCAR")
    shutil.copy2(poscar, dest)

    with open(os.path.join(rel_dir, "relaxation-input.json"), "w") as fh:
        json.dump(_relaxation_params(device), fh, indent=4)

    calc  = None
    atoms = None
    try:
        atoms = read_vasp(file=dest)
        # Write space group from the unrelaxed reference structure.
        spg = _write_space_group(sym_dir, atoms)
        if spg:
            print(f"  Space group: {spg}")

        calc       = build_calculator(device=device)
        atoms.calc = calc

        filter_atoms = (
            ExpCellFilter(atoms, constant_volume=config.RELAX_CONSTANT_VOLUME)
            if config.RELAX_CELL else atoms
        )
        traj_path = os.path.join(rel_dir, "run.traj")
        dyn = BFGS(filter_atoms, trajectory=traj_path, logfile=None)

        print(f"Relaxing {tag}  "
              f"(fmax={config.RELAX_FMAX} eV/Å, "
              f"relax_cell={config.RELAX_CELL}, "
              f"backend={calculator_label()}) ...")
        dyn.run(fmax=config.RELAX_FMAX, steps=config.RELAX_MAX_STEPS)

        relaxed = filter_atoms.atoms if config.RELAX_CELL else filter_atoms
        write_vasp(contcar, relaxed, direct=True, sort=True)

        e0_total    = float(relaxed.get_potential_energy())
        n_atoms     = len(relaxed)
        e0_per_atom = e0_total / n_atoms
        with open(os.path.join(rel_dir, "energy.json"), "w") as fh:
            json.dump({
                "energy_eV"          : e0_total,
                "energy_eV_per_atom" : e0_per_atom,
                "n_atoms"            : n_atoms,
                "calculator"         : calculator_label(),
            }, fh, indent=4)

        print(f"  Relaxation done.  E0 = {e0_per_atom:.6f} eV/atom → {contcar}")
        return True

    except Exception as exc:
        print(f"  [error] {tag}/relaxation: {exc}")
        traceback.print_exc()
        return False

    finally:
        if atoms is not None:
            atoms.calc = None
        release_calculator(calc)


def run_phonons(
        sym_dir: str,
        device: str = "cpu",
        force_rerun: bool = False,
        mode: str | None = None,
) -> bool:
    """Compute phonons for *sym_dir* using the configured mode.

    This function is a thin dispatch layer that delegates to
    :mod:`heatup.phonons`, which implements all three modes.

    The active mode is determined by ``config.PHONON_MODE`` (or the *mode*
    argument, which overrides it for this single call).

    Mode behaviour
    --------------
    ``"HA"``
        Finite-displacement harmonic phonons at V₀.  Always produces
        ``phonons/dos.json``, which is the fallback free-energy source for
        competing phases in the thermodynamic hull.  Uses
        ``config.PHONON_BACKEND`` (``"ase"`` or ``"phonopy"``) and
        ``config.FORCE_CONSTANT_ORDER`` (2 or 3, phonopy only).

    ``"QHA"``
        Quasi-harmonic phonons at ``config.QHA_N_VOLUMES`` volumes.
        Always also runs HA first (writes ``phonons/dos.json``) as a baseline
        and to ensure competing-phase data is available.  Requires phonopy.
        Results in ``phonons/qha/qha_result.json``.

    ``"VDOS"``
        No phonon displacement calculation is needed — the VDOS is computed
        from the AIMD trajectory by :func:`heatup.phonons.run_vdos_for_sim`.
        This function **always also runs HA** in VDOS mode to produce
        ``phonons/dos.json`` (used by competing phases and as a cross-check).

    All modes write a ``_manifest.json`` next to their output recording the
    full configuration (calculator, supercell, δ, IFC order, backend …).

    Args:
        sym_dir:     Path to the symmetry directory
                     (e.g. ``"database/AgI/P6_3mc"``).
        device:      Compute device for the calculator.
        force_rerun: Redo even if output files already exist.
        mode:        Override ``config.PHONON_MODE`` for this call only.
                     One of ``"HA"``, ``"QHA"``, ``"VDOS"``.

    Returns:
        ``True`` on success (or if already cached), ``False`` on failure.
    """
    from heatup.phonons import run_phonons as _phonons_run
    return _phonons_run(sym_dir, device=device, force_rerun=force_rerun, mode=mode)


# ---------------------------------------------------------------------------
# Legacy inline implementation kept as _run_phonons_ase() for reference.
# The public run_phonons() now dispatches through heatup.phonons.
# ---------------------------------------------------------------------------

def _run_phonons_ase_legacy(
        sym_dir: str,
        device: str = "cpu",
        force_rerun: bool = False,
) -> bool:
    """[Internal] Original ASE-based HA phonon implementation.

    Kept for reference and direct testing.  For production use, call
    :func:`run_phonons` which dispatches through :mod:`heatup.phonons`.

    Args:
        sym_dir:     Path to the symmetry directory.
        device:      Compute device.
        force_rerun: Redo even if cached.

    Returns:
        ``True`` on success.
    """
    tag     = _tag(sym_dir)
    ph_dir  = os.path.join(sym_dir, "phonons")
    dos_out = os.path.join(ph_dir, "dos.json")
    contcar = os.path.join(sym_dir, "relaxation", "CONTCAR")

    if not force_rerun and os.path.exists(dos_out):
        print(f"[done] {tag}/phonons — dos.json exists.")
        return True

    if not os.path.exists(contcar):
        print(f"[skip] {tag}/phonons: relaxation/CONTCAR not found.")
        return False

    os.makedirs(ph_dir, exist_ok=True)
    dest = os.path.join(ph_dir, "POSCAR")
    shutil.copy2(contcar, dest)

    calc  = None
    atoms = None
    try:
        atoms = read_vasp(file=dest)
        calc  = build_calculator(device=device)

        print(f"Computing phonons for {tag}  "
              f"(supercell={config.PHONON_SUPERCELL}, "
              f"delta={config.PHONON_DELTA} Å, "
              f"backend={calculator_label()}) ...")

        ph = Phonons(atoms, calc,
                     supercell=config.PHONON_SUPERCELL,
                     delta=config.PHONON_DELTA,
                     name=os.path.join(ph_dir, "phonon"))
        ph.run()
        ph.read(acoustic=True)

        path = atoms.cell.bandpath(npoints=config.PHONON_NPOINTS)
        bs   = ph.get_band_structure(path)
        with open(os.path.join(ph_dir, "band_structure.json"), "w") as fh:
            json.dump({"kpoints": path.kpts.tolist(),
                       "path": path.path,
                       "energies": bs.energies.tolist()}, fh, indent=2)

        dos = ph.get_dos().sample_grid(npts=config.PHONON_DOS_NPTS,
                                       width=config.PHONON_DOS_WIDTH)
        with open(dos_out, "w") as fh:
            json.dump({"energies_eV": dos.get_energies().tolist(),
                       "weights":     dos.get_weights().tolist()}, fh, indent=2)

        _plot_phonon(bs, dos, os.path.join(ph_dir, "phonon.pdf"))
        ph.clean()
        print(f"  Phonons done → {dos_out}")
        return True

    except Exception as exc:
        print(f"  [error] {tag}/phonons: {exc}")
        traceback.print_exc()
        return False

    finally:
        if atoms is not None:
            atoms.calc = None
        release_calculator(calc)


def _voigt_strain_matrix(voigt_index: int, delta: float) -> np.ndarray:
    """Return the 3×3 symmetric strain tensor for one Voigt component.

    Maps Voigt index (0–5) → (xx, yy, zz, yz, xz, xy) strain.  The
    off-diagonal entries use δ/2 so that the symmetric engineering shear
    strain γ_ij = 2·ε_ij = δ.

    Args:
        voigt_index: Integer 0–5.
        delta:       Strain magnitude (dimensionless).

    Returns:
        3×3 float64 symmetric strain tensor.
    """
    e = np.zeros((3, 3))
    if   voigt_index == 0: e[0, 0] = delta
    elif voigt_index == 1: e[1, 1] = delta
    elif voigt_index == 2: e[2, 2] = delta
    elif voigt_index == 3: e[1, 2] = e[2, 1] = delta / 2.0
    elif voigt_index == 4: e[0, 2] = e[2, 0] = delta / 2.0
    elif voigt_index == 5: e[0, 1] = e[1, 0] = delta / 2.0
    return e


def _apply_strain(atoms, strain: np.ndarray):
    """Return a new Atoms with the homogeneous strain F = I + ε applied.

    Cell rows are updated as ``cell_new = cell_old @ F.T``.  Fractional
    coordinates are preserved (``scale_atoms=True``).  No calculator is
    attached to the returned object.
    """
    F = np.eye(3) + strain
    strained = atoms.copy()
    strained.set_cell(atoms.cell.array @ F.T, scale_atoms=True)
    return strained


def run_elastic(
        sym_dir: str,
        device: str = "cpu",
        force_rerun: bool = False,
) -> bool:
    """Compute the 6×6 Voigt elastic stiffness tensor.

    Applies ±``config.ELASTIC_DELTA`` strain along each of the 6 independent
    Voigt directions, evaluates the calculator stress at each deformed
    geometry, and assembles ``C_αβ`` by central differencing:

    .. math::

        C_{\\alpha\\beta} =
            \\frac{\\sigma_\\alpha(+\\delta) - \\sigma_\\alpha(-\\delta)}{2\\delta}

    The result is symmetrised and converted to GPa.  Voigt-averaged bulk,
    shear, Young's moduli, and Poisson's ratio are computed analytically.
    The output ``elastic/elastic_tensor.json`` is the input to
    :func:`heatup.mechanical.assess_mechanical_stability` (Gate 1).

    Args:
        sym_dir:     Path to the symmetry directory.
        device:      Compute device for the calculator.
        force_rerun: Redo even if ``elastic/elastic_tensor.json`` already exists.

    Returns:
        ``True`` on success, ``False`` if ``relaxation/CONTCAR`` is missing.
    """
    tag     = _tag(sym_dir)
    el_dir  = os.path.join(sym_dir, "elastic")
    et_path = os.path.join(el_dir, "elastic_tensor.json")
    contcar = os.path.join(sym_dir, "relaxation", "CONTCAR")

    if not force_rerun and os.path.exists(et_path):
        print(f"[done] {tag}/elastic — elastic_tensor.json exists (use force_rerun=True).")
        return True

    if not os.path.exists(contcar):
        print(f"[skip] {tag}/elastic: relaxation/CONTCAR not found. Run relaxation first.")
        return False

    os.makedirs(el_dir, exist_ok=True)
    dest = os.path.join(el_dir, "POSCAR")
    shutil.copy2(contcar, dest)

    with open(os.path.join(el_dir, "elastic-input.json"), "w") as fh:
        json.dump(_elastic_params(device), fh, indent=4)

    calc = None
    try:
        atoms = read_vasp(file=dest)
        calc  = build_calculator(device=device)

        n_voigt = 6
        C       = np.zeros((n_voigt, n_voigt))

        print(f"Computing elastic tensor for {tag}  "
              f"(delta=±{config.ELASTIC_DELTA}, "
              f"{2 * n_voigt} single-points, "
              f"backend={calculator_label()}) ...")

        for beta in range(n_voigt):
            strain_mat = _voigt_strain_matrix(beta, config.ELASTIC_DELTA)

            fwd = _apply_strain(atoms, +strain_mat); fwd.calc = calc
            bwd = _apply_strain(atoms, -strain_mat); bwd.calc = calc

            sigma_fwd = fwd.get_stress(voigt=True)  # eV/Å³ [xx,yy,zz,yz,xz,xy]
            sigma_bwd = bwd.get_stress(voigt=True)
            C[:, beta] = (sigma_fwd - sigma_bwd) / (2.0 * config.ELASTIC_DELTA)

        C_GPa = 0.5 * (C + C.T) * 160.21766208  # symmetrise + eV/Å³ → GPa

        # Voigt isotropic averages.
        B = (  C_GPa[0,0] + C_GPa[1,1] + C_GPa[2,2]
             + 2*(C_GPa[0,1] + C_GPa[0,2] + C_GPa[1,2])) / 9.0
        G = (  C_GPa[0,0] + C_GPa[1,1] + C_GPa[2,2]
             - C_GPa[0,1] - C_GPa[0,2] - C_GPa[1,2]
             + 3*(C_GPa[3,3] + C_GPa[4,4] + C_GPa[5,5])) / 15.0
        E  = 9.0 * B * G / (3.0 * B + G)
        nu = (3.0 * B - 2.0 * G) / (6.0 * B + 2.0 * G)

        with open(et_path, "w") as fh:
            json.dump({
                "elastic_tensor_GPa" : C_GPa.tolist(),
                "voigt_labels"       : ["xx","yy","zz","yz","xz","xy"],
                "derived_moduli"     : {
                    "bulk_modulus_voigt_GPa"  : float(B),
                    "shear_modulus_voigt_GPa" : float(G),
                    "youngs_modulus_voigt_GPa": float(E),
                    "poissons_ratio_voigt"    : float(nu),
                },
                "calculator" : calculator_label(),
            }, fh, indent=4)

        print(f"  Elastic done.  B={B:.1f}  G={G:.1f}  E={E:.1f} GPa  ν={nu:.3f}")
        return True

    except Exception as exc:
        print(f"  [error] {tag}/elastic: {exc}")
        traceback.print_exc()
        return False

    finally:
        release_calculator(calc)


def prepare_aimd_folders(
        sym_dir: str,
        temperatures: list[float],
        force_rebuild: bool = False,
) -> bool:
    """Build an AIMD supercell and distribute it to temperature sub-folders.

    Reads ``relaxation/CONTCAR``, converts the cell to upper-triangular form
    (required by ASE's NPT integrator), tiles it until every lattice parameter
    is at least ``config.AIMD_MIN_CELL_ANG`` Å, then copies the result into
    ``aimd/<T>K/POSCAR`` for each requested temperature.

    Args:
        sym_dir:       Path to the symmetry directory.
        temperatures:  List of target temperatures in Kelvin.
        force_rebuild: Rebuild the supercell and overwrite all temperature
                       POSCAR files even if they already exist.

    Returns:
        ``True`` on success, ``False`` if ``relaxation/CONTCAR`` is missing.
    """
    tag     = _tag(sym_dir)
    contcar = os.path.join(sym_dir, "relaxation", "CONTCAR")

    if not os.path.exists(contcar):
        print(f"[skip] {tag}/aimd: relaxation/CONTCAR not found. Run relaxation first.")
        return False

    aimd_dir  = os.path.join(sym_dir, "aimd")
    sc_poscar = os.path.join(aimd_dir, "POSCAR")

    if force_rebuild or not os.path.exists(sc_poscar):
        os.makedirs(aimd_dir, exist_ok=True)

        unit  = read_vasp(file=contcar)
        unit  = _make_cell_upper_triangular(unit)
        reps  = _compute_supercell_repetitions(unit.get_cell(), config.AIMD_MIN_CELL_ANG)
        sc    = unit.repeat(reps)

        unit_len = np.linalg.norm(unit.get_cell(), axis=1)
        sc_len   = np.linalg.norm(sc.get_cell(), axis=1)
        print(f"AIMD supercell for {tag}:  reps={reps}  "
              f"unit={unit_len.round(2)} Å  sc={sc_len.round(2)} Å  "
              f"atoms={len(sc)}")

        write_vasp(sc_poscar,                                   sc,   direct=True, sort=True)
        write_vasp(os.path.join(aimd_dir, "POSCAR-unitcell"),   unit, direct=True, sort=True)

        with open(os.path.join(aimd_dir, "aimd-supercell.json"), "w") as fh:
            json.dump({
                "symmetry"          : os.path.basename(os.path.abspath(sym_dir)),
                "material"          : os.path.basename(os.path.dirname(os.path.abspath(sym_dir))),
                "repetitions"       : list(reps),
                "min_cell_ang"      : config.AIMD_MIN_CELL_ANG,
                "unit_cell_lengths" : unit_len.tolist(),
                "supercell_lengths" : sc_len.tolist(),
                "n_atoms_unit"      : len(unit),
                "n_atoms_supercell" : len(sc),
            }, fh, indent=4)
        print(f"  Master supercell → {sc_poscar}")
    else:
        print(f"[done] {tag}/aimd — POSCAR exists (use force_rebuild=True to rebuild).")

    for T in temperatures:
        t_dir = os.path.join(aimd_dir, f"{int(T)}K")
        os.makedirs(t_dir, exist_ok=True)
        dst = os.path.join(t_dir, "POSCAR")
        if force_rebuild or not os.path.exists(dst):
            shutil.copy2(sc_poscar, dst)
            print(f"  Written → {dst}")
        else:
            print(f"  [skip] {dst} already exists.")

    return True


# ---------------------------------------------------------------------------
# Subprocess wrappers — CUDA-isolated execution
# ---------------------------------------------------------------------------
# Running multiple materials in the same process leaves GPU memory fragmented
# even after ``del calc`` + ``gc.collect()``.  These wrappers spawn child
# processes; when a child exits, the OS reclaims the CUDA context entirely.
#
# Implementation note: instead of calling external script files, each wrapper
# re-invokes the **current module** with a ``--_run`` flag.  This keeps all
# logic in one place and avoids maintaining separate run_single_*.py scripts.

def _self_path() -> str:
    """Return the absolute path of this module file."""
    return os.path.abspath(__file__)


def _run_isolated(
        sym_dir: str,
        step: str,
        device: str,
        force_rerun: bool,
) -> bool:
    """Spawn a subprocess that calls ``run_<step>`` inside this module.

    Args:
        sym_dir:     Symmetry directory path.
        step:        One of ``"relaxation"``, ``"phonons"``, ``"elastic"``.
        device:      Compute device string.
        force_rerun: Whether to pass ``--force``.

    Returns:
        ``True`` if the subprocess exits with code 0.
    """
    cmd = [
        sys.executable, _self_path(),
        "--_run", step,
        "--sym_dir", sym_dir,
        "--device",  device,
    ]
    if force_rerun:
        cmd.append("--force")

    proc = subprocess.run(cmd, env=_cuda_env(), check=False)
    if proc.returncode != 0:
        print(f"  [error] {step} subprocess exited with code {proc.returncode}")
        return False
    return True


def run_relaxation_subprocess(
        sym_dir: str,
        device: str = "cpu",
        force_rerun: bool = False,
) -> bool:
    """Run geometry relaxation in a CUDA-isolated subprocess.

    Identical contract to :func:`run_relaxation` but spawns a child process
    so GPU memory is fully released on exit.  Recommended for batch runs.
    """
    return _run_isolated(sym_dir, "relaxation", device, force_rerun)


def run_phonons_subprocess(
        sym_dir: str,
        device: str = "cpu",
        force_rerun: bool = False,
) -> bool:
    """Run phonon calculation in a CUDA-isolated subprocess.

    Identical contract to :func:`run_phonons` but spawns a child process.
    Phonons require many forward passes (one per displacement) and are the
    most likely step to leave GPU memory fragmented.
    """
    return _run_isolated(sym_dir, "phonons", device, force_rerun)


def run_elastic_subprocess(
        sym_dir: str,
        device: str = "cpu",
        force_rerun: bool = False,
) -> bool:
    """Run elastic tensor calculation in a CUDA-isolated subprocess.

    Identical contract to :func:`run_elastic` but spawns a child process.
    """
    return _run_isolated(sym_dir, "elastic", device, force_rerun)


# ---------------------------------------------------------------------------
# __main__ entry point — used by subprocess wrappers above
# ---------------------------------------------------------------------------

def _main_subprocess() -> int:
    """Parse CLI args and run the requested step.  Called by subprocess wrappers."""
    import argparse
    parser = argparse.ArgumentParser(description="HeatUp structure step runner.")
    parser.add_argument("--_run",    required=True, choices=["relaxation","phonons","elastic"])
    parser.add_argument("--sym_dir", required=True)
    parser.add_argument("--device",  default="cpu")
    parser.add_argument("--force",   action="store_true")
    args = parser.parse_args()

    fn = {"relaxation": run_relaxation,
          "phonons":    run_phonons,
          "elastic":    run_elastic}[args._run]

    ok = fn(args.sym_dir, device=args.device, force_rerun=args.force)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_main_subprocess())
