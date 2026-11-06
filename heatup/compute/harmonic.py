"""heatup.compute.harmonic
=========================
HARMONIC phonons with phonopy (vocabulary and file meta: ``heatup.core.provenance``): second-order force constants at the
relaxed 0 K cell, no thermal expansion.  The quasi-harmonic level (the same at several volumes) is
``heatup.compute.quasiharmonic``, the perturbative one (fc3) ``heatup.compute.perturbative``, the anharmonic one (phonons
from MD) ``heatup.compute.md`` + ``heatup.analyze.trajectory``.

Step ``harmonic`` (:func:`run_harmonic`)  ->  ``phonons/harmonic/harmonic.json`` (+ ``phonon.png``)
    finite displacements (symmetry-reduced), residual-force subtraction, q-mesh frequencies reduced to a
    :class:`heatup.analyze.thermo.ModeSet` (3 modes per atom, imaginary modes preserved) + band structure along the
    ASE-generated high-symmetry path (with the q-points, so an imaginary branch can be located).  The displaced cells and
    forces are written in the same folder (``prepare.export`` / ``collect_harmonic`` take forces from any code).

Choices that matter: phonopy DOS is kept at 3 states/atom; the supercell comes from perpendicular widths, not a fixed
3x3x3; displacement 0.01 A (0.05 A was anharmonically contaminated); residual forces of the perfect supercell are subtracted.
"""

from __future__ import annotations

import logging
import os

import numpy as np
from ase import Atoms

from heatup.analyze.thermo import ModeSet
from heatup.compute.calculator import build_calculator, calculator_label, release_calculator
from heatup.core import ledger, provenance
from heatup.core.config import MEV_PER_THZ, cfg
from heatup.core.layout import PhaseDir, read_structure, write_json
from heatup.prepare.export import harmonic_cells, to_ase

log = logging.getLogger("heatup")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _forces(ph, calc, save_dir: str | None = None) -> np.ndarray:
    """Forces of all displaced supercells minus the residual force of the perfect one.
    ``save_dir``: also write ``forces.npy`` (what is returned; ready for :func:`collect_ha`) and ``forces_perfect.npy``."""
    ref = to_ase(ph.supercell)
    ref.calc = calc
    f0 = ref.get_forces()
    out = []
    for sc in ph.supercells_with_displacements:
        a = to_ase(sc)
        a.calc = calc
        out.append(a.get_forces() - f0)
        a.calc = None
    out = np.array(out)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "forces.npy"), out)
        np.save(os.path.join(save_dir, "forces_perfect.npy"), f0)
    return out


def modes_from_forces(ph, atoms: Atoms, forces: np.ndarray, sc, *, with_band: bool = False):
    """Forces of the displaced supercells (residual of the perfect one already subtracted) -> (ModeSet, info, phonopy)."""
    ph.forces = forces
    ph.produce_force_constants()
    ph.run_mesh(cfg.harmonic.mesh_length_A, is_gamma_center=True)
    mesh = ph.mesh
    if np.asarray(mesh.frequencies).shape[1] != 3 * len(atoms):
        raise RuntimeError(f"phonopy returned {np.asarray(mesh.frequencies).shape[1]} bands for "
                           f"{len(atoms)} atoms - primitive cell was reduced; ModeSet weights would be wrong")
    freq_meV = np.asarray(mesh.frequencies) * MEV_PER_THZ
    modes = ModeSet.from_frequencies(freq_meV, np.asarray(mesh.weights), n_atoms=len(atoms),
                                     source="harmonic", mesh=[int(x) for x in mesh.mesh_numbers],
                                     supercell=list(sc))
    info = {"supercell": list(sc), "mesh": [int(x) for x in mesh.mesh_numbers],
            "n_displacements": len(ph.supercells_with_displacements)}
    if with_band:
        info["band"] = _band(ph, atoms)
    return modes, info, ph


def harmonic_modes(atoms: Atoms, calc, *, with_band: bool = False, save_dir: str | None = None,
                   sc: tuple[int, int, int] | None = None):
    """Relaxed *atoms* -> (ModeSet, info dict, phonopy object), forces from *calc*.
    ``save_dir``: also write the forces (see :func:`_forces`); ``sc``: supercell repetitions (default: chosen from *atoms*)."""
    ph, sc = harmonic_cells(atoms, sc)
    log.info("  phonopy: supercell %s (%d atoms), %d displacements",
             sc, len(atoms) * int(np.prod(sc)), len(ph.supercells_with_displacements))
    return modes_from_forces(ph, atoms, _forces(ph, calc, save_dir), sc, with_band=with_band)


def _band(ph, atoms: Atoms) -> dict:
    path = atoms.cell.bandpath(npoints=cfg.harmonic.band_npoints)
    kpts = np.asarray(path.kpts)
    # one segment per contiguous sub-path; phonopy wants a list of segments
    x, X, labels = path.get_linear_kpoint_axis()
    ph.run_band_structure([kpts], with_eigenvectors=False)
    freq = np.asarray(ph.band_structure.frequencies[0]) * MEV_PER_THZ   # (nk, nbands)
    return {"x": np.asarray(x).tolist(), "special_x": np.asarray(X).tolist(), "labels": list(labels),
            "kpoints": kpts.tolist(), "frequencies_meV": freq.tolist()}


def _plot_harmonic(res: dict, path: str, title: str) -> None:
    """``phonon.png``: bands with the imaginary parts in red + DOS (the same drawing notebook 3 shows)."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    from heatup.analyze import vibrations
    ha = dict(res)
    ha["modes"] = ModeSet.from_dict(res["modes"]) if isinstance(res["modes"], dict) else res["modes"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 4), gridspec_kw={"width_ratios": [3, 1]})
    vibrations.plot_harmonic(ha, a1, a2, title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# step harmonic
# ---------------------------------------------------------------------------

def run_harmonic(phase_dir, device: str = "cpu", force: bool = False) -> None:
    phase = PhaseDir.of(phase_dir)
    out = os.path.join(phase.harmonic_dir, "harmonic.json")
    if not force and os.path.exists(out) and ledger.is_fresh(phase, "harmonic"):
        log.info("[cached] %s/harmonic (up to date, not recomputed)", phase.tag)
        return
    if not phase.has_relaxation():
        raise FileNotFoundError(f"{phase.tag}: converged relaxation required first")

    os.makedirs(phase.harmonic_dir, exist_ok=True)
    atoms = read_structure(phase.contcar)
    calc = build_calculator(device, material=phase.material)
    try:
        log.info("harmonic phonons %s (delta=%g A, %s)", phase.tag, cfg.harmonic.delta_A,
                 calculator_label(phase.material))
        modes, info, _ = harmonic_modes(atoms, calc, with_band=True,
                                        save_dir=phase.harmonic_dir)
    finally:
        release_calculator(calc)

    res = _write_ha(phase, atoms, modes, info, calculator_label(phase.material))
    ledger.record(phase, "harmonic")
    log.info("  imaginary weight %.3f%% (max %.2f meV), ZPE %.1f meV/atom",
             100 * res["imag_fraction"], res["max_imag_meV"], 1e3 * res["zpe_eV_per_atom"])


def _write_ha(phase: PhaseDir, atoms: Atoms, modes: ModeSet, info: dict, calculator: str) -> dict:
    """``harmonic/harmonic.json`` (+ plot) from reduced modes - shared by the MLIP step and by :func:`collect_harmonic`."""
    os.makedirs(phase.harmonic_dir, exist_ok=True)
    tol = cfg.harmonic.imag_tol_meV
    res = {
        "modes": modes.to_dict(max_points=6000),
        "imag_fraction": modes.imag_fraction(tol),
        "max_imag_meV": modes.max_imag_meV(),
        "imag_tol_meV": tol,
        "zpe_eV_per_atom": modes.zpe(),
        "n_atoms": len(atoms),
        "calculator": calculator,
        "meta": provenance.stamp("harmonic"),
        **{k: v for k, v in info.items() if k != "band"},
        "band": info.get("band"),
    }
    write_json(os.path.join(phase.harmonic_dir, "harmonic.json"), res, indent=None)
    try:
        _plot_harmonic(res, os.path.join(phase.harmonic_dir, "phonon.png"), phase.tag)
    except Exception as exc:  # noqa: BLE001
        log.warning("phonon plot failed: %s", exc)
    return res


def collect_harmonic(phase_dir, forces, perfect_forces=None, calculator: str = "external") -> dict:
    """``harmonic/harmonic.json`` from forces computed with ANY code on the cells of ``harmonic/`` (see ``prepare.export``).

    ``forces``: array (n_displacements, n_atoms_supercell, 3) in eV/A, ordered like ``disp-001, disp-002, ...`` (or a path to a
    ``.npy``); ``perfect_forces``: the same for ``SPOSCAR`` (its residual is subtracted).  No manifest is written: the result
    is reported ``external`` and accepted as it is.  ``calculator`` is the label shared with ``energy.json`` of the phase."""
    phase = PhaseDir.of(phase_dir)
    atoms = read_structure(phase.contcar)
    ph, sc = harmonic_cells(atoms)
    load = lambda x: np.load(x) if isinstance(x, (str, os.PathLike)) else np.asarray(x, float)
    f = load(forces)
    if perfect_forces is not None:
        f = f - load(perfect_forces)
    n_atoms_sc = len(to_ase(ph.supercell))
    if f.shape != (len(ph.supercells_with_displacements), n_atoms_sc, 3):
        raise ValueError(f"forces have shape {f.shape}, expected {(len(ph.supercells_with_displacements), n_atoms_sc, 3)} "
                         "(displacement cells of harmonic/ in the order disp-001, disp-002, ...)")
    modes, info, _ = modes_from_forces(ph, atoms, f, sc, with_band=True)
    res = _write_ha(phase, atoms, modes, info, calculator)
    log.info("%s: harmonic phonons collected (%s): imaginary weight %.3f%%, ZPE %.1f meV/atom", phase.tag, calculator,
             100 * res["imag_fraction"], 1e3 * res["zpe_eV_per_atom"])
    return res
