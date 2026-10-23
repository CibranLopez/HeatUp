"""heatup.compute.phonons
=========================
Harmonic (HA) and quasi-harmonic (QHA) phonons with phonopy.

``phonons`` (:func:`run_ha`)  ->  ``phonons/ha.json``
    finite displacements (symmetry-reduced), residual-force subtraction, q-mesh frequencies reduced to a
    :class:`heatup.analyze.thermo.ModeSet` (3 modes per atom, imaginary modes preserved) + band structure along the
    ASE-generated high-symmetry path.

``qha`` (:func:`run_qha`)  ->  ``phonons/qha.json``
    HA at ``qha.n_volumes`` volumes **with ions relaxed at each volume**, F(V,T) = E(V) + F_vib(V,T) fitted per
    temperature on the grid ``qha.temperatures``; gives G(T), V(T), alpha(T), B_T(T).  The cached results are read
    by :func:`heatup.analyze.thermo.load_ha` / ``load_qha``.

Choices that matter: phonopy DOS is kept at 3 states/atom; ``PhonopyQHA`` is not used (free-energy fit done here,
ions relaxed at each volume); the supercell comes from perpendicular widths, not a fixed 3x3x3; displacement 0.01 A
(0.05 A was anharmonically contaminated); residual forces of the perfect supercell are subtracted.
"""

from __future__ import annotations

import logging
import os

import numpy as np
from ase import Atoms
from ase.optimize import BFGS

from heatup.analyze.thermo import ModeSet
from heatup.compute.calculator import build_calculator, calculator_label, release_calculator
from heatup.core import ledger
from heatup.core.config import GPA_TO_EV_A3, MEV_PER_THZ, cfg
from heatup.core.layout import PhaseDir, read_structure, write_json
from heatup.prepare.export import phonon_cells, qha_scales, scaled_cell, to_ase

log = logging.getLogger("heatup")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _forces(ph, calc) -> np.ndarray:
    """Forces of all displaced supercells minus the residual force of the perfect one."""
    ref = to_ase(ph.supercell)
    ref.calc = calc
    f0 = ref.get_forces()
    out = []
    for sc in ph.supercells_with_displacements:
        a = to_ase(sc)
        a.calc = calc
        out.append(a.get_forces() - f0)
        a.calc = None
    return np.array(out)


def modes_from_forces(ph, atoms: Atoms, forces: np.ndarray, sc, *, with_band: bool = False):
    """Forces of the displaced supercells (residual of the perfect one already subtracted) -> (ModeSet, info, phonopy)."""
    ph.forces = forces
    ph.produce_force_constants()
    ph.run_mesh(cfg.phonon.mesh_length_A, is_gamma_center=True)
    mesh = ph.mesh
    if np.asarray(mesh.frequencies).shape[1] != 3 * len(atoms):
        raise RuntimeError(f"phonopy returned {np.asarray(mesh.frequencies).shape[1]} bands for "
                           f"{len(atoms)} atoms - primitive cell was reduced; ModeSet weights would be wrong")
    freq_meV = np.asarray(mesh.frequencies) * MEV_PER_THZ
    modes = ModeSet.from_frequencies(freq_meV, np.asarray(mesh.weights), n_atoms=len(atoms),
                                     source="HA", mesh=[int(x) for x in mesh.mesh_numbers],
                                     supercell=list(sc))
    info = {"supercell": list(sc), "mesh": [int(x) for x in mesh.mesh_numbers],
            "n_displacements": len(ph.supercells_with_displacements)}
    if with_band:
        info["band"] = _band(ph, atoms)
    return modes, info, ph


def harmonic_modes(atoms: Atoms, calc, *, with_band: bool = False):
    """Relaxed *atoms* -> (ModeSet, info dict, phonopy object), forces from *calc*."""
    ph, sc = phonon_cells(atoms)
    log.info("  phonopy: supercell %s (%d atoms), %d displacements",
             sc, len(atoms) * int(np.prod(sc)), len(ph.supercells_with_displacements))
    return modes_from_forces(ph, atoms, _forces(ph, calc), sc, with_band=with_band)


def _band(ph, atoms: Atoms) -> dict:
    path = atoms.cell.bandpath(npoints=cfg.phonon.band_npoints)
    kpts = np.asarray(path.kpts)
    # one segment per contiguous sub-path; phonopy wants a list of segments
    x, X, labels = path.get_linear_kpoint_axis()
    ph.run_band_structure([kpts], with_eigenvectors=False)
    freq = np.asarray(ph.band_structure.frequencies[0]) * MEV_PER_THZ   # (nk, nbands)
    return {"x": np.asarray(x).tolist(), "special_x": np.asarray(X).tolist(),
            "labels": list(labels), "frequencies_meV": freq.tolist()}


def _plot_ha(modes: ModeSet, band: dict | None, path: str, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2 if band else 1, figsize=(8 if band else 4, 4),
                            gridspec_kw={"width_ratios": [3, 1]} if band else None)
    axs = np.atleast_1d(axs)
    k = 0
    if band:
        f = np.array(band["frequencies_meV"])
        axs[0].plot(band["x"], f, color="C0", lw=0.9)
        for xx in band["special_x"]:
            axs[0].axvline(xx, color="0.7", lw=0.5)
        axs[0].set_xticks(band["special_x"])
        axs[0].set_xticklabels([l.replace("G", "Γ") for l in band["labels"]])
        axs[0].axhline(0, color="k", lw=0.5)
        axs[0].set_ylabel("ω (meV)")
        axs[0].set_title(title, fontsize=9)
        k = 1
    g, d = modes.smooth_dos(cfg.phonon.dos_sigma_meV)
    if band:
        axs[k].fill_betweenx(g, 0, d, color="C0", alpha=0.6)
        axs[k].set_ylim(axs[0].get_ylim())
        axs[k].set_xlabel("DOS")
        axs[k].set_yticks([])
    else:
        axs[k].fill_between(g, d, color="C0", alpha=0.6)
        axs[k].set_xlabel("ω (meV)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# HA
# ---------------------------------------------------------------------------

def run_ha(phase_dir, device: str = "cpu", force: bool = False) -> None:
    phase = PhaseDir.of(phase_dir)
    out = os.path.join(phase.phonon_dir, "ha.json")
    if not force and os.path.exists(out) and ledger.is_fresh(phase, "phonons"):
        log.info("[cached] %s/phonons (up to date, not recomputed)", phase.tag)
        return
    if not phase.has_relaxation():
        raise FileNotFoundError(f"{phase.tag}: converged relaxation required first")

    os.makedirs(phase.phonon_dir, exist_ok=True)
    atoms = read_structure(phase.contcar)
    calc = build_calculator(device, material=phase.material)
    try:
        log.info("HA phonons %s (delta=%g A, %s)", phase.tag, cfg.phonon.delta_A,
                 calculator_label(phase.material))
        modes, info, _ = harmonic_modes(atoms, calc, with_band=True)
    finally:
        release_calculator(calc)

    res = _write_ha(phase, atoms, modes, info, calculator_label(phase.material))
    ledger.record(phase, "phonons")
    log.info("  imaginary weight %.3f%% (max %.2f meV), ZPE %.1f meV/atom",
             100 * res["imag_fraction"], res["max_imag_meV"], 1e3 * res["zpe_eV_per_atom"])


def _write_ha(phase: PhaseDir, atoms: Atoms, modes: ModeSet, info: dict, calculator: str) -> dict:
    """``phonons/ha.json`` (+ plot) from reduced modes - shared by the MLIP step and by :func:`collect_ha`."""
    os.makedirs(phase.phonon_dir, exist_ok=True)
    tol = cfg.phonon.imag_tol_meV
    res = {
        "modes": modes.to_dict(max_points=6000),
        "imag_fraction": modes.imag_fraction(tol),
        "max_imag_meV": modes.max_imag_meV(),
        "imag_tol_meV": tol,
        "zpe_eV_per_atom": modes.zpe(),
        "n_atoms": len(atoms),
        "calculator": calculator,
        **{k: v for k, v in info.items() if k != "band"},
        "band": info.get("band"),
    }
    write_json(os.path.join(phase.phonon_dir, "ha.json"), res, indent=None)
    try:
        _plot_ha(modes, info.get("band"), os.path.join(phase.phonon_dir, "phonon.png"), phase.tag)
    except Exception as exc:  # noqa: BLE001
        log.warning("phonon plot failed: %s", exc)
    return res


def collect_ha(phase_dir, forces, perfect_forces=None, calculator: str = "external") -> dict:
    """``phonons/ha.json`` from forces computed with ANY code on the cells of ``phonons/ha/`` (see ``prepare.export``).

    ``forces``: array (n_displacements, n_atoms_supercell, 3) in eV/A, ordered like ``disp-001, disp-002, ...`` (or a path to a
    ``.npy``); ``perfect_forces``: the same for ``SPOSCAR`` (its residual is subtracted).  No manifest is written: the result
    is reported ``external`` and accepted as it is.  ``calculator`` is the label shared with ``energy.json`` of the phase."""
    phase = PhaseDir.of(phase_dir)
    atoms = read_structure(phase.contcar)
    ph, sc = phonon_cells(atoms)
    load = lambda x: np.load(x) if isinstance(x, (str, os.PathLike)) else np.asarray(x, float)
    f = load(forces)
    if perfect_forces is not None:
        f = f - load(perfect_forces)
    n_atoms_sc = len(to_ase(ph.supercell))
    if f.shape != (len(ph.supercells_with_displacements), n_atoms_sc, 3):
        raise ValueError(f"forces have shape {f.shape}, expected {(len(ph.supercells_with_displacements), n_atoms_sc, 3)} "
                         "(displacement cells of phonons/ha in the order disp-001, disp-002, ...)")
    modes, info, _ = modes_from_forces(ph, atoms, f, sc, with_band=True)
    res = _write_ha(phase, atoms, modes, info, calculator)
    log.info("%s: HA phonons collected (%s): imaginary weight %.3f%%, ZPE %.1f meV/atom", phase.tag, calculator,
             100 * res["imag_fraction"], 1e3 * res["zpe_eV_per_atom"])
    return res


# ---------------------------------------------------------------------------
# QHA
# ---------------------------------------------------------------------------

def run_qha(phase_dir, device: str = "cpu", force: bool = False) -> None:
    phase = PhaseDir.of(phase_dir)
    out = os.path.join(phase.phonon_dir, "qha.json")
    qc = cfg.qha
    if not force and os.path.exists(out) and ledger.is_fresh(phase, "qha"):
        log.info("[cached] %s/qha (up to date, not recomputed)", phase.tag)
        return
    if not phase.has_relaxation():
        raise FileNotFoundError(f"{phase.tag}: converged relaxation required first")
    if qc.n_volumes < 5 or qc.n_volumes % 2 == 0:
        raise ValueError("qha.n_volumes must be odd and >= 5")

    os.makedirs(phase.phonon_dir, exist_ok=True)
    ref = read_structure(phase.contcar)
    n = len(ref)
    V0 = ref.get_volume()
    scales = qha_scales()
    calc = build_calculator(device, material=phase.material)
    vols, e_el, mode_dicts = [], [], []
    try:
        for i, s in enumerate(scales):
            a = scaled_cell(ref, s)
            a.calc = calc
            if qc.relax_internal:
                BFGS(a, logfile=None).run(fmax=cfg.relax.fmax, steps=cfg.relax.max_steps)
            e_el.append(a.get_potential_energy() / n)
            vols.append(a.get_volume() / n)
            a.calc = None
            modes, _, _ = harmonic_modes(a, calc)
            mode_dicts.append(modes.to_dict(max_points=3000))
            log.info("  QHA volume %d/%d  V/V0=%.3f  E=%.5f eV/atom  imag=%.2f%%",
                     i + 1, len(scales), s, e_el[-1], 100 * modes.imag_fraction())
    finally:
        release_calculator(calc)

    T = np.array(qc.temperatures, float)
    vols, e_el = np.array(vols), np.array(e_el)
    modesets = [ModeSet.from_dict(d) for d in mode_dicts]
    Fvib = np.array([[m.f_vib(t) for t in T] for m in modesets])          # (nV, nT)
    P = cfg.relax.pressure_GPa * GPA_TO_EV_A3
    Vfine = np.linspace(vols.min(), vols.max(), 2001)
    G, Veq, B, inside = [], [], [], []
    for j in range(len(T)):
        Fv = e_el + Fvib[:, j]
        coef = np.polyfit(vols, Fv, qc.poly_order)
        Ffine = np.polyval(coef, Vfine) + P * Vfine
        k = int(np.argmin(Ffine))
        inside.append(0 < k < len(Vfine) - 1)
        Veq.append(Vfine[k])
        G.append(Ffine[k])
        d2 = np.polyval(np.polyder(coef, 2), Vfine[k])
        B.append(Vfine[k] * d2 / GPA_TO_EV_A3)
    Veq = np.array(Veq)
    alpha = np.gradient(Veq, T) / Veq if len(T) > 1 else np.zeros_like(T)
    write_json(out, {
        "temperatures": T, "G_eV_per_atom": G, "V_eq_A3_per_atom": Veq,
        "alpha_V_1_K": alpha, "B_T_GPa": B, "valid": inside,
        "volumes_A3_per_atom": vols, "E_el_eV_per_atom": e_el,
        "F_vib_eV_per_atom": Fvib, "poly_order": qc.poly_order,
        "pressure_GPa": cfg.relax.pressure_GPa, "V0_ref_A3_per_atom": V0 / n,
        "imag_fraction": [m.imag_fraction(cfg.phonon.imag_tol_meV) for m in modesets],
        "calculator": calculator_label(phase.material),
    }, indent=None)
    ledger.record(phase, "qha")
    if not all(inside):
        log.warning("QHA: V_eq leaves the sampled range above T = %.0f K - widen qha.volume_range",
                    T[~np.array(inside)].min())
