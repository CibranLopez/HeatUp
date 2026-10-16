"""heatup.phonons
==================
Harmonic (HA) and quasi-harmonic (QHA) phonons with phonopy.

Steps
-----
``phonons`` (:func:`run_ha`)  ->  ``phonons/ha.json``
    finite displacements (symmetry-reduced), residual-force subtraction,
    q-mesh frequencies reduced to a :class:`heatup.vibthermo.ModeSet`
    (3 modes per atom, imaginary modes preserved) + band structure along the
    ASE-generated high-symmetry path.

``qha`` (:func:`run_qha`)  ->  ``phonons/qha.json``
    HA at ``qha.n_volumes`` volumes **with ions relaxed at each volume**,
    F(V,T) = E(V) + F_vib(V,T) fitted per temperature; gives G(T), V(T),
    alpha(T), B_T(T).

What was wrong in v0.2
----------------------
* phonopy DOS was renormalised to 1 state/atom (3x error in F_vib);
* QHA called ``PhonopyQHA`` with an invalid signature (DOS arrays instead of
  free-energy / Cv / entropy matrices) and never relaxed the ions;
* fixed 3x3x3 supercell regardless of cell size (a 40-atom cell -> 1080 atoms,
  a 3 Å primitive cell -> 9 Å supercell); now chosen from perpendicular widths;
* displacement 0.05 Å (anharmonic contamination) -> 0.01 Å;
* no residual-force subtraction.
"""

from __future__ import annotations

import logging
import os

import numpy as np
from ase import Atoms
from ase.optimize import BFGS

from heatup import manifest
from heatup.calculator import build_calculator, calculator_label, release_calculator
from heatup.config import cfg
from heatup.constants import GPA_TO_EV_A3, MEV_PER_THZ
from heatup.paths import PhaseDir, read_json, write_json
from heatup.structure import read_structure, supercell_reps
from heatup.vibthermo import ModeSet

log = logging.getLogger("heatup")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def choose_supercell(atoms: Atoms) -> tuple[int, int, int]:
    pc = cfg.phonon
    if pc.supercell is not None:
        return tuple(int(x) for x in pc.supercell)  # type: ignore[return-value]
    return supercell_reps(atoms.cell.array, pc.min_length_A, pc.max_atoms, len(atoms))


def _phonopy(atoms: Atoms, sc: tuple[int, int, int]):
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    pa = PhonopyAtoms(symbols=atoms.get_chemical_symbols(), cell=atoms.cell.array,
                      scaled_positions=atoms.get_scaled_positions())
    return Phonopy(pa, supercell_matrix=np.diag(sc), primitive_matrix="P",   # identity: n_bands = 3 n_atoms (phonopy >= 4 defaults to "auto")
                   symprec=cfg.phonon.symprec)


def _to_ase(sc) -> Atoms:
    return Atoms(sc.symbols, cell=sc.cell, scaled_positions=sc.scaled_positions, pbc=True)


def _forces(ph, calc) -> np.ndarray:
    """Forces of all displaced supercells minus the residual force of the perfect one."""
    ref = _to_ase(ph.supercell)
    ref.calc = calc
    f0 = ref.get_forces()
    out = []
    for sc in ph.supercells_with_displacements:
        a = _to_ase(sc)
        a.calc = calc
        out.append(a.get_forces() - f0)
        a.calc = None
    return np.array(out)


def harmonic_modes(atoms: Atoms, calc, *, with_band: bool = False):
    """Relaxed *atoms* -> (ModeSet, info dict, phonopy object)."""
    sc = choose_supercell(atoms)
    ph = _phonopy(atoms, sc)
    ph.generate_displacements(distance=cfg.phonon.delta_A)
    log.info("  phonopy: supercell %s (%d atoms), %d displacements",
             sc, len(atoms) * int(np.prod(sc)), len(ph.supercells_with_displacements))
    ph.forces = _forces(ph, calc)
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
    inputs = [phase.contcar]
    if (not force and os.path.exists(out)
            and manifest.is_fresh(phase.phonon_dir, "phonons", phase.material, inputs)):
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

    tol = cfg.phonon.imag_tol_meV
    res = {
        "modes": modes.to_dict(max_points=6000),
        "imag_fraction": modes.imag_fraction(tol),
        "max_imag_meV": modes.max_imag_meV(),
        "imag_tol_meV": tol,
        "zpe_eV_per_atom": modes.zpe(),
        "n_atoms": len(atoms),
        "calculator": calculator_label(phase.material),
        **{k: v for k, v in info.items() if k != "band"},
        "band": info.get("band"),
    }
    write_json(out, res, indent=None)
    try:
        _plot_ha(modes, info.get("band"), os.path.join(phase.phonon_dir, "phonon.png"), phase.tag)
    except Exception as exc:  # noqa: BLE001
        log.warning("phonon plot failed: %s", exc)
    manifest.write_manifest(phase.phonon_dir, "phonons", phase.material, inputs)
    log.info("  imaginary weight %.3f%% (max %.2f meV), ZPE %.1f meV/atom",
             100 * res["imag_fraction"], res["max_imag_meV"], 1e3 * res["zpe_eV_per_atom"])


def load_ha(phase_dir) -> dict | None:
    """Cached HA result with ``modes`` converted to a ModeSet."""
    phase = PhaseDir.of(phase_dir)
    d = read_json(os.path.join(phase.phonon_dir, "ha.json"))
    if not d:
        return None
    d["modes"] = ModeSet.from_dict(d["modes"])
    return d


# ---------------------------------------------------------------------------
# QHA
# ---------------------------------------------------------------------------

def run_qha(phase_dir, device: str = "cpu", force: bool = False) -> None:
    phase = PhaseDir.of(phase_dir)
    out = os.path.join(phase.phonon_dir, "qha.json")
    inputs = [phase.contcar]
    qc = cfg.qha
    if (not force and os.path.exists(out)
            and manifest.is_fresh(phase.phonon_dir, "qha", phase.material, inputs)):
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
    scales = np.linspace(1 - qc.volume_range, 1 + qc.volume_range, qc.n_volumes)
    calc = build_calculator(device, material=phase.material)
    vols, e_el, mode_dicts = [], [], []
    try:
        for i, s in enumerate(scales):
            a = ref.copy()
            a.set_cell(ref.cell.array * s ** (1 / 3), scale_atoms=True)
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

    T = np.array(cfg.hull.temperatures, float)
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
    manifest.write_manifest(phase.phonon_dir, "qha", phase.material, inputs)
    if not all(inside):
        log.warning("QHA: V_eq leaves the sampled range above T = %.0f K - widen qha.volume_range",
                    T[~np.array(inside)].min())


def load_qha(phase_dir) -> dict | None:
    return read_json(os.path.join(PhaseDir.of(phase_dir).phonon_dir, "qha.json"))
