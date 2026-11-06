"""heatup.compute.quasiharmonic
============================
QUASI-HARMONIC phonons (vocabulary and file meta: ``heatup.core.provenance``): the harmonic fc2 of
``heatup.compute.harmonic`` at several volumes, V_eq(T) from minimising F + PV.  Anharmonicity enters only implicitly,
through the volume dependence of the harmonic frequencies.

Step ``quasiharmonic`` (:func:`run_quasiharmonic`)  ->  ``phonons/quasiharmonic/quasiharmonic.json`` (+ ``quasiharmonic.png``, ``V01/ ... V07/``)
    harmonic phonons at ``quasiharmonic.n_volumes`` volumes **with ions relaxed at each volume**, F(V,T) = E(V) + F_vib(V,T)
    fitted per temperature on the grid ``quasiharmonic.temperatures``; gives G(T), V(T), alpha(T), B_T(T) and the share of
    imaginary weight at every volume.  Read by :func:`heatup.analyze.thermo.load_quasiharmonic`.

``PhonopyQHA`` is not used (the free-energy fit is done here, ions relaxed at each volume).
"""

from __future__ import annotations

import logging
import os

import numpy as np
from ase.optimize import BFGS

from heatup.analyze.thermo import ModeSet
from heatup.compute.calculator import build_calculator, calculator_label, release_calculator
from heatup.compute.harmonic import harmonic_modes
from heatup.core import ledger, provenance
from heatup.core.config import GPA_TO_EV_A3, cfg
from heatup.core.layout import PhaseDir, read_json, read_structure, write_json, write_poscar
from heatup.prepare.export import choose_supercell, quasiharmonic_scales, scaled_cell

log = logging.getLogger("heatup")


def run_quasiharmonic(phase_dir, device: str = "cpu", force: bool = False) -> None:
    phase = PhaseDir.of(phase_dir)
    out = os.path.join(phase.quasiharmonic_dir, "quasiharmonic.json")
    qc = cfg.quasiharmonic
    if not force and os.path.exists(out) and ledger.is_fresh(phase, "quasiharmonic"):
        log.info("[cached] %s/quasiharmonic (up to date, not recomputed)", phase.tag)
        return
    if not phase.has_relaxation():
        raise FileNotFoundError(f"{phase.tag}: converged relaxation required first")
    if qc.n_volumes < 5 or qc.n_volumes % 2 == 0:
        raise ValueError("quasiharmonic.n_volumes must be odd and >= 5")

    os.makedirs(phase.quasiharmonic_dir, exist_ok=True)
    ref = read_structure(phase.contcar)
    n = len(ref)
    V0 = ref.get_volume()
    scales = quasiharmonic_scales()
    sc = choose_supercell(ref)          # chosen once, on the reference cell: the same repetitions at every volume
    log.info("quasi-harmonic %s: supercell %s at all %d volumes", phase.tag, sc, len(scales))
    calc = build_calculator(device, material=phase.material)
    vols, e_el, mode_dicts = [], [], []
    try:
        for i, s in enumerate(scales):
            vdir = os.path.join(phase.quasiharmonic_dir, f"V{i + 1:02d}")
            os.makedirs(vdir, exist_ok=True)
            a = scaled_cell(ref, s)
            a.calc = calc
            if qc.relax_internal:
                ok = BFGS(a, trajectory=os.path.join(vdir, "relax.traj"), logfile=None).run(
                    fmax=cfg.relax.fmax, steps=cfg.relax.max_steps)
                if not ok:
                    log.warning("%s: ions at V/V0=%.3f did not converge in %d steps (fmax %g): that volume carries "
                                "residual forces", phase.tag, s, cfg.relax.max_steps, cfg.relax.fmax)
            e_el.append(a.get_potential_energy() / n)
            vols.append(a.get_volume() / n)
            write_poscar(os.path.join(vdir, "CONTCAR"), a)
            a.calc = None
            modes, _, _ = harmonic_modes(a, calc, save_dir=vdir, sc=sc)
            mode_dicts.append(modes.to_dict(max_points=3000))
            write_json(os.path.join(vdir, "energy.json"), {
                "V_over_V0": float(s), "energy_eV_per_atom": float(e_el[-1]), "volume_A3_per_atom": float(vols[-1]),
                "supercell": list(sc), "imag_fraction": modes.imag_fraction(cfg.harmonic.imag_tol_meV),
                "relaxed_ions": bool(qc.relax_internal), "calculator": calculator_label(phase.material)})
            log.info("  volume %d/%d  V/V0=%.3f  E=%.5f eV/atom  imag=%.2f%%",
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
        "volumes_A3_per_atom": vols, "E_el_eV_per_atom": e_el, "supercell": list(sc),
        "F_vib_eV_per_atom": Fvib, "poly_order": qc.poly_order,
        "pressure_GPa": cfg.relax.pressure_GPa, "V0_ref_A3_per_atom": V0 / n,
        "imag_fraction": [m.imag_fraction(cfg.harmonic.imag_tol_meV) for m in modesets],
        "calculator": calculator_label(phase.material),
        "meta": provenance.stamp("quasiharmonic"),
    }, indent=None)
    try:
        from heatup.analyze.vibrations import plot_quasiharmonic_result
        plot_quasiharmonic_result(read_json(out), os.path.join(phase.quasiharmonic_dir, "quasiharmonic.png"), phase.tag)
    except Exception as exc:  # noqa: BLE001
        log.warning("quasi-harmonic plot failed: %s", exc)
    ledger.record(phase, "quasiharmonic")
    if not all(inside):
        log.warning("quasi-harmonic: V_eq leaves the sampled range above T = %.0f K - widen quasiharmonic.volume_range",
                    T[~np.array(inside)].min())
