"""heatup.elastic
==================
Elastic stiffness tensor C (6x6, GPa) by central-difference stress-strain
with **relaxed ions** (step ``elastic``); output ``elastic/elastic.json``.

Voigt order: xx, yy, zz, yz, xz, xy; shear strains are engineering strains.

Changes vs v0.2:

* relaxed-ion constants (clamped ions over-stiffen every structure with
  internal degrees of freedom and can *hide* a Born instability - e.g. a
  ferroelectric/octahedral-tilt softening shows up only after the ions relax);
* Voigt *and* Reuss *and* Hill averages (Voigt alone is an upper bound; the
  Born criterion itself is on C, not on the averages);
* larger default strain (1e-4 was inside the noise floor of a float32 model);
* residual stress is reported so a mis-relaxed cell is visible.
"""

from __future__ import annotations

import logging
import os

import numpy as np
from ase.optimize import BFGS

from heatup import manifest
from heatup.calculator import build_calculator, calculator_label, release_calculator
from heatup.config import cfg
from heatup.constants import EV_A3_TO_GPA
from heatup.paths import PhaseDir, write_json
from heatup.structure import read_structure

log = logging.getLogger("heatup")

VOIGT_LABELS = ["xx", "yy", "zz", "yz", "xz", "xy"]


def voigt_strain(i: int, delta: float) -> np.ndarray:
    e = np.zeros((3, 3))
    if i < 3:
        e[i, i] = delta
    else:
        a, b = {3: (1, 2), 4: (0, 2), 5: (0, 1)}[i]
        e[a, b] = e[b, a] = delta / 2.0      # engineering shear strain = delta
    return e


def strained(atoms, eps: np.ndarray):
    out = atoms.copy()
    out.set_cell(atoms.cell.array @ (np.eye(3) + eps).T, scale_atoms=True)
    return out


def elastic_tensor(atoms, calc, delta: float, relax_ions: bool, fmax: float) -> np.ndarray:
    """Return the symmetrised 6x6 stiffness tensor in GPa."""
    C = np.zeros((6, 6))
    for beta in range(6):
        sig = []
        for sign in (+1.0, -1.0):
            a = strained(atoms, sign * voigt_strain(beta, delta))
            a.calc = calc
            if relax_ions:
                BFGS(a, logfile=None).run(fmax=fmax, steps=300)
            sig.append(a.get_stress(voigt=True))
            a.calc = None
        C[:, beta] = (sig[0] - sig[1]) / (2.0 * delta)
    return C * EV_A3_TO_GPA


def moduli(C: np.ndarray) -> dict:
    """Voigt / Reuss / Hill averages and derived constants from a 6x6 C (GPa)."""
    S = np.linalg.inv(C)
    BV = (C[0, 0] + C[1, 1] + C[2, 2] + 2 * (C[0, 1] + C[0, 2] + C[1, 2])) / 9.0
    GV = (C[0, 0] + C[1, 1] + C[2, 2] - C[0, 1] - C[0, 2] - C[1, 2]
          + 3 * (C[3, 3] + C[4, 4] + C[5, 5])) / 15.0
    BR = 1.0 / (S[0, 0] + S[1, 1] + S[2, 2] + 2 * (S[0, 1] + S[0, 2] + S[1, 2]))
    GR = 15.0 / (4 * (S[0, 0] + S[1, 1] + S[2, 2]) - 4 * (S[0, 1] + S[0, 2] + S[1, 2])
                 + 3 * (S[3, 3] + S[4, 4] + S[5, 5]))
    B, G = 0.5 * (BV + BR), 0.5 * (GV + GR)
    E = 9 * B * G / (3 * B + G)
    nu = (3 * B - 2 * G) / (2 * (3 * B + G))
    return {
        "bulk_voigt_GPa": BV, "bulk_reuss_GPa": BR, "bulk_hill_GPa": B,
        "shear_voigt_GPa": GV, "shear_reuss_GPa": GR, "shear_hill_GPa": G,
        "youngs_hill_GPa": E, "poisson_hill": nu,
        "universal_anisotropy": 5 * GV / GR + BV / BR - 6.0,
    }


def run_elastic(phase_dir, device: str = "cpu", force: bool = False) -> None:
    phase = PhaseDir.of(phase_dir)
    out = phase.elastic_json
    inputs = [phase.contcar]
    if (not force and os.path.exists(out)
            and manifest.is_fresh(phase.elastic_dir, "elastic", phase.material, inputs)):
        log.info("[cached] %s/elastic (up to date, not recomputed)", phase.tag)
        return
    if not phase.has_relaxation():
        raise FileNotFoundError(f"{phase.tag}: converged relaxation required first")

    atoms = read_structure(phase.contcar)
    calc = build_calculator(device, material=phase.material)
    ecfg = cfg.elastic
    try:
        log.info("Elastic tensor %s (strain=±%g, relax_ions=%s)", phase.tag, ecfg.strain, ecfg.relax_ions)
        atoms.calc = calc
        sigma0 = atoms.get_stress(voigt=True) * EV_A3_TO_GPA
        atoms.calc = None
        C = elastic_tensor(atoms, calc, ecfg.strain, ecfg.relax_ions, ecfg.fmax)
        asym = float(np.abs(C - C.T).max())
        C = 0.5 * (C + C.T)
        mod = moduli(C)
        write_json(out, {
            "C_GPa": C.tolist(), "voigt_labels": VOIGT_LABELS,
            "moduli": mod, "asymmetry_GPa": asym, "residual_stress_GPa": sigma0.tolist(),
            "strain": ecfg.strain, "relax_ions": ecfg.relax_ions,
            "calculator": calculator_label(phase.material),
        })
        manifest.write_manifest(phase.elastic_dir, "elastic", phase.material, inputs)
        log.info("  B=%.1f G=%.1f E=%.1f GPa nu=%.3f", mod["bulk_hill_GPa"],
                 mod["shear_hill_GPa"], mod["youngs_hill_GPa"], mod["poisson_hill"])
    finally:
        atoms.calc = None
        release_calculator(calc)
