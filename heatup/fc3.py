"""heatup.fc3
=============
Third-order force constants (phono3py) -> anharmonic phonon diagnostics.

Step ``fc3`` (:func:`run_fc3`)  ->  ``phonons/fc3.json``
    * fc3 from finite displacements of a (smaller) supercell, forces from the active calculator with the
      residual force of the perfect supercell subtracted (same convention as the harmonic step);
    * fc2 from the harmonic supercell (``fc3.separate_fc2``) or from the fc3 supercell;
    * single-mode relaxation-time (RTA) lattice thermal conductivity kappa(T) on a q-mesh;
    * phonon linewidths -> ``gamma/nu`` (HWHM over frequency) per temperature: the quantitative statement of
      "how harmonic is this crystal at T".  ``omega*tau = nu / (2 gamma) < 1`` is the Ioffe-Regel limit.

It is enabled with ``cfg.phonon.fc_order = 3`` (the pipeline then runs it for the target after the harmonic
step) or called on its own with ``run_step("fc3", phase_dir)``.  It is a *diagnostic*: it never changes a gate
verdict, it adds ``report["anharmonic"]`` and a flag when ``gamma/nu`` at the operating temperature exceeds
``cfg.fc3.ratio_warn`` (the harmonic F_vib of Gate 3 is then less trustworthy: compare with the VDOS overlay).

Cost: the number of fc3 displacements grows quickly with supercell size and falls with symmetry.  Start with
``cfg.fc3.supercell = (2, 2, 2)`` and ``cfg.fc3.cutoff_pair_A`` (e.g. 4-5 A) and converge from there.
"""

from __future__ import annotations

import logging
import math
import os
import warnings

import numpy as np
from ase import Atoms

from heatup import manifest
from heatup.calculator import build_calculator, calculator_label, release_calculator
from heatup.config import cfg
from heatup.paths import PhaseDir, read_json, write_json
from heatup.structure import read_structure, supercell_reps

log = logging.getLogger("heatup")

#: omega*tau < 1 (Ioffe-Regel) <=> gamma/nu > 1/2
IOFFE_REGEL_RATIO = 0.5


def fc3_supercell(atoms: Atoms) -> tuple[int, int, int]:
    c = cfg.fc3
    if c.supercell is not None:
        return tuple(int(x) for x in c.supercell)  # type: ignore[return-value]
    return supercell_reps(atoms.cell.array, c.min_length_A, c.max_atoms, len(atoms))


def _to_ase(sc) -> Atoms:
    return Atoms(sc.symbols, cell=sc.cell, scaled_positions=sc.scaled_positions, pbc=True)


def _supercell_forces(reference, displaced, calc) -> np.ndarray:
    """Forces minus the perfect-supercell residual.  ``None`` entries (pairs beyond the cutoff) get zero
    forces, which is what phono3py expects for excluded displacements."""
    ref = _to_ase(reference)
    ref.calc = calc
    f0 = ref.get_forces()
    out = []
    for sc in displaced:
        if sc is None:
            out.append(np.zeros_like(f0))
            continue
        a = _to_ase(sc)
        a.calc = calc
        out.append(a.get_forces() - f0)
        a.calc = None
    return np.array(out)


def linewidth_statistics(freq_THz: np.ndarray, gamma_THz: np.ndarray, weights: np.ndarray,
                         min_freq_THz: float) -> dict:
    """Weighted ``gamma/nu`` statistics at one temperature.

    ``freq_THz``, ``gamma_THz``: (n_gridpoints, n_bands); ``weights``: (n_gridpoints,).  Modes below
    ``min_freq_THz`` (the acoustic modes at Gamma, where the ratio is 0/0) are excluded.
    """
    w = np.broadcast_to(np.asarray(weights, float)[:, None], freq_THz.shape)
    keep = freq_THz > min_freq_THz
    if not keep.any():
        return {"gamma_over_nu_mean": None, "frac_ioffe_regel": None, "tau_median_ps": None}
    ratio = gamma_THz[keep] / freq_THz[keep]
    wk = w[keep]
    tau = np.full(ratio.shape, np.inf)
    pos = gamma_THz[keep] > 0
    tau[pos] = 1.0 / (4.0 * math.pi * gamma_THz[keep][pos])          # ps (gamma in THz)
    order = np.argsort(tau)
    cum = np.cumsum(wk[order]) / wk.sum()
    tau_med = float(tau[order][np.searchsorted(cum, 0.5)])
    return {"gamma_over_nu_mean": float(np.sum(wk * ratio) / wk.sum()),
            "frac_ioffe_regel": float(wk[ratio > IOFFE_REGEL_RATIO].sum() / wk.sum()),
            "tau_median_ps": None if not np.isfinite(tau_med) else tau_med}


def _phono3py(atoms: Atoms, sc3, sc2):
    from phono3py import Phono3py
    from phonopy.structure.atoms import PhonopyAtoms
    pa = PhonopyAtoms(symbols=atoms.get_chemical_symbols(), cell=atoms.cell.array,
                      scaled_positions=atoms.get_scaled_positions())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                     # phono3py 4: primitive_matrix default notice
        return Phono3py(pa, supercell_matrix=np.diag(sc3),
                        phonon_supercell_matrix=np.diag(sc2) if sc2 is not None else None,
                        primitive_matrix="auto", symprec=cfg.fc3.symprec)


def compute_fc3(atoms: Atoms, calc) -> dict:
    """Relaxed *atoms* -> kappa(T) and linewidth statistics (plain dict)."""
    c = cfg.fc3
    sc3 = fc3_supercell(atoms)
    sc2 = None
    if c.separate_fc2:
        from heatup.phonons import choose_supercell
        sc2 = choose_supercell(atoms)
        if tuple(sc2) == tuple(sc3):
            sc2 = None
    ph = _phono3py(atoms, sc3, sc2)
    ph.generate_displacements(distance=c.delta_A, cutoff_pair_distance=c.cutoff_pair_A)
    n3 = sum(1 for s in ph.supercells_with_displacements if s is not None)
    log.info("  phono3py: fc3 supercell %s (%d atoms), %d displaced cells%s", sc3, len(atoms) * int(np.prod(sc3)), n3,
             "" if c.cutoff_pair_A is None else f" (pair cutoff {c.cutoff_pair_A} A)")
    ph.forces = _supercell_forces(ph.supercell, ph.supercells_with_displacements, calc)
    n2 = 0
    if sc2 is not None:
        ph.generate_fc2_displacements(distance=cfg.phonon.delta_A)
        n2 = len(ph.phonon_supercells_with_displacements)
        log.info("  phono3py: fc2 supercell %s, %d displaced cells", sc2, n2)
        ph.phonon_forces = _supercell_forces(ph.phonon_supercell, ph.phonon_supercells_with_displacements, calc)
        ph.produce_fc2()
    ph.produce_fc3()

    lengths = np.linalg.norm(np.asarray(ph.primitive.cell), axis=1)
    mesh = [max(1, int(math.ceil(c.kappa_mesh_length_A / L))) for L in lengths]
    ph.mesh_numbers = mesh
    ph.init_phph_interaction()
    temps = [float(t) for t in c.temperatures]
    log.info("  phono3py: RTA thermal conductivity on a %s mesh, T = %s K", mesh, temps)
    ph.run_thermal_conductivity(temperatures=temps)
    k = ph.thermal_conductivity
    kappa = np.asarray(k.kappa)[0]                                   # (nT, 6): xx yy zz yz xz xy
    gamma = np.asarray(k.gamma)[0]                                   # (nT, n_gp, n_band) THz
    freq = np.asarray(k.frequencies)
    weights = np.asarray(k.grid_weights)
    stats = [linewidth_statistics(freq, gamma[i], weights, c.min_freq_THz) for i in range(len(temps))]
    return {
        "fc3_supercell": list(sc3), "fc2_supercell": list(sc2 if sc2 is not None else sc3),
        "n_displacements_fc3": n3, "n_displacements_fc2": n2, "cutoff_pair_A": c.cutoff_pair_A,
        "kappa_mesh": mesh, "temperatures_K": temps,
        "kappa_W_mK": [float(np.mean(row[:3])) for row in kappa],
        "kappa_tensor_W_mK": kappa.tolist(),
        "gamma_over_nu_mean": [s["gamma_over_nu_mean"] for s in stats],
        "frac_ioffe_regel": [s["frac_ioffe_regel"] for s in stats],
        "tau_median_ps": [s["tau_median_ps"] for s in stats],
        "max_frequency_THz": float(freq.max()),
    }


def run_fc3(phase_dir, device: str = "cpu", force: bool = False) -> None:
    phase = PhaseDir.of(phase_dir)
    out = os.path.join(phase.phonon_dir, "fc3.json")
    inputs = [phase.contcar]
    if (not force and os.path.exists(out)
            and manifest.is_fresh(phase.phonon_dir, "fc3", phase.material, inputs)):
        log.info("[cached] %s/fc3 (up to date, not recomputed)", phase.tag)
        return
    if not phase.has_relaxation():
        raise FileNotFoundError(f"{phase.tag}: converged relaxation required first")
    import importlib.util
    if importlib.util.find_spec("phono3py") is None:
        raise ImportError("fc_order = 3 needs phono3py:  pip install phono3py")

    os.makedirs(phase.phonon_dir, exist_ok=True)
    atoms = read_structure(phase.contcar)
    calc = build_calculator(device, material=phase.material)
    try:
        log.info("fc3 / thermal conductivity %s (delta=%g A, %s)", phase.tag, cfg.fc3.delta_A,
                 calculator_label(phase.material))
        res = compute_fc3(atoms, calc)
    finally:
        release_calculator(calc)
    res.update({"n_atoms": len(atoms), "calculator": calculator_label(phase.material)})
    write_json(out, res)
    manifest.write_manifest(phase.phonon_dir, "fc3", phase.material, inputs)
    i = int(np.argmin(np.abs(np.array(res["temperatures_K"]) - cfg.hull.operating_T)))
    log.info("  kappa_RTA(%g K) = %.3g W/mK, <gamma/nu> = %s", res["temperatures_K"][i], res["kappa_W_mK"][i],
             "n/a" if res["gamma_over_nu_mean"][i] is None else f"{res['gamma_over_nu_mean'][i]:.3f}")


def load_fc3(phase_dir) -> dict | None:
    phase = PhaseDir.of(phase_dir)
    return read_json(os.path.join(phase.phonon_dir, "fc3.json"))


def anharmonic_summary(res: dict | None, T: float, ratio_warn: float | None = None) -> dict | None:
    """Values of a cached fc3 result at the temperature closest to *T*, plus a ``flag`` when the linewidths
    say the harmonic picture is strained there.  Pure function (unit-tested)."""
    if not res or not res.get("temperatures_K"):
        return None
    ratio_warn = cfg.fc3.ratio_warn if ratio_warn is None else ratio_warn
    temps = np.asarray(res["temperatures_K"], float)
    i = int(np.argmin(np.abs(temps - T)))
    ratio = res["gamma_over_nu_mean"][i]
    out = {"T_K": float(temps[i]), "kappa_W_mK": res["kappa_W_mK"][i], "gamma_over_nu_mean": ratio,
           "frac_ioffe_regel": res["frac_ioffe_regel"][i], "tau_median_ps": res["tau_median_ps"][i],
           "ratio_warn": ratio_warn, "flag": None}
    if ratio is not None and ratio > ratio_warn:
        out["flag"] = (f"[note] anharmonic: <gamma/nu> = {ratio:.3f} > {ratio_warn} at {temps[i]:g} K "
                       f"({100 * (out['frac_ioffe_regel'] or 0):.1f}% of the weight beyond the Ioffe-Regel limit): "
                       "harmonic F_vib is less reliable here, compare with the VDOS overlay")
    return out
