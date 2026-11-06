"""heatup.compute.perturbative
==========================
PERTURBATIVE ANHARMONIC level (vocabulary: ``heatup.core.provenance``): higher-order force constants on top of the harmonic
crystal, treated by perturbation theory.  Implemented: THIRD order (phono3py).  Not implemented: fourth order.

Why fc3 gives no free energy.  The lowest-order anharmonic free energy is of second order in the anharmonic expansion:
a first-order term in the QUARTIC force constants (fc4, positive: the "loop" diagram) plus a second-order term in the CUBIC
ones (fc3, negative: the "bubble").  They are of the same order and partly cancel, so fc3 alone is an unbalanced half of
the correction.  fc3 does give the three-phonon self-energy: linewidths and kappa_RTA, which is what is computed here.
The expansion is also about the harmonic crystal, so it is unreliable with imaginary modes or strong anharmonicity - the
regime the MD level is for.  ``cfg.perturbative.order = 4`` is reserved for an fc4 implementation and raises.

Here fc3 AND fc2 are both evaluated at the relaxed 0 K cell and the frequencies are the harmonic ones, so the temperature
enters only through the phonon occupation numbers and there is no thermal expansion; fc2/fc3 at V(T) is NOT implemented.
The result is a DIAGNOSTIC (gamma/nu, kappa_RTA): never a gate, never a free-energy term.  ``perturbative/perturbative.json``
states this in its ``"meta"`` block.

Step ``perturbative`` (:func:`run_perturbative`)  ->  ``phonons/perturbative/perturbative.json`` (+ ``perturbative.png``)
    * fc3 from finite displacements of a (smaller) supercell, forces from the active calculator with the
      residual force of the perfect supercell subtracted (same convention as the harmonic step);
    * fc2 from the harmonic supercell (``perturbative.separate_fc2``) or from the fc3 supercell;
    * single-mode relaxation-time (RTA) lattice thermal conductivity kappa(T) on a q-mesh;
    * phonon linewidths -> ``gamma/nu`` (HWHM over frequency) per temperature: the quantitative statement of
      "how harmonic is this crystal at T".  ``omega*tau = nu / (2 gamma) < 1`` is the Ioffe-Regel limit.

It is planned like any step: put ``"perturbative"`` in ``cfg.plan.target_steps`` (notebook 1) or call
``run_step("perturbative", phase_dir)``.  Notebook 3 reads its result (:func:`heatup.analyze.thermo.load_perturbative`,
``perturbative_summary``) into ``report["perturbative"]`` and adds a flag when ``gamma/nu`` at the operating temperature
exceeds ``cfg.perturbative.ratio_warn`` (the harmonic F_vib of the hull is then less trustworthy: compare with the
anharmonic overlay).

Cost: the number of fc3 displacements grows quickly with supercell size and falls with symmetry.  Start with
``cfg.perturbative.supercell = (2, 2, 2)`` and ``cfg.perturbative.cutoff_pair_A`` (e.g. 4-5 A) and converge from there.
"""

from __future__ import annotations

import logging
import math
import os

import numpy as np
from ase import Atoms

from heatup.compute.calculator import build_calculator, calculator_label, release_calculator
from heatup.core import ledger, provenance
from heatup.core.config import cfg
from heatup.core.layout import PhaseDir, read_structure, write_json
from heatup.prepare.export import perturbative_cells, to_ase

log = logging.getLogger("heatup")

#: omega*tau < 1 (Ioffe-Regel) <=> gamma/nu > 1/2
IOFFE_REGEL_RATIO = 0.5


def _supercell_forces(reference, displaced, calc, save_as: str | None = None) -> np.ndarray:
    """Forces minus the perfect-supercell residual.  ``None`` entries (pairs beyond the cutoff) get zero
    forces, which is what phono3py expects for excluded displacements.
    ``save_as``: path prefix; writes ``<save_as>.npy`` (what is returned) and ``<save_as>_perfect.npy``."""
    ref = to_ase(reference)
    ref.calc = calc
    f0 = ref.get_forces()
    out = []
    for sc in displaced:
        if sc is None:
            out.append(np.zeros_like(f0))
            continue
        a = to_ase(sc)
        a.calc = calc
        out.append(a.get_forces() - f0)
        a.calc = None
    out = np.array(out)
    if save_as:
        os.makedirs(os.path.dirname(save_as), exist_ok=True)
        np.save(save_as + ".npy", out)
        np.save(save_as + "_perfect.npy", f0)
    return out


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


def compute_fc3(atoms: Atoms, calc, out_dir: str | None = None) -> dict:
    """Relaxed *atoms* -> kappa(T) and linewidth statistics (plain dict).  ``out_dir``: also save the forces there."""
    c = cfg.perturbative
    save = (lambda name: None) if out_dir is None else (lambda name: os.path.join(out_dir, name))
    ph, sc3, sc2 = perturbative_cells(atoms)
    n3 = sum(1 for s in ph.supercells_with_displacements if s is not None)
    log.info("  phono3py: fc3 supercell %s (%d atoms), %d displaced cells%s", sc3, len(atoms) * int(np.prod(sc3)), n3,
             "" if c.cutoff_pair_A is None else f" (pair cutoff {c.cutoff_pair_A} A)")
    ph.forces = _supercell_forces(ph.supercell, ph.supercells_with_displacements, calc, save("forces_fc3"))
    n2 = 0
    if sc2 is not None:
        n2 = len(ph.phonon_supercells_with_displacements)
        log.info("  phono3py: fc2 supercell %s, %d displaced cells", sc2, n2)
        ph.phonon_forces = _supercell_forces(ph.phonon_supercell, ph.phonon_supercells_with_displacements, calc,
                                             save("forces_fc2"))
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


def run_perturbative(phase_dir, device: str = "cpu", force: bool = False) -> None:
    phase = PhaseDir.of(phase_dir)
    out = os.path.join(phase.perturbative_dir, "perturbative.json")
    if not force and os.path.exists(out) and ledger.is_fresh(phase, "perturbative"):
        log.info("[cached] %s/perturbative (up to date, not recomputed)", phase.tag)
        return
    if not phase.has_relaxation():
        raise FileNotFoundError(f"{phase.tag}: converged relaxation required first")
    if cfg.perturbative.order != 3:
        raise NotImplementedError(f"cfg.perturbative.order = {cfg.perturbative.order}: only third order (fc3) is implemented")
    import importlib.util
    if importlib.util.find_spec("phono3py") is None:
        raise ImportError("the fc3 step needs phono3py:  pip install phono3py")

    os.makedirs(phase.perturbative_dir, exist_ok=True)
    atoms = read_structure(phase.contcar)
    calc = build_calculator(device, material=phase.material)
    try:
        log.info("perturbative (fc3) / thermal conductivity %s (delta=%g A, %s)", phase.tag, cfg.perturbative.delta_A,
                 calculator_label(phase.material))
        res = compute_fc3(atoms, calc, out_dir=phase.perturbative_dir)
    finally:
        release_calculator(calc)
    res.update({"n_atoms": len(atoms), "calculator": calculator_label(phase.material),
                "meta": provenance.stamp("perturbative", order=3, fc3_supercell=res["fc3_supercell"], fc2_supercell=res["fc2_supercell"])})
    write_json(out, res)
    try:
        from heatup.analyze.vibrations import plot_perturbative_result
        plot_perturbative_result(res, os.path.join(phase.perturbative_dir, "perturbative.png"), phase.tag)
    except Exception as exc:  # noqa: BLE001
        log.warning("perturbative plot failed: %s", exc)
    ledger.record(phase, "perturbative")
    i = int(np.argmin(np.abs(np.array(res["temperatures_K"]) - ledger.operating_T(phase))))
    log.info("  kappa_RTA(%g K) = %.3g W/mK, <gamma/nu> = %s", res["temperatures_K"][i], res["kappa_W_mK"][i],
             "n/a" if res["gamma_over_nu_mean"][i] is None else f"{res['gamma_over_nu_mean'][i]:.3f}")
