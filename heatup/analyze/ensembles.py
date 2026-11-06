"""heatup.analyze.ensembles  --  optional free-energy terms for configurational, orientational and defect disorder (DRAFT)
==============================================================================================================
Each term is a function ``f(phase, T) -> eV/atom`` in the ``GibbsAssembler`` convention (added to H0 + F_vib), reads ONE
file under ``<phase>/disorder/`` and is switched on or off from ``cfg.terms`` (``heatup.core.contributions``).  A phase without
the file contributes 0 (a competitor with no disorder is a legitimate case) - ``contributions.inventory`` lists which
phases carry which term.  The physics lives in ``terms.py`` (pure functions, tested against analytic
limits); this module only reads data and normalises per atom.

SELECTION: ``cfg.terms.configurational`` ("off" | "ideal" | "enumerated" | "monte_carlo"), ``cfg.terms.orientational``
("off" | "rotors") and ``cfg.terms.defects`` ("off" | "pairs"); declared with their files and rules in
``heatup.core.contributions`` (``cfg.terms.show()`` prints them).  ONE configurational model at a time: "ideal",
"enumerated" and "monte_carlo" describe the SAME degrees of freedom.

DOUBLE-COUNTING RULES (enforced by documentation, not by code - read them)
    * orientations of a molecular ion: either as discrete states in ``rotors.json`` (kind "orientations") OR as site
      occupancies / enumerated configurations - never both.  ``orientational_entropy: false`` on a rotor removes the
      orientational entropy k ln(n/sigma) from the rotor term when you count it elsewhere.
    * a hindered rotor replaces ONE librational mode: ``remove_libration`` (default true) subtracts the harmonic free
      energy of that mode, so the harmonic part is not counted twice (HA/QHA/VDOS contain it).
    * vacancies/interstitials that are already site occupancies in ``site_occupancies.json`` must NOT also be defects.
    * a Monte Carlo free energy built from 0 K energies has no configuration-dependent vibrational part: add it through
      the enumerated route (coarse graining), not by adding F_vib twice.
References: see the docstring of each function.
"""
from __future__ import annotations

import logging
import os

import numpy as np

from heatup.analyze import terms
from heatup.core import ledger
from heatup.core.config import cfg
from heatup.core.layout import PhaseDir, read_json, read_structure

log = logging.getLogger("heatup")


def _missing(msg: str):
    from heatup.analyze.thermo import MissingData
    return MissingData(msg)


def _n_atoms(phase: PhaseDir) -> int:
    e = phase.energy() or {}
    return int(e["n_atoms"]) if e.get("n_atoms") else len(read_structure(phase.contcar))


def _disorder(phase: PhaseDir, name: str):
    return read_json(phase.join("disorder", name))


# =================================================================================================================
# configurational: enumerated configurations
# =================================================================================================================

def _base_free_energy(phase: PhaseDir, T: np.ndarray, method: str | None) -> np.ndarray:
    """H0 + F_vib[method] (eV/atom) of a phase, WITHOUT any other term (same vibrational model as the hull)."""
    from heatup.core import contributions as C
    sel = {k: "off" for k in C.TERMS}
    sel["vibrational"] = C.resolve(phase, vibrational=method)["vibrational"]
    r = C.assemble(sel).compute(phase, T)
    if not r.get("available", False):
        raise _missing(f"{phase.tag}: {r.get('message', 'free energy not available')}")
    return np.asarray(r["G_eV_per_atom"], float)


def f_conf_enumerated(phase: PhaseDir, T: np.ndarray) -> np.ndarray:
    """Configurational free energy from an explicit set of configurations of the same cell.

    File ``disorder/configurations.json``::

        {"configurations": [{"phase": "SiGe/sqs_1", "degeneracy": 6}, {"phase": "self", "degeneracy": 2}, ...]}

    ``phase``: ``"<material>/<label>"`` under ``cfg.phases.database_root`` (each is an ordinary phase with its own
    relaxation and F_vib, computed with the SAME method as the hull) or ``"self"``; ``degeneracy``: number of
    symmetry-equivalent copies of that configuration in the cell (from the enumeration).  All configurations must
    have the same number of atoms.  With N atoms per cell and G_s the free energy per atom of configuration s,

        G_ens(T) = -(kT/N) ln sum_s g_s exp(-N G_s(T) / kT),       contribution = G_ens(T) - G_self(T).

    Vibrations enter as the configuration-dependent F_vib,s(T) (coarse graining): van de Walle & Ceder, Rev. Mod.
    Phys. 74, 11 (2002), arXiv:cond-mat/0106490 [verified]; same formalism as ATAT's nonconfigurational entropy,
    van de Walle, Calphad 33, 266 (2009) [verified: title]; vibrational entropy differences of 0.1-0.2 kB/atom are
    comparable to configurational ones (abstract of the RMP).  Limits: only the supplied cell's configurations are
    sampled (finite-cell entropy; no long-range sampling) - use ``monte_carlo`` for that."""
    T = np.atleast_1d(np.asarray(T, float))
    d = _disorder(phase, "configurations.json")
    if not d or not d.get("configurations"):
        return np.zeros_like(T)
    method = None                                  # the hull's vibrational model (cfg.terms.vibrational / as prepared)
    n_ref = _n_atoms(phase)
    g_cfg, deg = [], []
    for c in d["configurations"]:
        ph = phase if c["phase"] == "self" else PhaseDir(os.path.join(cfg.phases.database_root, c["phase"]))
        if _n_atoms(ph) != n_ref:
            raise ValueError(f"{ph.tag}: {_n_atoms(ph)} atoms, the target cell has {n_ref}: configurations must share the cell")
        g_cfg.append(_base_free_energy(ph, T, method))
        deg.append(float(c.get("degeneracy", 1.0)))
    G_ens = terms.coarse_grained_free_energy(np.zeros(len(g_cfg)), n_ref * np.array(g_cfg), T, n_ref, degeneracy=deg)
    return G_ens - _base_free_energy(phase, T, method)


# =================================================================================================================
# configurational: lattice-model Monte Carlo
# =================================================================================================================

def f_conf_monte_carlo(phase: PhaseDir, T: np.ndarray) -> np.ndarray:
    """Configurational free energy from lattice-model Monte Carlo averages (cluster expansion; external code).

    File ``disorder/monte_carlo.json``::

        {"temperatures_K": [300, 600, 900, 1200], "E_mean_eV_per_atom": [...],      # <E>(T) from canonical MC
         "E_infinite_T_eV_per_atom": -1.234, "s_inf_over_kB": 0.693,               # random state: <E>(beta=0), ln(#states)/atom
         "E_reference_eV_per_atom": -1.250,                                        # energy that H0 of THIS phase has in the CE zero
         "E_ground_state_eV_per_atom": -1.250}                                     # optional; default = reference

    Thermodynamic integration over inverse temperature (Gibbs-Helmholtz, same identity as for MD enthalpies):

        beta F(beta) = -s_inf + int_0^beta <E>(beta') d beta',        contribution = F_MC(T) - E_reference

    Lattice-model MC free energies from <E>: van de Walle & Asta, Modelling Simul. Mater. Sci. Eng. 10, 521 (2002)
    [recall: cited in the ATAT literature; check]; cluster expansion + MC packages: icet, smol, CASM, ATAT
    (arXiv:0906.1608 [verified]).  Below the lowest MC temperature the contribution is interpolated linearly from
    its 0 K value (E_ground_state - E_reference); above the highest it raises.  The MC energies carry no
    configuration-dependent vibrations (see the module docstring)."""
    T = np.atleast_1d(np.asarray(T, float))
    d = _disorder(phase, "monte_carlo.json")
    if not d:
        return np.zeros_like(T)
    Tm = np.asarray(d["temperatures_K"], float)
    F = terms.free_energy_by_beta_integration(Tm, d["E_mean_eV_per_atom"], d["E_infinite_T_eV_per_atom"], d["s_inf_over_kB"])
    order = np.argsort(Tm)
    Tm, F = Tm[order], F[order]
    e_ref = float(d["E_reference_eV_per_atom"])
    e_gs = float(d.get("E_ground_state_eV_per_atom", e_ref))
    if T.max() > Tm.max() + 1e-9:
        raise _missing(f"{phase.tag}: Monte Carlo free energy computed up to {Tm.max():g} K, the hull needs {T.max():g} K")
    return np.interp(T, np.concatenate([[0.0], Tm]), np.concatenate([[e_gs], F]) - e_ref)


# =================================================================================================================
# orientational / rotational
# =================================================================================================================

def f_orientational(phase: PhaseDir, T: np.ndarray) -> np.ndarray:
    """Hindered rotors and discrete cation orientations.  File ``disorder/rotors.json``::

        {"rotors": [
          {"name": "CH3 of MA+", "kind": "hindered_rotor", "count_per_cell": 4, "n_fold": 3,
           "inertia_amu_A2": 3.1,            # or "B_meV" (= hbar^2/2I)
           "libration_meV": 18.0,            # libration seen in HA / VDOS (sets the barrier)   -- or "barrier_meV"
           "symmetry_number": 3,             # minima related by permuting identical atoms (CH3: 3)
           "orientational_entropy": true,    # false: you count the orientations elsewhere (F_conf)
           "remove_libration": true},        # subtract the harmonic libration (it is already in F_vib)
          {"name": "MA+ orientation", "kind": "orientations", "count_per_cell": 4,
           "states": [{"E_meV": 0.0, "degeneracy": 1}, {"E_meV": 12.0, "degeneracy": 6}]}   # relative to the structure's own
        ]}

    hindered_rotor:  H = -B d^2/dphi^2 + (V0/2)(1 - cos n phi), exact quantum levels by diagonalisation; contribution
        per rotor = F_hr(T; B, V0, n, sigma) - F_harm(hw_lib)   [-> -kT ln(n/sigma) + anharmonic terms in the deep-well
        limit: the orientational entropy and the anharmonic correction of the libration, nothing else].
        Pitzer & Gwinn, J. Chem. Phys. 10, 428 (1942) [recall; cited in the RSC 2025 hindered-rotor literature];
        low-frequency modes treated as free/hindered rotors: Grimme, Chem. Eur. J. 18, 9955 (2012) [verified via the
        ORCA/Q-Chem manuals].  Model limits: one uniaxial rotor, no rotor-lattice coupling, cosine potential.
    orientations:   per cation  -kT ln sum_s g_s exp(-E_s/kT)  (independent cations: high-T / mean-field limit,
        neglects cation-cation correlations, vibrational differences between orientations ignored).  Jump model of
        MA+ (C3 x C4 reorientations): arXiv:2012.05115; four/eight-fold orientational disorder: Weller et al., Chem.
        Commun. 51, 4180 (2015) [both found in the search; check before quoting].  For correlated cation ordering use
        ``configurations.json`` instead (and switch this term off).
    Normalisation: ``count_per_cell`` rotors per relaxed cell, divided by the atoms in that cell."""
    T = np.atleast_1d(np.asarray(T, float))
    d = _disorder(phase, "rotors.json")
    if not d or not d.get("rotors"):
        return np.zeros_like(T)
    n_atoms = _n_atoms(phase)
    out = np.zeros_like(T)
    for r in d["rotors"]:
        kind = r.get("kind", "hindered_rotor")
        if kind == "hindered_rotor":
            B = float(r["B_meV"]) if "B_meV" in r else terms.rotational_constant_meV(float(r["inertia_amu_A2"]))
            n = int(r["n_fold"])
            if "barrier_meV" in r:
                V0 = float(r["barrier_meV"])
                om = float(r.get("libration_meV", n * np.sqrt(B * V0)))
            elif "libration_meV" in r:
                om = float(r["libration_meV"])
                V0 = terms.barrier_from_libration(om, B, n)
            else:
                raise ValueError(f"rotor {r.get('name')}: give 'libration_meV' or 'barrier_meV'")
            sigma = n if not r.get("orientational_entropy", True) else int(r.get("symmetry_number", 1))
            f = terms.hindered_rotor_free_energy(T, B, V0, n, symmetry_number=sigma)
            if r.get("remove_libration", True):
                f = f - terms.harmonic_oscillator_free_energy(om, T)
        elif kind == "orientations":
            st = r["states"]
            f = terms.independent_state_free_energy([1e-3 * x["E_meV"] for x in st], [x.get("degeneracy", 1) for x in st], T)
        else:
            raise ValueError(f"unknown rotor kind {kind!r}: 'hindered_rotor' | 'orientations'")
        out += float(r["count_per_cell"]) * f / n_atoms
    return out


# =================================================================================================================
# point defects
# =================================================================================================================

def f_defects(phase: PhaseDir, T: np.ndarray) -> np.ndarray:
    """Composition-conserving point-defect pairs in the dilute, non-interacting limit.  ``disorder/defects.json``::

        {"defects": [
          {"name": "Frenkel Li", "kind": "pair",
           "formation_energy_eV": 1.35,                 # of the PAIR, at the relaxed cell
           "formation_entropy_kB": 3.0,                 # optional, T-independent vibrational formation entropy of the pair
           "formation_free_energy_table": {"T_K": [...], "g_eV": [...]},   # optional; replaces the two lines above
           "sites_1_per_cell": 8, "sites_2_per_cell": 16, "degeneracy": 1}
        ]}

    g_f(T) = E_f - T S_f (or the table); F per cell from ``terms.defect_pair_free_energy`` (exact for ideal defects),
    divided by the atoms in the cell.  Equilibrium defect concentration n_d = N_site g exp(-DeltaG_f/kT) [Kim et al.,
    arXiv:2005.01941, Eq. 1 - verified]; formation FREE energies (vibrational, spin, electronic, orientational and
    configurational terms; "imperfections are not 0 K") [Mosquera-Lois et al., Chem. Soc. Rev. 52, 5812 (2023),
    arXiv:2307.10451 - verified]; thermodynamic formalism [Freysoldt et al., Rev. Mod. Phys. 86, 253 (2014) - verified].
    NOT included: single defects that exchange atoms with a reservoir (``kind: "single"`` raises: they change the
    composition and belong on the composition axis of the hull, or in a grand-canonical treatment with chemical
    potentials), defect interactions/clustering, charged defects (Fermi level), supercell finite-size corrections.
    Supply g_f from a calculation that already contains the vibrational formation entropy (e.g. harmonic or the
    anharmonic route) - it is NOT recomputed here."""
    T = np.atleast_1d(np.asarray(T, float))
    d = _disorder(phase, "defects.json")
    if not d or not d.get("defects"):
        return np.zeros_like(T)
    n_atoms = _n_atoms(phase)
    out = np.zeros_like(T)
    for r in d["defects"]:
        kind = r.get("kind", "pair")
        if kind != "pair":
            raise NotImplementedError(
                f"defect {r.get('name')}: kind {kind!r} changes the composition (needs a chemical-potential reservoir); "
                "only composition-conserving pairs ('pair') are supported")
        if "formation_free_energy_table" in r:
            tab = r["formation_free_energy_table"]
            gf = np.interp(T, tab["T_K"], tab["g_eV"])
        else:
            gf = float(r["formation_energy_eV"]) - T * terms.KB_EV * float(r.get("formation_entropy_kB", 0.0))
        out += terms.defect_pair_free_energy(gf, r["sites_1_per_cell"], r["sites_2_per_cell"], T,
                                             float(r.get("degeneracy", 1.0))) / n_atoms
    return out
