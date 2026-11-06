"""heatup.core.provenance  --  what approximation does a vibrational result represent?  (and stamping it into the files)
=======================================================================================================================
The vocabulary used everywhere (code, registry, config, JSON files).  Four levels, named by what the dynamical matrix is
built from:

HARMONIC               second-order force constants (fc2) at ONE fixed cell (the relaxed 0 K cell).  The frequencies do
                       not depend on temperature and there is no thermal expansion.       -> ``phonons/harmonic/harmonic.json``
QUASI-HARMONIC         the SAME fc2, evaluated at several volumes; the equilibrium volume follows from minimising
                       F(V,T) + PV.  Anharmonicity enters IMPLICITLY, only through the volume dependence of the
                       harmonic frequencies (thermal expansion).               -> ``phonons/quasiharmonic/quasiharmonic.json``
ANHARMONIC             the phonons come from the MD itself: the velocity autocorrelation gives, at every MD temperature,
                       an effective-harmonic spectrum whose frequencies are renormalised by the anharmonicity of the true
                       potential at that temperature and cell (NPT).  The spectra live next to the trajectories
                       (``md/<T>K``); their post-processing lives in ``phonons/anharmonic/`` (``<T>K/vdos.json``,
                       ``<T>K/vdos.png``, ``anharmonic.json``, ``source.json`` says where it comes from).  Effective-
                       harmonic means: the free energy built from them has no explicit anharmonic term (Delta F_AH, see
                       ``analyze.terms``).
PERTURBATIVE           higher-order force constants (fc3, later fc4) on top of the harmonic crystal, treated by
ANHARMONIC             perturbation theory.  Implemented: fc3 -> three-phonon linewidths and kappa_RTA, a DIAGNOSTIC (never
                       a gate, never a free-energy term).  A perturbative free-energy correction needs fc3 AND fc4
                       (same order of the expansion); fc4 is not implemented.            -> ``phonons/perturbative/perturbative.json``

Third-order force constants are therefore NEITHER harmonic NOR quasi-harmonic ("harmonic" means second order) and they are
NOT the "anharmonic" level of this package (which is the MD one).  As computed here (``compute/perturbative.py``) fc3 and
fc2 are both taken at the relaxed 0 K cell and the frequencies are the harmonic ones, so temperature enters only through the
phonon occupation numbers (no thermal expansion).  fc2 AND fc3 at V(T) is possible in principle and is NOT implemented.

Every vibrational JSON carries a ``"meta"`` block built by :func:`stamp`, so a file states by itself what it is.
"""
from __future__ import annotations

KINDS: dict[str, dict] = {
    "harmonic": dict(
        file="phonons/harmonic/harmonic.json", family="harmonic", force_constants="second order (fc2)",
        anharmonicity="none", cell="relaxed 0 K cell (fixed)", frequencies="harmonic, temperature independent",
        statistics="quantum (Bose-Einstein)", thermal_expansion=False,
        role="vibrational free energy (models harmonic, ZPE); imaginary branches for the vibrational check"),
    "quasiharmonic": dict(
        file="phonons/quasiharmonic/quasiharmonic.json", family="quasi-harmonic",
        force_constants="second order (fc2) at every volume",
        anharmonicity="implicit: only the volume dependence of the harmonic frequencies",
        cell="V_eq(T) minimising F(V,T) + PV (ions relaxed at each volume)",
        frequencies="harmonic at each volume", statistics="quantum (Bose-Einstein)", thermal_expansion=True,
        role="vibrational free energy (model quasiharmonic): G(T), V(T), alpha(T), B_T(T)"),
    "anharmonic": dict(
        file="phonons/anharmonic/anharmonic.json", family="anharmonic",
        force_constants="none: effective-harmonic spectrum from the velocity autocorrelation of the MD",
        anharmonicity="effective (temperature-renormalised frequencies); no explicit Delta F_AH",
        cell="NPT cell of the MD run at each temperature",
        frequencies="effective, phonons/anharmonic/<T>K/vdos.json (from the MD run md/<T>K)",
        statistics="classical dynamics, spectra read with the quantum formula", thermal_expansion=True,
        role="SUMMARY of the MD-derived phonons (model anharmonic, experimental) and evidence for the vibrational check; "
             "no calculation of its own: taken from the md/<T>K runs"),
    "perturbative": dict(
        file="phonons/perturbative/perturbative.json", family="perturbative anharmonic",
        force_constants="third order (fc3) with fc2 of the harmonic supercell",
        anharmonicity="explicit, lowest order (three-phonon), perturbative (RTA); NOT harmonic and NOT quasi-harmonic",
        cell="relaxed 0 K cell (fc3 and fc2 both evaluated there)",
        frequencies="harmonic (fc2 at that cell): no thermal expansion, so not quasi-harmonic",
        statistics="quantum (Bose-Einstein occupations in the scattering rates)", thermal_expansion=False,
        role="DIAGNOSTIC (gamma/nu, kappa_RTA): never a gate and never a free-energy term; a free-energy correction "
             "needs fc4 (not implemented)"),
}

#: names used by earlier versions -> the current ones (migration and readers of old files)
LEGACY_KINDS = {"ha": "harmonic", "qha": "quasiharmonic", "anha_md": "anharmonic", "fc3": "perturbative"}


def stamp(kind: str, **extra) -> dict:
    """The ``"meta"`` block of a vibrational result file (see :data:`KINDS`); ``extra`` adds run-specific facts."""
    kind = LEGACY_KINDS.get(kind, kind)
    if kind not in KINDS:
        raise KeyError(f"unknown result kind {kind!r}; known: {sorted(KINDS)}")
    try:
        from heatup import __version__ as version
    except Exception:  # noqa: BLE001
        version = "?"
    return {"kind": kind, **KINDS[kind], "heatup_version": version, **extra}


def describe(kind: str) -> str:
    """One line for logs and notebooks: ``perturbative: perturbative anharmonic; third order ...; relaxed 0 K cell``."""
    k = KINDS[LEGACY_KINDS.get(kind, kind)]
    return f"{kind}: {k['family']}; {k['force_constants']}; {k['cell']}"
