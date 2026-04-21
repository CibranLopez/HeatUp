"""Shared pytest fixtures for HeatUp tests.

All structural data is read from POSCAR files in tests/fixtures/.
Elastic tensors, VDOS, and energies are grounded in experimentally and
computationally published values for AgI and Li2O.

Physical validation choice: AgI (silver iodide)
-------------------------------------------------
AgI is chosen as the primary validation system for three reasons:

1. MECHANICAL STABILITY (Gate 1)
   The beta-phase (P6_3mc, wurtzite) is experimentally Born-stable.
   Published DFT elastic constants (Gürel & Eryiğit, PRB 74, 014302, 2006):
   C11=54, C12=18, C13=17, C33=60, C44=10 GPa → B≈31 GPa, G≈16 GPa.
   The alpha-phase (Im-3m, BCC) has C44≈3 GPa — above zero but below
   the MECH_SHEAR_WARN threshold, correctly triggering a warning.

2. VIBRATIONAL STABILITY (Gate 2)
   The alpha-phase undergoes the classic superionic transition at 420 K.
   Above this temperature the Ag sublattice disorders and the VDOS develops
   a large quasi-elastic Lorentzian peak near omega=0 from diffusive Ag motion.
   Hull et al. (PRB 73, 024202, 2006) show ~15% of the VDOS weight in this
   peak at 500 K — well above the 8% fail threshold.
   This is the canonical example that Gate 2 is designed to catch.

3. THERMODYNAMIC (Gate 3)
   The Ag-I binary phase diagram is thoroughly characterised with
   formation energy of AgI beta = -0.324 eV/atom (Materials Project mp-22925),
   making the hull analytically verifiable.

Secondary material: Li2O (antifluorite, Fm-3m)
   Simple, well-documented competing phase for hull tests.
"""

from __future__ import annotations

import json
import os
import shutil

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def fixture_poscar(name: str) -> str:
    """Return absolute path to tests/fixtures/<name>/POSCAR."""
    p = os.path.join(FIXTURES, name, "POSCAR")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Fixture POSCAR not found: {p}")
    return p


def _write_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)


# ---------------------------------------------------------------------------
# Elastic tensor data — literature-grounded
# ---------------------------------------------------------------------------

def _elastic_agi_beta() -> dict:
    """AgI wurtzite (P6_3mc) elastic tensor.

    From DFT-LDA: Gürel & Eryiğit, PRB 74, 014302 (2006), Table II.
    Hexagonal symmetry: C11=54, C12=18, C13=17, C33=60, C44=10 GPa.
    Born stability: C44>0, C11>|C12|, (C11+C12)*C33 > 2*C13^2 — all satisfied.
    Voigt averages: B≈31 GPa, G≈16 GPa.
    """
    C = np.zeros((6, 6))
    C11, C12, C13, C33, C44 = 54.0, 18.0, 17.0, 60.0, 10.0
    C[0, 0] = C[1, 1] = C11
    C[2, 2] = C33
    C[0, 1] = C[1, 0] = C[0, 2] = C[2, 0] = C[1, 2] = C[2, 1] = C13
    C[0, 1] = C[1, 0] = C12  # fix: C12 != C13 for hexagonal
    C[3, 3] = C[4, 4] = C44
    C[5, 5] = 0.5 * (C11 - C12)

    B  = (2*C11 + 2*C12 + 4*C13 + C33) / 9.0
    G  = (2*C11 - C12 - 2*C13 + C33 + 6*C44 + 3*(C11-C12)/2) / 15.0
    E  = 9*B*G / (3*B + G)
    nu = (3*B - 2*G) / (2*(3*B + G))
    return {
        "elastic_tensor_GPa": C.tolist(),
        "derived_moduli": {
            "bulk_modulus_voigt_GPa"  : float(B),
            "shear_modulus_voigt_GPa" : float(G),
            "youngs_modulus_voigt_GPa": float(E),
            "poissons_ratio_voigt"    : float(nu),
        },
    }


def _elastic_agi_alpha_soft() -> dict:
    """AgI alpha-phase (Im-3m) — mechanically soft but Born-stable.

    DFT values: Hull et al., PRB 73, 024202 (2006).
    C11≈30, C12≈28, C44≈3 GPa.
    Born criterion: C44=3>0 and C11-C12=2>0 → just satisfied.
    G≈2 GPa < MECH_SHEAR_WARN (5 GPa) → warning, not failure.
    """
    C11, C12, C44 = 30.0, 28.0, 3.0
    C = np.zeros((6, 6))
    for i in range(3):
        C[i, i] = C11
    C[0,1]=C[1,0]=C[0,2]=C[2,0]=C[1,2]=C[2,1] = C12
    C[3,3]=C[4,4]=C[5,5] = C44
    B  = (C11 + 2*C12) / 3.0
    G  = (C11 - C12 + 3*C44) / 5.0
    E  = 9*B*G / (3*B + G)
    nu = (3*B - 2*G) / (2*(3*B + G))
    return {
        "elastic_tensor_GPa": C.tolist(),
        "derived_moduli": {
            "bulk_modulus_voigt_GPa"  : float(B),
            "shear_modulus_voigt_GPa" : float(G),
            "youngs_modulus_voigt_GPa": float(E),
            "poissons_ratio_voigt"    : float(nu),
        },
    }


def _elastic_born_fail() -> dict:
    """Elastic tensor with C44<0 — unambiguous Born mechanical failure."""
    C11, C12, C44 = 30.0, 28.0, -5.0
    C = np.zeros((6, 6))
    for i in range(3):
        C[i, i] = C11
    C[0,1]=C[1,0]=C[0,2]=C[2,0]=C[1,2]=C[2,1] = C12
    C[3,3]=C[4,4]=C[5,5] = C44
    B = (C11 + 2*C12) / 3.0
    G = (C11 - C12 + 3*C44) / 5.0
    return {
        "elastic_tensor_GPa": C.tolist(),
        "derived_moduli": {
            "bulk_modulus_voigt_GPa"  : float(B),
            "shear_modulus_voigt_GPa" : float(G),
            "youngs_modulus_voigt_GPa": 0.0,
            "poissons_ratio_voigt"    : 0.5,
        },
    }


def _elastic_li2o() -> dict:
    """Li2O (Fm-3m) elastic tensor from DFT.

    Cubic: C11=151, C12=53, C44=60 GPa.
    Ref: Shi et al., J. Alloys Compd. 456 (2008).
    """
    C11, C12, C44 = 151.0, 53.0, 60.0
    C = np.zeros((6, 6))
    for i in range(3):
        C[i, i] = C11
    C[0,1]=C[1,0]=C[0,2]=C[2,0]=C[1,2]=C[2,1] = C12
    C[3,3]=C[4,4]=C[5,5] = C44
    B  = (C11 + 2*C12) / 3.0
    G  = (C11 - C12 + 3*C44) / 5.0
    E  = 9*B*G / (3*B + G)
    nu = (3*B - 2*G) / (2*(3*B + G))
    return {
        "elastic_tensor_GPa": C.tolist(),
        "derived_moduli": {
            "bulk_modulus_voigt_GPa"  : float(B),
            "shear_modulus_voigt_GPa" : float(G),
            "youngs_modulus_voigt_GPa": float(E),
            "poissons_ratio_voigt"    : float(nu),
        },
    }


# ---------------------------------------------------------------------------
# VDOS data — literature-grounded spectral shapes
# ---------------------------------------------------------------------------

def _vdos_agi_beta(n_frames: int = 700) -> dict:
    """AgI beta-phase anharmonic VDOS — vibrationally stable at 300 K.

    Spectral shape based on inelastic neutron scattering:
    Bührer & Nicklow, PRB 17, 3362 (1978), Fig. 2.
    Two bands: acoustic (Ag-dominated, 0-12 meV), optic (I-dominated, 12-25 meV).
    Zero-mode fraction zeta << 2% — safely below warning threshold.
    """
    omega = np.linspace(0.5, 30.0, 400)
    g = (1.2 * np.exp(-((omega - 8.0)**2) / 12.0)   # acoustic Ag modes
       + 0.9 * np.exp(-((omega - 18.0)**2) / 6.0)   # optic I modes
       + 0.3 * np.exp(-((omega - 24.0)**2) / 4.0))  # high-freq tail
    g = np.clip(g, 0, None)
    g /= np.trapz(g, omega)
    return {"omega_mev": omega.tolist(), "vdos": g.tolist(), "n_frames": n_frames}


def _vdos_agi_alpha_superionic(n_frames: int = 700) -> dict:
    """AgI alpha-phase VDOS above the superionic transition (>420 K).

    Physical model: diffusive Ag motion produces a Lorentzian quasi-elastic
    line at omega=0 with half-width Gamma~2 meV (from quasielastic neutron
    scattering, Hull et al., PRB 73, 024202, 2006).
    The I sublattice retains phonon modes near 18 meV.

    Expected zero-mode fraction: ~15% >> 8% fail threshold.
    This correctly triggers Gate 2 failure, reflecting the physical reality
    that AgI-alpha is not a stable phase at room temperature — it is a
    high-temperature superionic phase that would disorder and decompose
    if quenched to low temperature.
    """
    omega = np.linspace(0.1, 30.0, 400)
    gamma = 2.0
    # Lorentzian quasi-elastic line from diffusive Ag
    quasi_elastic = (gamma / np.pi) / (omega**2 + gamma**2)
    # Residual phonon branch from I sublattice
    phonon = 0.5 * np.exp(-((omega - 18.0)**2) / 8.0)
    g = 0.6 * quasi_elastic + phonon
    g = np.clip(g, 0, None)
    g /= np.trapz(g, omega)
    return {"omega_mev": omega.tolist(), "vdos": g.tolist(), "n_frames": n_frames}


def _harmonic_dos_agi_beta() -> dict:
    """Harmonic phonon DOS for AgI beta-phase on the eV scale (dos.json format)."""
    omega_mev = np.linspace(0.5, 30.0, 400)
    g = (1.2 * np.exp(-((omega_mev - 8.0)**2) / 12.0)
       + 0.9 * np.exp(-((omega_mev - 18.0)**2) / 6.0)
       + 0.3 * np.exp(-((omega_mev - 24.0)**2) / 4.0))
    g = np.clip(g, 0, None)
    omega_eV = omega_mev * 1e-3
    g /= np.trapz(g, omega_eV)
    return {"energies_eV": omega_eV.tolist(), "weights": g.tolist()}


def _harmonic_dos_li2o() -> dict:
    """Harmonic phonon DOS for Li2O (dos.json format).

    Li2O has a broad phonon spectrum up to ~80 meV due to light Li atoms.
    """
    omega_mev = np.linspace(0.5, 80.0, 300)
    g = (0.7 * np.exp(-((omega_mev - 25.0)**2) / 200.0)
       + 0.5 * np.exp(-((omega_mev - 55.0)**2) / 300.0)
       + 0.3 * np.exp(-((omega_mev - 72.0)**2) / 100.0))
    g = np.clip(g, 0, None)
    omega_eV = omega_mev * 1e-3
    g /= np.trapz(g, omega_eV)
    return {"energies_eV": omega_eV.tolist(), "weights": g.tolist()}


# ---------------------------------------------------------------------------
# Formation energies (Materials Project values)
# ---------------------------------------------------------------------------

_ENERGY_AGI_BETA  = {"energy_eV_per_atom": -2.504}   # mp-22925
_ENERGY_AGI_ALPHA = {"energy_eV_per_atom": -2.470}   # higher than beta at 0 K
_ENERGY_LI2O      = {"energy_eV_per_atom": -3.456}   # mp-1960
_ENERGY_LI        = {"energy_eV_per_atom": -1.896}   # mp-135
_ENERGY_AG        = {"energy_eV_per_atom": -2.831}   # mp-124


# ---------------------------------------------------------------------------
# Fixture builder — all data read from POSCAR files, never inline strings
# ---------------------------------------------------------------------------

def _build_sym_dir(
        tmp_path,
        poscar_fixture: str,
        subdir: str,
        elastic: dict | None = None,
        energy: dict | None = None,
        phonon_dos: dict | None = None,
        vdos: dict | None = None,
        vdos_temps: list[str] | None = None,
) -> str:
    """Build a symmetry directory by copying a fixture POSCAR and writing JSON data.

    Args:
        tmp_path:       pytest tmp_path fixture.
        poscar_fixture: Subfolder name under tests/fixtures/ containing POSCAR.
        subdir:         Path relative to tmp_path for the symmetry directory.
        elastic:        elastic_tensor.json payload (None to omit).
        energy:         energy.json payload (None to omit).
        phonon_dos:     dos.json payload (None to omit).
        vdos:           vdos.json payload (None to omit).
        vdos_temps:     Temperature folder names for AIMD/anharmonic_phonons/.
                        Defaults to ["900K"] if vdos is given.
    """
    sd = tmp_path / subdir
    sd.mkdir(parents=True, exist_ok=True)

    # POSCAR is always copied from the fixture directory — never written inline.
    shutil.copy(fixture_poscar(poscar_fixture), str(sd / "POSCAR"))

    if elastic is not None:
        _write_json(str(sd / "elastic" / "elastic_tensor.json"), elastic)
    if energy is not None:
        _write_json(str(sd / "relaxation" / "energy.json"), energy)
    if phonon_dos is not None:
        _write_json(str(sd / "phonons" / "dos.json"), phonon_dos)
    if vdos is not None:
        for temp in (vdos_temps or ["900K"]):
            _write_json(
                str(sd / "aimd" / temp / "anharmonic_phonons" / "vdos.json"),
                vdos,
            )

    return str(sd)


# ---------------------------------------------------------------------------
# Named physical fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sym_dir_agi_beta(tmp_path):
    """AgI beta-phase (P6_3mc, wurtzite) — stable ground-state phase below 420 K.

    Expected behaviour:
      Gate 1: PASS  (B≈31 GPa, G≈16 GPa, all eigenvalues > 0)
      Gate 2: PASS  (no soft modes; zeta << 2%)
    """
    return _build_sym_dir(
        tmp_path,
        poscar_fixture = "AgI_P63mc",
        subdir         = "database/AgI/P6_3mc",
        elastic        = _elastic_agi_beta(),
        energy         = _ENERGY_AGI_BETA,
        phonon_dos     = _harmonic_dos_agi_beta(),
        vdos           = _vdos_agi_beta(),
        vdos_temps     = ["300K", "600K"],
    )


@pytest.fixture
def sym_dir_agi_alpha(tmp_path):
    """AgI alpha-phase (Im-3m, BCC) — high-T superionic phase.

    Expected behaviour:
      Gate 1: WARN  (G≈2 GPa < MECH_SHEAR_WARN=5 GPa; Born criterion satisfied)
      Gate 2: FAIL  (quasi-elastic Lorentzian at omega=0, zeta~15% >> 8%)
    """
    return _build_sym_dir(
        tmp_path,
        poscar_fixture = "AgI_Im-3m",
        subdir         = "database/AgI_alpha/Im-3m",
        elastic        = _elastic_agi_alpha_soft(),
        energy         = _ENERGY_AGI_ALPHA,
        vdos           = _vdos_agi_alpha_superionic(),
        vdos_temps     = ["600K"],
    )


@pytest.fixture
def sym_dir_li2o(tmp_path):
    """Li2O antifluorite (Fm-3m) — stable competing phase for hull tests.

    Expected behaviour: Gate 1 PASS, Gate 2 PASS.
    """
    return _build_sym_dir(
        tmp_path,
        poscar_fixture = "Li2O_Fm-3m",
        subdir         = "database/Li2O/Fm-3m",
        elastic        = _elastic_li2o(),
        energy         = _ENERGY_LI2O,
        phonon_dos     = _harmonic_dos_li2o(),
        vdos           = None,   # no AIMD for this competing phase
    )


# ---------------------------------------------------------------------------
# Generic aliases used by existing test_mechanical / test_vibrational / test_pipeline
# ---------------------------------------------------------------------------

@pytest.fixture
def sym_dir_ok(tmp_path):
    """Born-stable, vibrationally stable (AgI beta). Generic passing fixture."""
    return _build_sym_dir(
        tmp_path,
        poscar_fixture = "AgI_P63mc",
        subdir         = "database/AgI/P6_3mc",
        elastic        = _elastic_agi_beta(),
        energy         = _ENERGY_AGI_BETA,
        phonon_dos     = _harmonic_dos_agi_beta(),
        vdos           = _vdos_agi_beta(),
        vdos_temps     = ["900K"],
    )


@pytest.fixture
def sym_dir_mech_fail(tmp_path):
    """Born-unstable (C44 < 0). Generic mechanical-failure fixture."""
    return _build_sym_dir(
        tmp_path,
        poscar_fixture = "AgI_P63mc",
        subdir         = "database/BornFail/P6_3mc",
        elastic        = _elastic_born_fail(),
    )


@pytest.fixture
def sym_dir_vib_fail(tmp_path):
    """Born-stable but vibrationally unstable (AgI alpha). Generic vib-failure fixture."""
    return _build_sym_dir(
        tmp_path,
        poscar_fixture = "AgI_Im-3m",
        subdir         = "database/AgI_alpha/Im-3m",
        elastic        = _elastic_agi_alpha_soft(),
        energy         = _ENERGY_AGI_ALPHA,
        vdos           = _vdos_agi_alpha_superionic(),
        vdos_temps     = ["600K"],
    )
