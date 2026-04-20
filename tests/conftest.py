"""Shared pytest fixtures for thermophasepy tests."""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers to build minimal fake database directories
# ---------------------------------------------------------------------------

def _write_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh)


def _make_elastic_tensor(born_stable: bool = True) -> dict:
    """Return a minimal elastic_tensor.json payload."""
    if born_stable:
        # Cubic-like, all positive eigenvalues.
        C = [
            [300, 100,  80,   0,   0,   0],
            [100, 300,  80,   0,   0,   0],
            [ 80,  80, 320,   0,   0,   0],
            [  0,   0,   0,  80,   0,   0],
            [  0,   0,   0,   0,  80,   0],
            [  0,   0,   0,   0,   0,  80],
        ]
        B, G, E, nu = 160.0, 95.0, 240.0, 0.26
    else:
        # Negative B and min eigenvalue → mechanical fail.
        C = [
            [-10,  50,  50,   0,   0,   0],
            [ 50, -10,  50,   0,   0,   0],
            [ 50,  50, -10,   0,   0,   0],
            [  0,   0,   0,  10,   0,   0],
            [  0,   0,   0,   0,  10,   0],
            [  0,   0,   0,   0,   0,  10],
        ]
        B, G, E, nu = -10.0, 10.0, 25.0, 0.45
    return {
        "elastic_tensor_GPa": C,
        "derived_moduli": {
            "bulk_modulus_voigt_GPa"  : B,
            "shear_modulus_voigt_GPa" : G,
            "youngs_modulus_voigt_GPa": E,
            "poissons_ratio_voigt"    : nu,
        },
    }


def _make_vdos(soft_mode: bool = False) -> dict:
    """Return a minimal vdos.json payload."""
    rng   = np.random.default_rng(42)
    omega = np.linspace(0.5, 80.0, 200)
    if soft_mode:
        # Large spike at ω = 0 → vibrational fail.
        g  = np.exp(-((omega - 30) ** 2) / 200)
        g += 5.0 * np.exp(-(omega ** 2) / 0.2)   # artificial zero-frequency weight
    else:
        g  = np.exp(-((omega - 30) ** 2) / 200)
    norm = np.trapezoid(g, omega)
    g   /= norm
    return {"omega_mev": omega.tolist(), "vdos": g.tolist()}


def _make_dos() -> dict:
    """Return a minimal phonon dos.json payload."""
    omega = np.linspace(0.001, 0.1, 200)
    g     = np.exp(-((omega - 0.05) ** 2) / 0.0005)
    norm  = np.trapezoid(g, omega)
    g    /= norm
    return {"energies_eV": omega.tolist(), "weights": g.tolist()}


def _make_energy() -> dict:
    return {"energy_eV_per_atom": -3.42}


def _make_poscar_content(elements: list[str] | None = None) -> str:
    """Minimal POSCAR for Li₂O (or custom elements)."""
    if elements is None:
        elements = ["Li", "O"]
    counts = " ".join(["2"] * (len(elements) - 1) + ["1"])
    el_line = " ".join(elements)
    return (
        f"{''.join(elements)}\n"
        "1.0\n"
        "4.0 0.0 0.0\n"
        "0.0 4.0 0.0\n"
        "0.0 0.0 4.0\n"
        f"{el_line}\n"
        f"{counts}\n"
        "Direct\n"
        "0.00 0.00 0.00\n"
        "0.50 0.50 0.00\n"
        "0.25 0.25 0.25\n"
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sym_dir_ok(tmp_path):
    """A symmetry directory that should pass all three gates."""
    sd = tmp_path / "database" / "Li2O" / "Fm-3m"

    # POSCAR
    poscar_path = sd / "POSCAR"
    poscar_path.parent.mkdir(parents=True, exist_ok=True)
    poscar_path.write_text(_make_poscar_content(["Li", "O"]))

    # Elastic
    _write_json(str(sd / "elastic" / "elastic_tensor.json"),
                _make_elastic_tensor(born_stable=True))

    # Relaxation energy
    _write_json(str(sd / "relaxation" / "energy.json"), _make_energy())

    # Harmonic phonon DOS
    _write_json(str(sd / "phonons" / "dos.json"), _make_dos())

    # Anharmonic VDOS (stable)
    anh_dir = sd / "aimd" / "900K" / "anharmonic_phonons"
    _write_json(str(anh_dir / "vdos.json"), _make_vdos(soft_mode=False))

    return str(sd)


@pytest.fixture
def sym_dir_mech_fail(tmp_path):
    """A symmetry directory that fails Gate 1 (mechanical)."""
    sd = tmp_path / "database" / "BadMat" / "P1"
    poscar_path = sd / "POSCAR"
    poscar_path.parent.mkdir(parents=True, exist_ok=True)
    poscar_path.write_text(_make_poscar_content(["Li", "O"]))
    _write_json(str(sd / "elastic" / "elastic_tensor.json"),
                _make_elastic_tensor(born_stable=False))
    return str(sd)


@pytest.fixture
def sym_dir_vib_fail(tmp_path):
    """A symmetry directory that passes Gate 1 but fails Gate 2 (vibrational)."""
    sd = tmp_path / "database" / "SoftMat" / "Pnma"
    poscar_path = sd / "POSCAR"
    poscar_path.parent.mkdir(parents=True, exist_ok=True)
    poscar_path.write_text(_make_poscar_content(["Li", "O"]))

    _write_json(str(sd / "elastic" / "elastic_tensor.json"),
                _make_elastic_tensor(born_stable=True))
    _write_json(str(sd / "relaxation" / "energy.json"), _make_energy())

    # Anharmonic VDOS with large zero-frequency weight.
    anh_dir = sd / "aimd" / "900K" / "anharmonic_phonons"
    _write_json(str(anh_dir / "vdos.json"), _make_vdos(soft_mode=True))

    return str(sd)
