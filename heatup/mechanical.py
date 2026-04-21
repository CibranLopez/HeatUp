"""heatup.mechanical
==========================
Gate 1: Born–Huang mechanical stability assessment.

Reads ``<sym_dir>/elastic/elastic_tensor.json`` (written by the MACE
elastic-constant workflow) and evaluates:

1. **Born criterion** — all eigenvalues of the 6×6 Voigt stiffness tensor
   must be positive.  Any negative eigenvalue means the crystal is unstable
   to an affine deformation along that mode.

2. **Bulk modulus** — the Voigt-averaged bulk modulus B must exceed
   ``config.MECH_BULK_WARN_GPa`` (warning) and ``config.MECH_BULK_FAIL_GPa``
   (failure).  A negative B means the crystal expands under hydrostatic
   compression — an unambiguous mechanical failure.

3. **Shear modulus** — the Voigt-averaged shear modulus G must exceed
   ``config.MECH_SHEAR_WARN_GPa``.  Very low G signals near-zero resistance
   to shear deformation, relevant for grain-boundary stability in ceramic
   electrolytes.

Expected JSON schema for ``elastic_tensor.json``::

    {
        "elastic_tensor_GPa": [[...6x6 floats...]],
        "derived_moduli": {
            "bulk_modulus_voigt_GPa":   float,
            "shear_modulus_voigt_GPa":  float,
            "youngs_modulus_voigt_GPa": float,
            "poissons_ratio_voigt":     float
        }
    }
"""

from __future__ import annotations

import json
import os

import numpy as np

from heatup import config


def assess_mechanical_stability(sym_dir: str) -> dict:
    """Evaluate Born mechanical stability from the elastic tensor.

    Args:
        sym_dir: Path to the symmetry directory (``database/<mat>/<sym>/``).

    Returns:
        Dict with keys:

        ``'available'`` (bool)
            True if ``elastic_tensor.json`` was found and read.
        ``'born_stable'`` (bool | None)
            True if all stiffness eigenvalues are positive.
        ``'bulk_modulus_GPa'`` (float | None)
            Voigt-averaged bulk modulus.
        ``'shear_modulus_GPa'`` (float | None)
            Voigt-averaged shear modulus.
        ``'youngs_modulus_GPa'`` (float | None)
            Voigt-averaged Young's modulus.
        ``'poissons_ratio'`` (float | None)
            Voigt-averaged Poisson's ratio.
        ``'min_eigenvalue_GPa'`` (float | None)
            Smallest eigenvalue of C (negative → Born unstable).
        ``'eigenvalues_GPa'`` (list[float] | None)
            All six eigenvalues of C.
        ``'elastic_tensor_GPa'`` (list[list[float]] | None)
            The full 6×6 stiffness tensor.
        ``'status'`` (str)
            ``'ok'`` | ``'warn'`` | ``'fail'`` | ``'missing'``.
        ``'message'`` (str)
            Human-readable explanation.
    """
    et_path = os.path.join(sym_dir, "elastic", "elastic_tensor.json")

    if not os.path.exists(et_path):
        return {
            "available"   : False,
            "born_stable" : None,
            "status"      : config.STATUS_MISSING,
            "message"     : (
                "Elastic tensor not computed — run the elastic-constant "
                "workflow first."
            ),
        }

    with open(et_path) as fh:
        data = json.load(fh)

    C    = np.array(data["elastic_tensor_GPa"])
    mods = data["derived_moduli"]
    B    = float(mods["bulk_modulus_voigt_GPa"])
    G    = float(mods["shear_modulus_voigt_GPa"])
    E    = float(mods["youngs_modulus_voigt_GPa"])
    nu   = float(mods["poissons_ratio_voigt"])

    eigvals = np.linalg.eigvalsh(C)
    min_eig = float(eigvals.min())
    born_ok = bool(min_eig > config.MECH_BORN_EIGENVALUE_FAIL_GPa)

    reasons: list[str] = []
    if not born_ok:
        reasons.append(
            f"Born criterion violated: min eigenvalue = {min_eig:.2f} GPa"
        )
    if B <= config.MECH_BULK_FAIL_GPa:
        reasons.append(
            f"B = {B:.1f} GPa (negative — collapses under compression)"
        )
    elif B < config.MECH_BULK_WARN_GPa:
        reasons.append(f"B = {B:.1f} GPa (very soft)")
    if G < config.MECH_SHEAR_WARN_GPa:
        reasons.append(f"G = {G:.1f} GPa (low shear stiffness)")

    if not born_ok or B <= config.MECH_BULK_FAIL_GPa:
        status = config.STATUS_FAIL
    elif reasons:
        status = config.STATUS_WARN
    else:
        status = config.STATUS_OK

    message = "; ".join(reasons) if reasons else (
        f"Born stable — min eigenvalue {min_eig:.1f} GPa, "
        f"B = {B:.1f} GPa, G = {G:.1f} GPa."
    )

    return {
        "available"          : True,
        "born_stable"        : born_ok,
        "bulk_modulus_GPa"   : B,
        "shear_modulus_GPa"  : G,
        "youngs_modulus_GPa" : E,
        "poissons_ratio"     : nu,
        "min_eigenvalue_GPa" : min_eig,
        "eigenvalues_GPa"    : eigvals.tolist(),
        "elastic_tensor_GPa" : C.tolist(),
        "status"             : status,
        "message"            : message,
    }
