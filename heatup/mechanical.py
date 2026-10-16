"""heatup.mechanical
=====================
Gate 1 - Born-Huang mechanical stability.

The necessary and sufficient condition for a crystal at (near) zero stress is
a positive-definite 6x6 stiffness matrix (Mouhat & Coudert, PRB 90, 224104,
2014): all eigenvalues of C must be > 0.  Additionally the Hill-averaged bulk
and shear moduli are checked against soft-material thresholds.
"""

from __future__ import annotations

import numpy as np

from heatup.config import (STATUS_FAIL, STATUS_MISSING, STATUS_OK, STATUS_WARN, cfg)
from heatup.paths import PhaseDir, read_json


def assess_mechanical_stability(phase_dir) -> dict:
    phase = PhaseDir.of(phase_dir)
    data = read_json(phase.elastic_json)
    if not data:
        return {"available": False, "born_stable": None, "status": STATUS_MISSING,
                "message": "Elastic tensor not computed (run step 'elastic')."}

    g = cfg.gates
    C = np.array(data["C_GPa"])
    mod = data["moduli"]
    B, G = mod["bulk_hill_GPa"], mod["shear_hill_GPa"]
    eig = np.linalg.eigvalsh(C)
    min_eig = float(eig.min())
    born = bool(min_eig > g.born_min_eigenvalue_GPa)

    reasons: list[str] = []
    if not born:
        reasons.append(f"Born criterion violated (min eigenvalue {min_eig:.2f} GPa)")
    if B <= g.bulk_fail_GPa:
        reasons.append(f"B = {B:.1f} GPa (<= {g.bulk_fail_GPa:g})")
    elif B < g.bulk_warn_GPa:
        reasons.append(f"B = {B:.1f} GPa (very soft)")
    if G < g.shear_warn_GPa and born:
        reasons.append(f"G = {G:.1f} GPa (low shear stiffness)")
    resid = float(np.abs(data.get("residual_stress_GPa", [0] * 6)).max())
    if resid > 0.2:
        reasons.append(f"residual stress {resid:.2f} GPa - cell not at equilibrium")

    if not born or B <= g.bulk_fail_GPa:
        status = STATUS_FAIL
    elif reasons:
        status = STATUS_WARN
    else:
        status = STATUS_OK
    msg = "; ".join(reasons) or (
        f"Born stable: min eigenvalue {min_eig:.1f} GPa, B={B:.1f}, G={G:.1f} GPa (Hill).")
    return {
        "available": True, "born_stable": born, "min_eigenvalue_GPa": min_eig,
        "eigenvalues_GPa": eig.tolist(), "C_GPa": C.tolist(),
        "bulk_GPa": B, "shear_GPa": G, "youngs_GPa": mod["youngs_hill_GPa"],
        "poisson": mod["poisson_hill"], "moduli": mod,
        "status": status, "message": msg,
    }
