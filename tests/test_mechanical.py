"""Tests for Gate 1: mechanical stability.

Physical context
----------------
We validate against AgI, for which published DFT elastic constants exist:
  - Beta-phase (P6_3mc, wurtzite): Born-stable, B≈31 GPa, G≈16 GPa.
    Gürel & Eryiğit, PRB 74, 014302 (2006).
  - Alpha-phase (Im-3m, BCC): Born-stable but mechanically soft (G≈2 GPa).
    Hull et al., PRB 73, 024202 (2006).

The Born-fail fixture uses a synthetic C44<0 tensor representing a
hypothetical mechanically unstable phase.
"""

import json
import os

import numpy as np
import pytest

from heatup.mechanical import assess_mechanical_stability
from heatup import config


class TestMissingData:

    def test_missing_elastic_tensor(self, tmp_path):
        """No elastic_tensor.json → status=missing."""
        result = assess_mechanical_stability(str(tmp_path))
        assert result["status"] == config.STATUS_MISSING
        assert result["available"] is False
        assert result["born_stable"] is None


class TestAgIBeta:
    """AgI beta-phase (P6_3mc, wurtzite) — the experimentally stable phase."""

    def test_born_stable(self, sym_dir_agi_beta):
        result = assess_mechanical_stability(sym_dir_agi_beta)
        assert result["available"] is True
        assert result["born_stable"] is True
        assert result["status"] == config.STATUS_OK

    def test_all_eigenvalues_positive(self, sym_dir_agi_beta):
        result = assess_mechanical_stability(sym_dir_agi_beta)
        assert result["min_eigenvalue_GPa"] > 0.0
        assert all(e > 0 for e in result["eigenvalues_GPa"])

    def test_bulk_modulus_reasonable(self, sym_dir_agi_beta):
        """B ≈ 31 GPa for AgI beta — above the warn threshold of 10 GPa."""
        result = assess_mechanical_stability(sym_dir_agi_beta)
        assert result["bulk_modulus_GPa"] > config.MECH_BULK_WARN_GPa
        # Physical sanity: AgI has B in the range 25–40 GPa
        assert 20.0 < result["bulk_modulus_GPa"] < 50.0

    def test_shear_modulus_reasonable(self, sym_dir_agi_beta):
        """G ≈ 16 GPa for AgI beta — above the warn threshold of 5 GPa."""
        result = assess_mechanical_stability(sym_dir_agi_beta)
        assert result["shear_modulus_GPa"] > config.MECH_SHEAR_WARN_GPa

    def test_returns_full_tensor(self, sym_dir_agi_beta):
        result = assess_mechanical_stability(sym_dir_agi_beta)
        assert "elastic_tensor_GPa" in result
        assert "eigenvalues_GPa" in result
        C = np.array(result["elastic_tensor_GPa"])
        assert C.shape == (6, 6)
        assert len(result["eigenvalues_GPa"]) == 6


class TestAgIAlpha:
    """AgI alpha-phase (Im-3m, BCC) — mechanically soft but Born-stable.

    The superionic phase has C44≈3 GPa — above zero (Born criterion satisfied)
    but below MECH_SHEAR_WARN.  Expected: WARN, not FAIL.
    """

    def test_born_stable_but_warns(self, sym_dir_agi_alpha):
        result = assess_mechanical_stability(sym_dir_agi_alpha)
        assert result["available"] is True
        assert result["born_stable"] is True       # C44>0 and C11>C12
        assert result["status"] == config.STATUS_WARN  # G < 5 GPa → warn

    def test_shear_below_warn_threshold(self, sym_dir_agi_alpha):
        """G ≈ 2 GPa for AgI alpha < MECH_SHEAR_WARN = 5 GPa."""
        result = assess_mechanical_stability(sym_dir_agi_alpha)
        assert result["shear_modulus_GPa"] < config.MECH_SHEAR_WARN_GPa

    def test_all_eigenvalues_still_positive(self, sym_dir_agi_alpha):
        """Born criterion satisfied even though G is low."""
        result = assess_mechanical_stability(sym_dir_agi_alpha)
        assert result["min_eigenvalue_GPa"] > 0.0


class TestBornFail:
    """Phase with C44 < 0 — unambiguous Born mechanical failure."""

    def test_born_unstable(self, sym_dir_mech_fail):
        result = assess_mechanical_stability(sym_dir_mech_fail)
        assert result["available"] is True
        assert result["born_stable"] is False
        assert result["status"] == config.STATUS_FAIL

    def test_min_eigenvalue_negative(self, sym_dir_mech_fail):
        result = assess_mechanical_stability(sym_dir_mech_fail)
        assert result["min_eigenvalue_GPa"] < 0.0


class TestBulkModulusThresholds:

    def test_negative_bulk_modulus_fails(self, tmp_path, sym_dir_ok):
        """Inject a modified elastic_tensor.json with B < 0 into a copied dir."""
        import shutil
        dest = str(tmp_path / "negative_B")
        shutil.copytree(sym_dir_ok, dest)
        et_path = os.path.join(dest, "elastic", "elastic_tensor.json")
        with open(et_path) as f:
            data = json.load(f)
        data["derived_moduli"]["bulk_modulus_voigt_GPa"] = -5.0
        with open(et_path, "w") as f:
            json.dump(data, f)
        result = assess_mechanical_stability(dest)
        assert result["status"] == config.STATUS_FAIL

    def test_soft_but_positive_bulk_warns(self, tmp_path, sym_dir_ok):
        """B between 0 and MECH_BULK_WARN → warn."""
        import shutil
        dest = str(tmp_path / "soft_B")
        shutil.copytree(sym_dir_ok, dest)
        et_path = os.path.join(dest, "elastic", "elastic_tensor.json")
        with open(et_path) as f:
            data = json.load(f)
        data["derived_moduli"]["bulk_modulus_voigt_GPa"] = 5.0   # 0 < 5 < 10
        with open(et_path, "w") as f:
            json.dump(data, f)
        result = assess_mechanical_stability(dest)
        assert result["status"] == config.STATUS_WARN
