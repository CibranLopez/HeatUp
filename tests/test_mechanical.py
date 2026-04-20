"""Tests for Gate 1: mechanical stability."""

import json
import os

import numpy as np
import pytest

from thermophasepy.mechanical import assess_mechanical_stability
from thermophasepy import config


class TestAssessMechanicalStability:

    def test_missing_elastic_tensor(self, tmp_path):
        result = assess_mechanical_stability(str(tmp_path))
        assert result["status"] == config.STATUS_MISSING
        assert result["available"] is False
        assert result["born_stable"] is None

    def test_born_stable_material(self, sym_dir_ok):
        result = assess_mechanical_stability(sym_dir_ok)
        assert result["available"] is True
        assert result["born_stable"] is True
        assert result["status"] == config.STATUS_OK
        assert result["bulk_modulus_GPa"] > config.MECH_BULK_WARN_GPa
        assert result["shear_modulus_GPa"] > config.MECH_SHEAR_WARN_GPa
        assert result["min_eigenvalue_GPa"] > 0.0

    def test_born_unstable_material(self, sym_dir_mech_fail):
        result = assess_mechanical_stability(sym_dir_mech_fail)
        assert result["available"] is True
        assert result["born_stable"] is False
        assert result["status"] == config.STATUS_FAIL

    def test_returns_tensor_and_eigenvalues(self, sym_dir_ok):
        result = assess_mechanical_stability(sym_dir_ok)
        assert "elastic_tensor_GPa" in result
        assert "eigenvalues_GPa" in result
        assert len(result["eigenvalues_GPa"]) == 6
        C = np.array(result["elastic_tensor_GPa"])
        assert C.shape == (6, 6)

    def test_soft_bulk_modulus_warns(self, tmp_path):
        """B between FAIL and WARN thresholds → warn."""
        sd = tmp_path / "sym"
        sd.mkdir()
        (sd / "POSCAR").write_text("Li\n1.0\n4 0 0\n0 4 0\n0 0 4\nLi\n1\nDirect\n0 0 0\n")
        et_path = sd / "elastic" / "elastic_tensor.json"
        et_path.parent.mkdir()
        C = np.eye(6) * 30.0
        C[0, 0] = C[1, 1] = C[2, 2] = 5.0   # very soft but positive
        et_path.write_text(json.dumps({
            "elastic_tensor_GPa": C.tolist(),
            "derived_moduli": {
                "bulk_modulus_voigt_GPa"  : 5.0,   # between 0 and 10 → warn
                "shear_modulus_voigt_GPa" : 3.0,   # < 5 → warn
                "youngs_modulus_voigt_GPa": 15.0,
                "poissons_ratio_voigt"    : 0.30,
            },
        }))
        result = assess_mechanical_stability(str(sd))
        assert result["status"] == config.STATUS_WARN

    def test_negative_bulk_modulus_fails(self, tmp_path):
        sd = tmp_path / "sym"
        sd.mkdir()
        (sd / "POSCAR").write_text("Li\n1.0\n4 0 0\n0 4 0\n0 0 4\nLi\n1\nDirect\n0 0 0\n")
        et_path = sd / "elastic" / "elastic_tensor.json"
        et_path.parent.mkdir()
        C = np.eye(6) * 10.0
        et_path.write_text(json.dumps({
            "elastic_tensor_GPa": C.tolist(),
            "derived_moduli": {
                "bulk_modulus_voigt_GPa"  : -5.0,   # negative → fail
                "shear_modulus_voigt_GPa" : 10.0,
                "youngs_modulus_voigt_GPa": 25.0,
                "poissons_ratio_voigt"    : 0.30,
            },
        }))
        result = assess_mechanical_stability(str(sd))
        assert result["status"] == config.STATUS_FAIL
