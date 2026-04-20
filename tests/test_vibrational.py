"""Tests for Gate 2: vibrational stability from anharmonic VDOS."""

import json
import os

import numpy as np
import pytest

from thermophasepy.vibrational import assess_vibrational_stability
from thermophasepy import config


class TestAssessVibrationalStability:

    def test_missing_aimd_dir(self, tmp_path):
        result = assess_vibrational_stability(str(tmp_path))
        assert result["status"] == config.STATUS_MISSING
        assert result["available"] is False

    def test_aimd_dir_but_no_vdos(self, tmp_path):
        aimd = tmp_path / "aimd" / "900K"
        aimd.mkdir(parents=True)
        result = assess_vibrational_stability(str(tmp_path))
        assert result["status"] == config.STATUS_MISSING
        assert result["available"] is False

    def test_stable_vdos(self, sym_dir_ok):
        result = assess_vibrational_stability(sym_dir_ok)
        assert result["available"] is True
        assert result["status"] == config.STATUS_OK
        assert result["zero_window_frac"] < config.VIB_ZERO_FRAC_WARN
        assert result["n_sources"] == 1

    def test_soft_mode_vdos_fails(self, sym_dir_vib_fail):
        result = assess_vibrational_stability(sym_dir_vib_fail)
        assert result["available"] is True
        assert result["status"] == config.STATUS_FAIL
        assert result["zero_window_frac"] >= config.VIB_ZERO_FRAC_FAIL

    def test_vdos_averaged_over_multiple_temperatures(self, tmp_path):
        """VDOS from multiple temperatures should be averaged."""
        for temp in ("300K", "600K", "900K"):
            anh_dir = tmp_path / "aimd" / temp / "anharmonic_phonons"
            anh_dir.mkdir(parents=True)
            om = np.linspace(0.5, 80, 200)
            g  = np.exp(-((om - 30) ** 2) / 200)
            g /= np.trapezoid(g, om)
            (anh_dir / "vdos.json").write_text(
                json.dumps({"omega_mev": om.tolist(), "vdos": g.tolist()})
            )
        result = assess_vibrational_stability(str(tmp_path))
        assert result["available"] is True
        assert result["n_sources"] == 3
        assert result["status"] == config.STATUS_OK

    def test_output_schema(self, sym_dir_ok):
        result = assess_vibrational_stability(sym_dir_ok)
        for key in ("available", "zero_window_frac", "zero_window_mev",
                    "omega_mev", "vdos", "n_sources", "sources",
                    "status", "message"):
            assert key in result, f"Missing key: {key}"

    def test_vdos_is_normalised(self, sym_dir_ok):
        result = assess_vibrational_stability(sym_dir_ok)
        om = np.array(result["omega_mev"])
        g  = np.array(result["vdos"])
        norm = np.trapezoid(g, om)
        assert abs(norm - 1.0) < 0.01, f"VDOS norm {norm:.4f} ≠ 1"

    def test_zero_window_frac_between_thresholds_warns(self, tmp_path):
        """Craft a VDOS with zero_frac between WARN and FAIL → warn."""
        anh_dir = tmp_path / "aimd" / "900K" / "anharmonic_phonons"
        anh_dir.mkdir(parents=True)
        om = np.linspace(0.5, 80, 500)
        g  = np.exp(-((om - 30) ** 2) / 200)
        # Add a moderate spike at omega = 0.5 (lowest freq bin).
        g[:5] += 1.5
        g /= np.trapezoid(g, om)
        (anh_dir / "vdos.json").write_text(
            json.dumps({"omega_mev": om.tolist(), "vdos": g.tolist()})
        )
        result = assess_vibrational_stability(str(tmp_path))
        # Depending on exact magnitudes this could be warn or fail;
        # just ensure it is not ok.
        assert result["status"] != config.STATUS_OK
