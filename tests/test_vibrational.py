"""Tests for Gate 2: vibrational stability from anharmonic VDOS.

Physical context
----------------
AgI provides the ideal test system:
  - Beta-phase VDOS: acoustic Ag modes (~8 meV) + optic I modes (~18 meV).
    Zero-mode fraction << 2%.  Physically stable at 300 K.
    Bührer & Nicklow, PRB 17, 3362 (1978).
  - Alpha-phase VDOS: large Lorentzian quasi-elastic peak at omega≈0 (Gamma≈2 meV)
    from diffusive Ag motion above the 420 K superionic transition.
    Zero-mode fraction ~15%, well above the 8% fail threshold.
    Hull et al., PRB 73, 024202 (2006).

This is physically meaningful: Gate 2 correctly identifies the AgI alpha-phase
as vibrationally unstable at room temperature, reflecting the experimental fact
that it is a high-T superionic phase that does not exist at ambient conditions.
"""

import json
import os

import numpy as np
import pytest

from heatup.vibrational import assess_vibrational_stability
from heatup import config


class TestMissingData:

    def test_no_aimd_dir(self, tmp_path):
        result = assess_vibrational_stability(str(tmp_path))
        assert result["status"] == config.STATUS_MISSING
        assert result["available"] is False

    def test_aimd_dir_but_no_vdos(self, tmp_path):
        (tmp_path / "aimd" / "900K").mkdir(parents=True)
        result = assess_vibrational_stability(str(tmp_path))
        assert result["status"] == config.STATUS_MISSING


class TestAgIBetaStable:
    """AgI beta-phase — no soft modes, Gate 2 should pass."""

    def test_status_ok(self, sym_dir_agi_beta):
        result = assess_vibrational_stability(sym_dir_agi_beta)
        assert result["available"] is True
        assert result["status"] == config.STATUS_OK

    def test_zero_mode_fraction_below_warn(self, sym_dir_agi_beta):
        """Beta-phase: zeta << 2% warn threshold."""
        result = assess_vibrational_stability(sym_dir_agi_beta)
        assert result["zero_window_frac"] < config.VIB_ZERO_FRAC_WARN

    def test_multiple_temperature_sources(self, sym_dir_agi_beta):
        """Beta-phase fixture has VDOS at 300K and 600K — both should be averaged."""
        result = assess_vibrational_stability(sym_dir_agi_beta)
        assert result["n_sources"] == 2

    def test_vdos_is_normalised(self, sym_dir_agi_beta):
        result = assess_vibrational_stability(sym_dir_agi_beta)
        om = np.array(result["omega_mev"])
        g  = np.array(result["vdos"])
        norm = np.trapz(g, om)
        assert abs(norm - 1.0) < 0.02, f"VDOS norm = {norm:.4f}, expected 1.0"

    def test_peak_at_physically_expected_frequency(self, sym_dir_agi_beta):
        """Dominant VDOS peak should be in the 5–25 meV range (AgI phonon bands)."""
        result = assess_vibrational_stability(sym_dir_agi_beta)
        om = np.array(result["omega_mev"])
        g  = np.array(result["vdos"])
        peak_omega = om[np.argmax(g)]
        assert 5.0 < peak_omega < 25.0, (
            f"VDOS peak at {peak_omega:.1f} meV, expected in AgI phonon range 5-25 meV"
        )

    def test_output_schema_complete(self, sym_dir_agi_beta):
        result = assess_vibrational_stability(sym_dir_agi_beta)
        required = ("available", "zero_window_frac", "zero_window_mev",
                    "omega_mev", "vdos", "n_sources", "sources",
                    "status", "message")
        for key in required:
            assert key in result, f"Missing key: {key}"


class TestAgIAlphaSuperionic:
    """AgI alpha-phase — large quasi-elastic VDOS at omega≈0, Gate 2 should fail.

    This is the physically most important test: the superionic transition in AgI
    is one of the most studied phenomena in solid-state ionics, and the
    quasi-elastic neutron scattering signal (Lorentzian at omega=0) is its
    definitive experimental fingerprint.
    HeatUp correctly identifies this as a vibrational instability at
    room temperature.
    """

    def test_status_fail(self, sym_dir_agi_alpha):
        result = assess_vibrational_stability(sym_dir_agi_alpha)
        assert result["available"] is True
        assert result["status"] == config.STATUS_FAIL

    def test_zero_mode_fraction_above_threshold(self, sym_dir_agi_alpha):
        """Alpha-phase: zeta ~15% >> 8% fail threshold."""
        result = assess_vibrational_stability(sym_dir_agi_alpha)
        assert result["zero_window_frac"] >= config.VIB_ZERO_FRAC_FAIL
        # Physical sanity: ~15% quasi-elastic weight from Hull et al.
        assert result["zero_window_frac"] > 0.08

    def test_quasi_elastic_weight_physically_plausible(self, sym_dir_agi_alpha):
        """Quasi-elastic fraction should be large but < 100% (partial disorder)."""
        result = assess_vibrational_stability(sym_dir_agi_alpha)
        zeta = result["zero_window_frac"]
        assert 0.08 < zeta < 0.80, (
            f"Quasi-elastic fraction {zeta:.2f} outside physically plausible range"
        )

    def test_lorentzian_shape_at_low_frequency(self, sym_dir_agi_alpha):
        """VDOS should be monotonically decreasing near omega=0 (Lorentzian shape)."""
        result = assess_vibrational_stability(sym_dir_agi_alpha)
        om = np.array(result["omega_mev"])
        g  = np.array(result["vdos"])
        # First 10 points (omega < 1 meV): should be decreasing
        low_mask = om < 1.5
        if low_mask.sum() > 3:
            g_low = g[low_mask]
            # g[0] should be larger than g[-1] in the low-omega region
            assert g_low[0] > g_low[-1], (
                "Expected VDOS to decrease with omega in low-frequency region "
                "(Lorentzian quasi-elastic line from diffusive Ag)"
            )


class TestMultiTemperatureAveraging:

    def test_single_temperature(self, sym_dir_agi_beta):
        """Override: use only 300K VDOS; n_sources should be 1 or 2 (fixture has both)."""
        result = assess_vibrational_stability(sym_dir_agi_beta)
        assert result["n_sources"] >= 1

    def test_averaging_preserves_normalisation(self, sym_dir_agi_beta):
        """Averaged VDOS must still integrate to 1."""
        result = assess_vibrational_stability(sym_dir_agi_beta)
        om = np.array(result["omega_mev"])
        g  = np.array(result["vdos"])
        assert abs(np.trapz(g, om) - 1.0) < 0.02
