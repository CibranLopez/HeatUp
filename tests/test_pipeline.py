"""Integration tests for the HeatUp stability pipeline.

Physical scenario
-----------------
AgI beta-phase (P6_3mc):   Gate 1 PASS, Gate 2 PASS → pipeline continues.
AgI alpha-phase (Im-3m):   Gate 1 WARN, Gate 2 FAIL → pipeline stops at Gate 2.
Born-fail phase:            Gate 1 FAIL              → pipeline stops at Gate 1.

These match the known physics of the AgI system.
"""

import json
import os

import pytest

from heatup.pipeline import run_stability_pipeline, _finalise
from heatup import config


class TestAgIBetaPassesBothGates:

    def test_mechanical_gate_passes(self, sym_dir_agi_beta):
        report = run_stability_pipeline(
            sym_dir                 = sym_dir_agi_beta,
            generate_missing_phases = False,
            save_figure             = False,
        )
        assert report["mechanical"]["status"] == config.STATUS_OK

    def test_vibrational_gate_passes(self, sym_dir_agi_beta):
        report = run_stability_pipeline(
            sym_dir                 = sym_dir_agi_beta,
            generate_missing_phases = False,
            save_figure             = False,
        )
        assert report["vibrational"]["status"] == config.STATUS_OK

    def test_pipeline_does_not_stop_early(self, sym_dir_agi_beta):
        """Beta-phase should reach Gate 3 (or get missing if no competing phases)."""
        report = run_stability_pipeline(
            sym_dir                 = sym_dir_agi_beta,
            generate_missing_phases = False,
            save_figure             = False,
        )
        assert report["stopped_at"] is None

    def test_overall_not_fail(self, sym_dir_agi_beta):
        report = run_stability_pipeline(
            sym_dir                 = sym_dir_agi_beta,
            generate_missing_phases = False,
            save_figure             = False,
        )
        assert report["overall"] != config.STATUS_FAIL


class TestAgIAlphaStopsAtVibrational:
    """Alpha-phase: mechanically soft (WARN) but vibrationally unstable (FAIL)."""

    def test_mechanical_warns(self, sym_dir_agi_alpha):
        report = run_stability_pipeline(
            sym_dir                 = sym_dir_agi_alpha,
            generate_missing_phases = False,
            save_figure             = False,
        )
        assert report["mechanical"]["status"] == config.STATUS_WARN

    def test_vibrational_fails(self, sym_dir_agi_alpha):
        report = run_stability_pipeline(
            sym_dir                 = sym_dir_agi_alpha,
            generate_missing_phases = False,
            save_figure             = False,
        )
        assert report["vibrational"]["status"] == config.STATUS_FAIL

    def test_pipeline_stops_at_vibrational(self, sym_dir_agi_alpha):
        report = run_stability_pipeline(
            sym_dir                 = sym_dir_agi_alpha,
            generate_missing_phases = False,
            save_figure             = False,
        )
        assert report["stopped_at"] == "vibrational"

    def test_thermodynamic_not_evaluated(self, sym_dir_agi_alpha):
        """Gate 3 should never run if Gate 2 fails."""
        report = run_stability_pipeline(
            sym_dir                 = sym_dir_agi_alpha,
            generate_missing_phases = False,
            save_figure             = False,
        )
        assert report["thermodynamic"] == {}

    def test_overall_fail(self, sym_dir_agi_alpha):
        report = run_stability_pipeline(
            sym_dir                 = sym_dir_agi_alpha,
            generate_missing_phases = False,
            save_figure             = False,
        )
        assert report["overall"] == config.STATUS_FAIL


class TestBornFailStopsAtMechanical:

    def test_stops_at_mechanical(self, sym_dir_mech_fail):
        report = run_stability_pipeline(
            sym_dir                 = sym_dir_mech_fail,
            generate_missing_phases = False,
            save_figure             = False,
        )
        assert report["stopped_at"] == "mechanical"
        assert report["vibrational"] == {}
        assert report["thermodynamic"] == {}
        assert report["overall"] == config.STATUS_FAIL

    def test_flags_contain_mechanical_message(self, sym_dir_mech_fail):
        report = run_stability_pipeline(
            sym_dir                 = sym_dir_mech_fail,
            generate_missing_phases = False,
            save_figure             = False,
        )
        assert any("mechanical" in f.lower() or "FAIL" in f
                   for f in report["flags"])


class TestWarnDoesNotStopPipeline:
    """A Gate 1 WARN must not terminate the pipeline (only FAIL does)."""

    def test_mechanical_warn_continues_to_vibrational(self,
                                                        sym_dir_agi_alpha,
                                                        monkeypatch):
        """Alpha-phase: Gate 1 is WARN, Gate 2 runs and fails."""
        report = run_stability_pipeline(
            sym_dir                 = sym_dir_agi_alpha,
            generate_missing_phases = False,
            save_figure             = False,
        )
        # Gate 1 = WARN → Gate 2 was evaluated (not empty)
        assert report["vibrational"] != {}
        # Pipeline stopped at Gate 2 (FAIL), not Gate 1
        assert report["stopped_at"] == "vibrational"


class TestReportSchema:

    def test_all_keys_present(self, sym_dir_mech_fail):
        report = run_stability_pipeline(
            sym_dir                 = sym_dir_mech_fail,
            generate_missing_phases = False,
            save_figure             = False,
        )
        required = ("material", "symmetry", "sym_dir", "operating_T_K",
                    "mechanical", "vibrational", "thermodynamic",
                    "overall", "flags", "stopped_at")
        for key in required:
            assert key in report, f"Missing key: {key}"

    def test_material_and_symmetry_extracted_correctly(self, sym_dir_agi_beta):
        report = run_stability_pipeline(
            sym_dir                 = sym_dir_agi_beta,
            generate_missing_phases = False,
            save_figure             = False,
        )
        assert report["material"] == "AgI"
        assert report["symmetry"] == "P6_3mc"


class TestCaching:

    def test_report_written_to_disk(self, sym_dir_mech_fail):
        run_stability_pipeline(
            sym_dir                 = sym_dir_mech_fail,
            generate_missing_phases = False,
            save_figure             = False,
        )
        report_path = os.path.join(sym_dir_mech_fail, "stability",
                                   "stability_report.json")
        assert os.path.exists(report_path)

    def test_cached_report_loaded_on_second_call(self, sym_dir_mech_fail):
        run_stability_pipeline(
            sym_dir                 = sym_dir_mech_fail,
            generate_missing_phases = False,
            save_figure             = False,
        )
        # Corrupt the elastic tensor so a recompute would give a different result.
        et = os.path.join(sym_dir_mech_fail, "elastic", "elastic_tensor.json")
        with open(et, "w") as f:
            f.write("{}")
        # Second call without force_rerun: should load cache, not recompute.
        report2 = run_stability_pipeline(
            sym_dir                 = sym_dir_mech_fail,
            generate_missing_phases = False,
            save_figure             = False,
            force_rerun             = False,
        )
        # Cache was used: verdict unchanged despite corrupt elastic tensor.
        assert report2["overall"] == config.STATUS_FAIL

    def test_force_rerun_ignores_cache(self, sym_dir_mech_fail):
        run_stability_pipeline(
            sym_dir                 = sym_dir_mech_fail,
            generate_missing_phases = False,
            save_figure             = False,
        )
        report = run_stability_pipeline(
            sym_dir                 = sym_dir_mech_fail,
            generate_missing_phases = False,
            save_figure             = False,
            force_rerun             = True,
        )
        assert "overall" in report


class TestFinalise:

    def test_all_ok_gives_ok(self):
        report = {
            "mechanical"    : {"status": config.STATUS_OK},
            "vibrational"   : {"status": config.STATUS_OK},
            "thermodynamic" : {"status": config.STATUS_OK},
        }
        _finalise(report)
        assert report["overall"] == config.STATUS_OK

    def test_one_warn_gives_warn(self):
        report = {
            "mechanical"    : {"status": config.STATUS_WARN},
            "vibrational"   : {"status": config.STATUS_OK},
            "thermodynamic" : {"status": config.STATUS_OK},
        }
        _finalise(report)
        assert report["overall"] == config.STATUS_WARN

    def test_any_fail_gives_fail(self):
        report = {
            "mechanical"    : {"status": config.STATUS_FAIL},
            "vibrational"   : {"status": config.STATUS_OK},
            "thermodynamic" : {"status": config.STATUS_OK},
        }
        _finalise(report)
        assert report["overall"] == config.STATUS_FAIL

    def test_missing_counts_as_warn(self):
        report = {
            "mechanical"    : {"status": config.STATUS_OK},
            "vibrational"   : {"status": config.STATUS_MISSING},
            "thermodynamic" : {"status": config.STATUS_OK},
        }
        _finalise(report)
        assert report["overall"] == config.STATUS_WARN
