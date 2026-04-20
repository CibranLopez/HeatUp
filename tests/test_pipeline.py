"""Integration tests for the stability pipeline."""

import json
import os

import pytest

from thermophasepy.pipeline import run_stability_pipeline, _finalise
from thermophasepy import config


class TestPipelineGating:

    def test_mechanical_fail_stops_pipeline(self, sym_dir_mech_fail):
        report = run_stability_pipeline(
            sym_dir                 = sym_dir_mech_fail,
            operating_T             = 1200.0,
            generate_missing_phases = False,
            save_figure             = False,
        )
        assert report["overall"] == config.STATUS_FAIL
        assert report["stopped_at"] == "mechanical"
        # Vibrational and thermodynamic should be empty dicts (never ran).
        assert report["vibrational"]   == {}
        assert report["thermodynamic"] == {}

    def test_vibrational_fail_stops_pipeline(self, sym_dir_vib_fail):
        report = run_stability_pipeline(
            sym_dir                 = sym_dir_vib_fail,
            operating_T             = 1200.0,
            generate_missing_phases = False,
            save_figure             = False,
        )
        assert report["overall"] == config.STATUS_FAIL
        assert report["stopped_at"] == "vibrational"
        assert report["thermodynamic"] == {}

    def test_warn_does_not_stop_pipeline(self, sym_dir_ok, tmp_path, monkeypatch):
        """A warn in Gate 1 must not stop the pipeline."""
        # Patch assess_mechanical to return warn instead of ok.
        import thermophasepy.pipeline as pipe
        orig = pipe.assess_mechanical_stability

        def mock_mech(sym_dir):
            res = orig(sym_dir)
            res["status"] = config.STATUS_WARN
            return res

        monkeypatch.setattr(pipe, "assess_mechanical_stability", mock_mech)
        report = run_stability_pipeline(
            sym_dir                 = sym_dir_ok,
            operating_T             = 1200.0,
            generate_missing_phases = False,
            save_figure             = False,
        )
        assert report["stopped_at"] is None
        # Vibrational must have run.
        assert report["vibrational"].get("status") is not None

    def test_report_schema(self, sym_dir_mech_fail):
        report = run_stability_pipeline(
            sym_dir                 = sym_dir_mech_fail,
            operating_T             = 1200.0,
            generate_missing_phases = False,
            save_figure             = False,
        )
        for key in ("material", "symmetry", "sym_dir", "operating_T_K",
                    "mechanical", "vibrational", "thermodynamic",
                    "overall", "flags", "stopped_at"):
            assert key in report, f"Missing key: {key}"

    def test_report_cached_on_rerun(self, sym_dir_mech_fail):
        """Second call without force_rerun should return cached JSON."""
        run_stability_pipeline(
            sym_dir                 = sym_dir_mech_fail,
            generate_missing_phases = False,
            save_figure             = False,
        )
        report_path = os.path.join(sym_dir_mech_fail, "stability",
                                   "stability_report.json")
        assert os.path.exists(report_path)

        # Second call should hit the cache.
        report2 = run_stability_pipeline(
            sym_dir                 = sym_dir_mech_fail,
            generate_missing_phases = False,
            save_figure             = False,
            force_rerun             = False,
        )
        assert report2["overall"] == config.STATUS_FAIL

    def test_force_rerun_ignores_cache(self, sym_dir_mech_fail):
        run_stability_pipeline(
            sym_dir                 = sym_dir_mech_fail,
            generate_missing_phases = False,
            save_figure             = False,
        )
        # Corrupt the cached file.
        report_path = os.path.join(sym_dir_mech_fail, "stability",
                                   "stability_report.json")
        with open(report_path, "w") as fh:
            fh.write("not valid json at all")

        # Force rerun should recompute successfully despite corrupt cache.
        report = run_stability_pipeline(
            sym_dir                 = sym_dir_mech_fail,
            generate_missing_phases = False,
            save_figure             = False,
            force_rerun             = True,
        )
        assert "overall" in report


class TestFinalise:

    def test_all_ok(self):
        report = {
            "mechanical"    : {"status": config.STATUS_OK},
            "vibrational"   : {"status": config.STATUS_OK},
            "thermodynamic" : {"status": config.STATUS_OK},
        }
        _finalise(report)
        assert report["overall"] == config.STATUS_OK

    def test_one_warn(self):
        report = {
            "mechanical"    : {"status": config.STATUS_WARN},
            "vibrational"   : {"status": config.STATUS_OK},
            "thermodynamic" : {"status": config.STATUS_OK},
        }
        _finalise(report)
        assert report["overall"] == config.STATUS_WARN

    def test_one_fail(self):
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
