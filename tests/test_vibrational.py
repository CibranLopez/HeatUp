"""Tests of the vibrational levels: naming, instability report, evidence selection, independence of the checks, migration."""
import json
import os

import numpy as np
import pytest

from heatup import cfg, reset
from heatup.analyze import gates, vibrations
from heatup.analyze.thermo import ModeSet
from heatup.core import ledger, provenance
from heatup.core.layout import PhaseDir, write_json


@pytest.fixture(autouse=True)
def _fresh_cfg():
    reset()
    yield
    reset()


def _band(neg_depth=-5.0):
    x = np.linspace(0, 1, 61)
    low = np.where(np.abs(x - 0.5) < 0.1, neg_depth * (1 - np.abs(x - 0.5) / 0.1), 4 * np.sin(np.pi * x))
    f = np.stack([low, 10 + 0 * x, 20 + 0 * x], axis=1)
    kp = np.stack([x * 0.5, x * 0.5, 0 * x], axis=1)
    return {"x": x.tolist(), "special_x": [0.0, 0.5, 1.0], "labels": ["G", "X", "M"], "kpoints": kp.tolist(),
            "frequencies_meV": f.tolist()}


def _write_harmonic(phase_dir, neg_depth=-5.0):
    phase = PhaseDir(phase_dir)
    band = _band(neg_depth)
    freq = np.array([[neg_depth, 10.0, 20.0], [4.0, 10.0, 20.0], [3.0, 10.0, 20.0], [5.0, 11.0, 21.0]])
    modes = ModeSet.from_frequencies(freq, np.ones(4), n_atoms=1, source="harmonic")
    write_json(os.path.join(phase.harmonic_dir, "harmonic.json"), {
        "modes": modes.to_dict(), "imag_fraction": modes.imag_fraction(0.5), "max_imag_meV": modes.max_imag_meV(),
        "imag_tol_meV": 0.5, "band": band, "meta": provenance.stamp("harmonic")})
    return phase


def test_vocabulary_and_steps():
    assert set(provenance.KINDS) == {"harmonic", "quasiharmonic", "anharmonic", "perturbative"}
    assert provenance.KINDS["perturbative"]["role"].startswith("DIAGNOSTIC")
    assert "anharmonic" not in ledger.STEPS                       # the QHA+ANH step is gone
    assert {"harmonic", "quasiharmonic", "perturbative"} <= set(ledger.STEPS)
    assert not {"phonons", "qha", "fc3"} & set(ledger.STEPS)


def test_imaginary_branch_is_located():
    br = vibrations.imaginary_branches(_band(-5.0), tol_meV=0.5)
    assert len(br) == 1 and br[0]["branch"] == 0
    assert br[0]["min_meV"] == pytest.approx(-5.0, abs=0.2)
    assert br[0]["nearest_label"] == "X"
    assert br[0]["q_frac"] == pytest.approx([0.25, 0.25, 0.0], abs=0.03)
    assert 0.1 < br[0]["fraction_of_path"] < 0.4
    assert vibrations.imaginary_branches(_band(+1.0), tol_meV=0.5) == []


def test_harmonic_source_reports_branches(tmp_path):
    phase = _write_harmonic(str(tmp_path / "A" / "x"))
    v = gates.assess_vibrational_stability(phase, sources=["harmonic"], save_figure=False)
    assert v["evidence"] == "harmonic" and v["harmonic"]["n_unstable_branches"] == 1
    assert "near X" in v["message"] and v["status"] in ("warn", "fail")


def test_sources_select_evidence(tmp_path):
    phase = _write_harmonic(str(tmp_path / "A" / "x"))
    v = gates.assess_vibrational_stability(phase, sources=["quasiharmonic"], save_figure=False)
    assert v["status"] == "missing" and v["missing_sources"] == ["quasiharmonic"]
    v = gates.assess_vibrational_stability(phase, sources=["harmonic", "anharmonic"], save_figure=False)
    assert v["evidence"] == "harmonic" and v["missing_sources"] == ["anharmonic"]       # no MD: reported, not fatal
    with pytest.raises(ValueError):
        gates.assess_vibrational_stability(phase, sources=["VDOS"], save_figure=False)


def test_vibrational_figure_is_written(tmp_path):
    pytest.importorskip("matplotlib")
    phase = _write_harmonic(str(tmp_path / "A" / "x"))
    v = gates.assess_vibrational_stability(phase, sources=["harmonic"])
    assert v["figure"] and os.path.exists(v["figure"])


def test_checks_are_independent(tmp_path):
    phase = PhaseDir(str(tmp_path / "A" / "x"))
    os.makedirs(phase.path)
    # no evidence at all: every check answers "missing" by itself, none raises or needs another
    assert gates.assess_mechanical_stability(phase)["status"] == "missing"
    assert gates.assess_structural_transition(phase)["status"] == "missing"
    assert gates.assess_vibrational_stability(phase)["status"] == "missing"
    # vibrational alone does not need elastic or md steps
    cfg.vibrational.sources = ["harmonic"]
    assert ledger.gate_steps("vibrational") == ("harmonic",)
    assert ledger.steps_for(["mechanical"]) == ["relax", "elastic"]
    assert ledger.steps_for(["structural_transition"]) == ["relax", "md"]


def test_terms_names_and_legacy_aliases():
    from heatup.core import contributions
    cfg.terms.vibrational = "QHA"                                 # old name still read
    assert contributions.resolve()["vibrational"] == "quasiharmonic"
    cfg.terms.vibrational = "VDOS"
    assert contributions.resolve()["vibrational"] == "anharmonic"


def test_every_level_keeps_its_files_in_its_own_phonons_folder(tmp_path):
    p = PhaseDir(str(tmp_path / "M" / "a"))
    rel = lambda d: os.path.relpath(d, p.path)
    assert rel(p.harmonic_dir) == os.path.join("phonons", "harmonic")
    assert rel(p.quasiharmonic_dir) == os.path.join("phonons", "quasiharmonic")
    assert rel(p.perturbative_dir) == os.path.join("phonons", "perturbative")
    assert rel(p.anharmonic_run(300)) == os.path.join("phonons", "anharmonic", "300K")
    assert rel(ledger.STEPS["vdos"].where(p, 300)) == os.path.join("phonons", "anharmonic", "300K")
    assert rel(ledger.STEPS["harmonic"].where(p, None)) == os.path.join("phonons", "harmonic")
    assert rel(ledger.STEPS["mdstats"].where(p, 300)) == os.path.join("md", "300K")      # not a phonon result
    assert ledger.STEPS["mdstats"].outputs == ("mdstats.json",)


def test_anharmonic_folder_says_where_it_comes_from(tmp_path):
    from heatup.analyze.anharmonic import write_source_marker
    p = PhaseDir(str(tmp_path / "M" / "a"))
    path = write_source_marker(p, [300, 600])
    d = json.loads(open(path).read())
    assert d["calculation"] == "none" and d["source_folder"] == "md"
    assert d["runs"] == {"300": "md/300K/output.traj", "600": "md/600K/output.traj"}
    assert path == os.path.join(p.path, "phonons", "anharmonic", "source.json")


def _old_layout(ph):
    (ph / "phonons" / "ha").mkdir(parents=True)
    (ph / "phonons" / "qha" / "V01").mkdir(parents=True)
    (ph / "md" / "300K").mkdir(parents=True)
    (ph / "POSCAR").write_text("x")
    (ph / "phonons" / "ha.json").write_text(json.dumps({"modes": {}, "meta": {"kind": "ha"}}))
    (ph / "phonons" / "ha" / "forces.npy").write_bytes(b"0")
    (ph / "phonons" / "qha.json").write_text("{}")
    (ph / "phonons" / "anharmonic.json").write_text("{}")
    (ph / "md" / "300K" / "vdos.json").write_text("{}")
    (ph / "md" / "300K" / "vdos.png").write_bytes(b"png")
    (ph / "md" / "300K" / "mdstats.json").write_text("{}")
    (ph / "phonons" / "step_phonons.json").write_text(json.dumps(
        {"step": "phonons", "task": {}, "export": {}, "result": {"fingerprint": {"phonon.delta_A": 0.01, "phonon.fc_order": 2}}}))
    (ph / "plan.json").write_text(json.dumps(
        {"role": "target", "steps": ["relax", "phonons", "qha", "anharmonic"],
         "options": {"phonon.delta_A": 0.01, "phonon.fc_order": 2, "qha.n_volumes": 7, "terms.vibrational": "QHA",
                     "hull.phonon_method": "QHA", "terms.anharmonic": "off"}}))


def _check_migrated(ph):
    ph_ = ph / "phonons"
    assert (ph_ / "harmonic" / "harmonic.json").exists() and (ph_ / "harmonic" / "forces.npy").exists()
    assert (ph_ / "quasiharmonic" / "quasiharmonic.json").exists()
    assert (ph_ / "anharmonic" / "300K" / "vdos.json").exists() and (ph_ / "anharmonic" / "300K" / "vdos.png").exists()
    assert (ph_ / "anharmonic" / "source.json").exists()                     # says the phonons come from md/
    assert not (ph / "md" / "300K" / "vdos.json").exists() and (ph / "md" / "300K" / "mdstats.json").exists()
    assert (ph_ / "anharmonic.json").exists()                                # retired results are never deleted
    plan = json.loads((ph / "plan.json").read_text())
    assert plan["steps"] == ["relax", "harmonic", "quasiharmonic"]
    assert plan["options"] == {"harmonic.delta_A": 0.01, "quasiharmonic.n_volumes": 7, "terms.vibrational": "quasiharmonic"}
    rec = json.loads((ph_ / "harmonic" / "step_harmonic.json").read_text())
    assert rec["step"] == "harmonic" and rec["result"]["fingerprint"] == {"harmonic.delta_A": 0.01}
    assert "task" not in rec and "export" not in rec
    assert json.loads((ph_ / "harmonic" / "harmonic.json").read_text())["meta"]["kind"] == "harmonic"


def test_migration_of_the_original_layout(tmp_path):
    from heatup.core.migrate import migrate_database
    root = tmp_path / "db"
    ph = root / "M" / "a"
    _old_layout(ph)
    dry = migrate_database(str(root), dry_run=True)
    assert dry and (ph / "phonons" / "ha.json").exists()                     # a dry run changes nothing
    migrate_database(str(root))
    _check_migrated(ph)
    assert migrate_database(str(root)) == {}                                 # second run: nothing left to do


def test_migration_of_the_050_draft_layout(tmp_path):
    from heatup.core.migrate import migrate_database
    root = tmp_path / "db"
    ph = root / "M" / "a"
    (ph / "harmonic").mkdir(parents=True)
    (ph / "quasiharmonic").mkdir()
    (ph / "md" / "300K").mkdir(parents=True)
    (ph / "POSCAR").write_text("x")
    (ph / "harmonic" / "harmonic.json").write_text(json.dumps({"modes": {}}))
    (ph / "harmonic" / "forces.npy").write_bytes(b"0")
    (ph / "quasiharmonic" / "quasiharmonic.json").write_text("{}")
    (ph / "md" / "anharmonic.json").write_text(json.dumps({"runs": []}))
    (ph / "md" / "300K" / "vdos.json").write_text("{}")
    (ph / "md" / "300K" / "mdstats.json").write_text("{}")
    migrate_database(str(root))
    assert (ph / "phonons" / "harmonic" / "harmonic.json").exists() and not (ph / "harmonic").exists()
    assert (ph / "phonons" / "quasiharmonic" / "quasiharmonic.json").exists()
    assert (ph / "phonons" / "anharmonic" / "anharmonic.json").exists() and (ph / "phonons" / "anharmonic" / "300K" / "vdos.json").exists()
    assert not (ph / "md" / "anharmonic.json").exists()
