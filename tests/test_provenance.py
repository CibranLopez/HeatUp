"""Tests for provenance.py and for the meta blocks of the phonon-type result files.   pytest -q test_provenance.py"""
import json
import os

import pytest

from heatup.core import contributions as C, provenance as P


def test_every_kind_is_complete_and_the_vocabulary_is_closed():
    need = {"file", "family", "force_constants", "anharmonicity", "cell", "frequencies", "statistics", "thermal_expansion", "role"}
    for k, d in P.KINDS.items():
        assert need <= set(d), k
        assert d["family"] in ("harmonic", "quasi-harmonic", "anharmonic"), k
        assert d["file"].startswith("phonons/"), k
    assert {k: P.KINDS[k]["family"] for k in P.KINDS} == {
        "ha": "harmonic", "qha": "quasi-harmonic", "fc3": "anharmonic", "anha": "anharmonic", "anha_md": "anharmonic"}
    with pytest.raises(KeyError):
        P.stamp("nonsense")


def test_fc3_is_neither_harmonic_nor_quasi_harmonic():
    f = P.KINDS["fc3"]
    assert f["family"] == "anharmonic" and f["thermal_expansion"] is False
    assert "third order" in f["force_constants"] and "relaxed 0 K cell" in f["cell"]
    assert "not quasi-harmonic" in f["frequencies"] and "DIAGNOSTIC" in f["role"] and "never a gate" in f["role"]
    m = P.stamp("fc3", fc3_supercell=[2, 2, 2])
    assert m["kind"] == "fc3" and m["fc3_supercell"] == [2, 2, 2] and "heatup_version" in m


def test_registry_families_match_the_provenance_vocabulary():
    for m in C.TERMS["vibrational"].models:
        if m.name in ("ZPE", "HA"):
            assert m.family == P.KINDS["ha"]["family"]
        if m.name == "QHA":
            assert m.family == P.KINDS["qha"]["family"]
        if m.name in ("QHA+ANH", "VDOS"):
            assert m.family == "anharmonic"
        for r in m.results:                                      # every result file named by a model is a known kind or an md file
            assert r.startswith("md/") or any(r == d["file"] for d in P.KINDS.values()), (m.name, r)


@pytest.mark.skipif(not os.environ.get("HEATUP_SLOW"), reason="runs phonopy + EMT (about a minute)")
def test_written_files_carry_their_classification(tmp_path):
    pytest.importorskip("phonopy")
    from ase.build import bulk
    import heatup
    from heatup.core.config import cfg
    from heatup.core.layout import PhaseDir, write_poscar
    from heatup.compute.static import run_relaxation
    from heatup.compute.phonons import run_ha, run_qha
    heatup.reset()
    cfg.calc.backend = "emt"
    cfg.qha.temperatures = [0, 300, 600]
    d = tmp_path / "Cu" / "fcc"
    d.mkdir(parents=True)
    write_poscar(str(d / "POSCAR"), bulk("Cu", "fcc", a=3.6))
    ph = PhaseDir(str(d))
    run_relaxation(ph, device="cpu")
    run_ha(ph, device="cpu")
    run_qha(ph, device="cpu")
    ha = json.load(open(d / "phonons" / "ha.json"))["meta"]
    qha = json.load(open(d / "phonons" / "qha.json"))["meta"]
    assert ha["family"] == "harmonic" and ha["thermal_expansion"] is False and "0 K cell" in ha["cell"]
    assert qha["family"] == "quasi-harmonic" and qha["thermal_expansion"] is True and "implicit" in qha["anharmonicity"]
