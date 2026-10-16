"""screen() contract tests that need no calculator."""
import os

import pytest
from pymatgen.core import Lattice, Structure

from heatup import ScreenOptions, cfg, screen
from heatup.api import GateResult, _dotted, decide
from heatup import phases


def G(name, status, msg=""):
    return GateResult(name, status, msg)


def test_invalid_stage_and_option_names_raise():
    with pytest.raises(ValueError):
        screen("x/y", stage="mid_md")
    with pytest.raises(ValueError):
        screen("x/y", "pre_md", options=ScreenOptions(thresholds={"no_such_threshold": 1}))
    with pytest.raises(ValueError):
        screen("x/y", "pre_md", options=ScreenOptions(config={"md": 1}))


def test_options_map_to_config_keys():
    d = _dotted(ScreenOptions(operating_T=600, mobile_species=["Li"], thresholds={"hull_ok_eV": 0.02},
                              model="m", competing_root="/c", config={"md.n_steps": 5}))
    assert d == {"hull.operating_T": 600, "gates.mobile_species": ["Li"], "gates.hull_ok_eV": 0.02,
                 "calc.model": "m", "phases.competing_root": "/c", "md.n_steps": 5}


def test_options_are_scoped_to_the_call(tmp_path):
    d = tmp_path / "X" / "a"
    d.mkdir(parents=True)                                              # no POSCAR -> early, compute-free return
    before = (cfg.gates.mobile_species, cfg.hull.operating_T, cfg.calc.model, cfg.gates.hull_ok_eV)
    r = screen(d, "pre_md", options=ScreenOptions(operating_T=999, mobile_species=["Li"], model="other",
                                                  thresholds={"hull_ok_eV": 0.5}))
    assert r.verdict == "inconclusive" and "POSCAR" in r.reasons[0] and r.operating_T_K == 999
    assert r.model == "other"
    assert before == (cfg.gates.mobile_species, cfg.hull.operating_T, cfg.calc.model, cfg.gates.hull_ok_eV)


def test_legacy_model_name_is_normalised():
    cfg.calc.model = "mace-mpa-0-medium"
    assert cfg.calc.model_for("anything") == "medium-mpa-0"
    cfg.calc.model_overrides = {"BTO": "/p/BTO.model"}
    assert cfg.calc.model_for("BTO") == "/p/BTO.model"


KW = dict(hull_reject_eV=0.1, accept_warn=False, errors=[])


def test_decide_reject_on_any_fail():
    v, why = decide("post_md", {"a": G("a", "ok"), "b": G("b", "fail", "bad")}, None, True, **KW)
    assert v == "reject" and "b: bad" in why[0]


def test_decide_accept_only_when_all_ok():
    assert decide("post_md", {"a": G("a", "ok"), "b": G("b", "ok")}, None, True, **KW)[0] == "accept"
    assert decide("post_md", {"a": G("a", "ok"), "b": G("b", "warn", "w")}, None, True, **KW)[0] == "inconclusive"
    assert decide("post_md", {"a": G("a", "warn", "w")}, None, True, **{**KW, "accept_warn": True})[0] == "accept"
    assert decide("post_md", {"a": G("a", "ok"), "b": G("b", "missing")}, None, True, **KW)[0] == "inconclusive"
    assert decide("post_md", {"a": G("a", "ok")}, None, True, **{**KW, "errors": ["[ERROR] x"]})[0] == "inconclusive"


def test_decide_fail_beats_errors_and_pre_md_hull_rule():
    assert decide("post_md", {"a": G("a", "fail", "x")}, None, True, **{**KW, "errors": ["e"]})[0] == "reject"
    ok = {"mechanical": G("mechanical", "ok")}
    assert decide("pre_md", ok, 0.05, True, **KW)[0] == "accept"
    v, why = decide("pre_md", ok, 0.12, True, **KW)
    assert v == "reject" and "hull" in why[0]
    assert decide("pre_md", ok, None, False, **{**KW, "errors": ["0 K hull: none"]})[0] == "inconclusive"


class _Doc:
    def __init__(self, mid, a):
        self.structure = Structure(Lattice.cubic(a), ["Cu"], [[0, 0, 0]])
        self.material_id, self.energy_above_hull = mid, 0.0

        class S:
            symbol = "Fm-3m"
        self.symmetry = S()


def test_mp_results_go_to_competing_root_and_query_is_cached(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(phases, "_mp_query", lambda els: calls.append(1) or [_Doc("mp-30", 3.6)])
    root = str(tmp_path / "competing")
    els = frozenset({"Cu"})
    a = phases.fetch_mp_phases(els, root)
    assert [p.phase_dir.path for p in a] == [os.path.join(root, "Cu", "Fm-3m_mp-30")]
    monkeypatch.setattr(phases, "_mp_query", lambda els: pytest.fail("network used despite the index"))
    b = phases.fetch_mp_phases(els, root)
    assert [p.tag for p in b] == [p.tag for p in a] and len(calls) == 1
    cfg.phases.mp_max_per_chemsys = 3                                  # changed query parameters -> re-query
    monkeypatch.setattr(phases, "_mp_query", lambda els: calls.append(1) or [_Doc("mp-30", 3.6)])
    phases.fetch_mp_phases(els, root)
    assert len(calls) == 2


def test_candidates_are_mirrored_never_written(tmp_path):
    from ase.build import bulk
    from ase.io import write
    cand, comp = tmp_path / "input" / "candidates", tmp_path / "competing"
    d = cand / "Cu" / "x"
    d.mkdir(parents=True)
    write(str(d / "POSCAR"), bulk("Cu", "fcc", a=3.61, cubic=True), format="vasp", direct=True)
    cfg.phases.sources = ["candidates", "competing"]
    cfg.phases.candidates_root, cfg.phases.competing_root = str(cand), str(comp)
    before = sorted(str(p) for p in cand.rglob("*"))
    refs = phases.find_reference_phases(["Cu"])
    assert len(refs) == 1 and str(refs[0].phase_dir.path).startswith(str(comp))
    assert sorted(str(p) for p in cand.rglob("*")) == before            # untouched
    # the structure being screened must not become its own reference
    cfg.phases.competing_root = str(tmp_path / "competing2")
    assert phases.find_reference_phases(["Cu"], exclude=[d]) == []


def test_decide_with_no_gates_is_inconclusive():
    v, reasons = decide("pre_md", {}, None, False, hull_reject_eV=0.1, accept_warn=False, errors=[])[:2]
    assert v == "inconclusive"


def test_gate_and_force_options_map_to_config():
    d = _dotted(ScreenOptions(gates=["mechanical"], force=["md"]))
    assert d["gates.enabled"] == ["mechanical"] and d["run.force"] == ["md"]
    assert _dotted(ScreenOptions(force=True))["run.force"] == ["all"]
