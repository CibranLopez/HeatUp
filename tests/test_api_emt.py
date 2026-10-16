"""screen() end to end with EMT - every option passed as an argument, the global cfg is never touched."""
import os

import pytest
from ase.build import bulk
from ase.io import write

from heatup import ScreenOptions, cfg, screen

pytestmark = pytest.mark.slow


def put(root, mat, lab, atoms):
    d = os.path.join(root, mat, lab)
    os.makedirs(d, exist_ok=True)
    write(os.path.join(d, "POSCAR"), atoms, format="vasp", direct=True, sort=True)
    return d


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    base = tmp_path_factory.mktemp("w")
    db, cand = str(base / "database"), str(base / "input" / "candidates")
    # references arrive as *candidates* (user-supplied, read-only); targets live in the database
    put(cand, "Ag", "Fm-3m", bulk("Ag", "fcc", a=4.09, cubic=True))
    put(cand, "Au", "Fm-3m", bulk("Au", "fcc", a=4.08, cubic=True))
    put(cand, "Cu", "Fm-3m", bulk("Cu", "fcc", a=3.61, cubic=True))
    c3a = bulk("Cu", "fcc", a=3.75, cubic=True)
    c3a[0].symbol = "Au"
    ag_cu = bulk("Cu", "fcc", a=3.85, cubic=True)
    ag_cu.set_chemical_symbols(["Cu", "Ag", "Ag", "Cu"])
    return {"db": db, "cand": cand, "competing": str(base / "competing"),
            "cu3au": put(db, "Cu3Au", "Pm-3m", c3a), "agcu": put(db, "AgCu", "P4-mmm", ag_cu),
            "bcc": put(db, "Cu_bcc", "Im-3m", bulk("Cu", "bcc", a=2.87, cubic=True))}


def opts(w, **kw):
    cfg_ = {"md.n_steps": 6000, "md.equil_ps": 1.0, "md.min_cell_A": 8.0, "gates.block_ps": 1.0,
            "gates.min_production_ps": 2.0, "hull.temperatures": [0.0, 100.0, 200.0, 300.0, 400.0],
            "phases.candidates_root": w["cand"]}
    cfg_.update(kw.pop("config", {}))
    return ScreenOptions(backend="emt", database_root=w["db"], competing_root=w["competing"],
                         sources=["database", "candidates", "competing"], operating_T=300.0, device="cpu",
                         config=cfg_, **kw)


def test_pre_md_accepts_and_never_writes_into_candidates(world):
    before = sorted(os.listdir(world["cand"]))
    snap = sorted(str(p) for p in __import__("pathlib").Path(world["cand"]).rglob("*"))
    r = screen(world["cu3au"], "pre_md", options=opts(world))
    assert r.verdict == "accept", r.reasons
    assert r.stage == "pre_md" and r.e_hull_eV == r.e_hull_0K_eV and r.e_hull_eV < 0.001
    assert set(r.gates) == {"mechanical", "hull_0K"}
    assert sorted(str(p) for p in __import__("pathlib").Path(world["cand"]).rglob("*")) == snap
    assert sorted(os.listdir(world["cand"])) == before
    assert os.path.isdir(os.path.join(world["competing"], "Cu")) and r.complete
    assert os.path.exists(os.path.join(world["cu3au"], "stability", "result_pre_md.json"))
    assert not os.path.exists(os.path.join(world["cu3au"], "stability", "report_post_md.json"))
    assert cfg.calc.backend != "emt"                                   # global config untouched


def test_pre_md_gates_are_independent(world):
    r = screen(world["bcc"], "pre_md", options=opts(world))
    assert r.verdict == "reject" and any("mechanical" in x for x in r.reasons)
    assert "hull_0K" in r.gates                                          # the failed Born gate did not stop it
    only_hull = screen(world["bcc"], "pre_md", options=opts(world, gates=["thermodynamic"]))
    assert set(only_hull.gates) == {"hull_0K"}                           # config selects the gates


def test_thresholds_are_arguments(world):
    base = screen(world["agcu"], "pre_md", options=opts(world))
    assert base.verdict == "accept" and 0.015 < base.e_hull_0K_eV < 0.05          # immiscible, ~0.02 eV
    strict = screen(world["agcu"], "pre_md", options=opts(world, thresholds={"hull_fail_eV": 0.01}))
    assert strict.verdict == "reject"
    again = screen(world["agcu"], "pre_md", options=opts(world))
    assert again.verdict == "accept"                                    # nothing leaked from the strict call


def test_stages_do_not_mask_each_other(world):
    d = world["cu3au"]
    pre = screen(d, "pre_md", options=opts(world))
    post = screen(d, "post_md", options=opts(world))
    assert post.stage == "post_md" and post.verdict == "accept", post.reasons
    assert post.complete and set(post.gates) == {"mechanical", "structural_transition", "vibrational",
                                                 "thermodynamic"}
    again = screen(d, "pre_md", options=opts(world))                    # after a full run: still the pre_md answer
    assert again.stage == "pre_md" and set(again.gates) == {"mechanical", "hull_0K"} and pre.verdict == again.verdict
    st = os.path.join(d, "stability")
    assert {"report_pre_md.json", "report_post_md.json", "result_pre_md.json", "result_post_md.json"} <= set(
        os.listdir(st))
    import json
    assert json.load(open(os.path.join(st, "report_post_md.json")))["stage"] == "post_md"
    assert json.load(open(os.path.join(st, "report_pre_md.json")))["stage"] == "pre_md"


def test_post_md_cache_only_is_inconclusive_when_md_is_missing(world):
    r = screen(world["agcu"], "post_md", options=opts(world, compute=False))
    assert r.verdict == "inconclusive" and not r.complete


def test_post_md_rejects_immiscible_compound_with_tight_hull_threshold(world):
    r = screen(world["agcu"], "post_md", options=opts(world, thresholds={"hull_fail_eV": 0.01}))
    assert r.verdict == "reject" and r.gates["thermodynamic"].status == "fail"
    assert r.e_hull_eV > 0.01
