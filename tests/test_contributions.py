"""Tests for heatup.core.contributions (needs the heatup package on PYTHONPATH).   pytest -q test_contributions.py"""
import importlib
import json
import os

import numpy as np
import pytest

import heatup
from heatup.core import contributions as C, ledger
from heatup.core.config import cfg, override
from heatup.core.layout import PhaseDir
from heatup.analyze import thermo

KB = 8.617333262e-5


@pytest.fixture(autouse=True)
def clean_cfg():
    heatup.reset()
    yield
    heatup.reset()


def make_phase(root, tag, h0, n_atoms=1, files=None, plan_options=None):
    d = os.path.join(root, *tag.split("/"))
    os.makedirs(os.path.join(d, "relaxation"), exist_ok=True)
    json.dump({"energy_eV_per_atom": h0, "enthalpy_eV_per_atom": h0, "n_atoms": n_atoms, "converged": True,
               "calculator": "stub"}, open(os.path.join(d, "relaxation", "energy.json"), "w"))
    open(os.path.join(d, "relaxation", "CONTCAR"), "w").write("stub")
    open(os.path.join(d, "POSCAR"), "w").write("stub")
    for rel, content in (files or {}).items():
        os.makedirs(os.path.dirname(os.path.join(d, rel)), exist_ok=True)
        json.dump(content, open(os.path.join(d, rel), "w"))
    if plan_options is not None:
        ledger.write_plan(d, ledger.Plan(role="target", steps=["relax"], gates=["thermodynamic"], options=plan_options))
    return PhaseDir(d)


# ---------------------------------------------------------------- the registry is self-consistent -------------
def test_registry_is_consistent():
    for key, term in C.TERMS.items():
        assert term.default in [m.name for m in term.models], key
        assert term.models[0].name == "off", f"{key}: first model must be 'off'"
        for m in term.models:
            if m.func:
                mod, fn = m.func.split(":")
                assert callable(getattr(importlib.import_module(mod), fn)), m.func
            for s in m.steps:
                assert s in ledger.STEPS, f"{key}/{m.name}: unknown step {s}"
            for sec in m.settings:
                assert hasattr(cfg, sec), f"{key}/{m.name}: no cfg.{sec}"
            for t, allowed in m.requires:
                assert t in C.TERMS and all(a in [x.name for x in C.TERMS[t].models] for a in allowed)
    for key in C.TERMS:
        assert hasattr(cfg.terms, key), f"cfg.terms has no field {key}"
    assert set(ledger.THERMO_STEPS) == {m.name.upper() for m in C.TERMS["vibrational"].models}


# ---------------------------------------------------------------- precedence ---------------------------------
def test_defaults_keep_current_behaviour():
    assert C.resolve() == {"vibrational": "HA", "anharmonic": "off", "electronic": "dos", "magnetic": "ihj",
                           "configurational": "ideal", "orientational": "off", "defects": "off"}


def test_precedence(tmp_path):
    ph = make_phase(str(tmp_path), "M/a", 0.0, plan_options={"terms.vibrational": "QHA", "terms.configurational": "enumerated",
                                                              "hull.phonon_method": "QHA"})
    assert C.resolve(ph)["vibrational"] == "QHA" and C.resolve(ph)["configurational"] == "enumerated"   # as prepared
    cfg.hull.phonon_method = "HA"                                    # legacy switch beats the plan
    assert C.resolve(ph)["vibrational"] == "HA"
    cfg.terms.vibrational = "VDOS"                                   # cfg.terms beats the legacy switch
    assert C.resolve(ph)["vibrational"] == "VDOS"
    assert C.resolve(ph, vibrational="ZPE")["vibrational"] == "ZPE"  # explicit argument beats everything
    assert ledger.phonon_method(ph) == "VDOS" and ledger.phonon_method(ph, "QHA") == "QHA"
    heatup.reset()
    cfg.hull.include_electronic = False                              # legacy booleans: False = off
    assert C.resolve()["electronic"] == "off"
    cfg.terms.electronic = "dos"
    assert C.resolve()["electronic"] == "dos"


def test_names_are_case_insensitive_and_validated():
    cfg.terms.vibrational = "qha"
    assert C.resolve()["vibrational"] == "QHA"
    cfg.terms.vibrational = "nonsense"
    with pytest.raises(ValueError, match="choose from"):
        C.resolve()
    with pytest.raises(ValueError):
        C.resolve(bogus="x")


def test_only_switches_everything_else_off():
    cfg.terms.only(configurational="ideal")
    assert C.resolve() == {"vibrational": "off", "anharmonic": "off", "electronic": "off", "magnetic": "off",
                           "configurational": "ideal", "orientational": "off", "defects": "off"}
    with pytest.raises(ValueError):
        cfg.terms.only(nonsense="x")
    with pytest.raises(ValueError):
        cfg.terms.only(vibrational="nonsense")
    assert ledger.phonon_method(None) == "OFF"


# ---------------------------------------------------------------- rules --------------------------------------
def test_anharmonic_requires_qha():
    cfg.terms.vibrational, cfg.terms.anharmonic = "HA", "isochore"
    assert any(l == "error" for l, _ in C.check())
    with pytest.raises(ValueError, match="requires vibrational"):
        C.assemble()
    names = [n for n, _ in C.assemble(drop_invalid=True)._contrib]
    assert "F_anh" not in names                                      # diagnostics may drop it, never silently
    cfg.terms.vibrational = "QHA"
    assert not any(l == "error" for l, _ in C.check())
    assert "F_anh" in [n for n, _ in C.assemble()._contrib]


def test_warnings():
    cfg.terms.only(vibrational="off", orientational="rotors")
    assert any("libration" in m for l, m in C.check() if l == "warn")
    cfg.terms.only(vibrational="VDOS")
    assert any("experimental" in m for l, m in C.check() if l == "warn")
    cfg.terms.only(orientational="rotors", configurational="enumerated", vibrational="HA")
    assert any("ONCE" in m for l, m in C.check())
    cfg.terms.only(configurational="ideal", defects="pairs")
    assert any("vacancies" in m for l, m in C.check())
    heatup.reset()
    assert all(l != "warn" for l, _ in C.check())                    # the defaults are quiet (notes only)


# ---------------------------------------------------------------- what a selection needs ---------------------
def test_required_steps():
    assert C.required_steps(C.resolve(vibrational="off", electronic="off")) == ()
    cfg.terms.vibrational = "QHA"
    cfg.terms.anharmonic = "isochore"
    assert C.required_steps() == ("qha", "anharmonic")               # registry (execution) order
    cfg.terms.only(vibrational="VDOS")
    assert C.required_steps() == ("md",)
    assert C.required_steps(exclude=("vibrational",)) == ()


def test_planner_uses_the_selection(tmp_path):
    cfg.terms.vibrational, cfg.terms.anharmonic = "QHA", "isochore"
    for role in ("target", "competitor"):                            # the SAME terms for every phase
        steps = ledger.steps_for(["thermodynamic"], ledger.phonon_method(None), role)
        assert "qha" in steps and "anharmonic" in steps, role
    cfg.terms.only(configurational="ideal")                          # no calculation beyond the relaxation
    assert ledger.steps_for(["thermodynamic"], ledger.phonon_method(None), "target") == ["relax"]
    # notebook 3 follows what notebook 1 prepared
    ph = make_phase(str(tmp_path), "M/p", 0.0, plan_options={"terms.vibrational": "QHA", "terms.anharmonic": "isochore"})
    heatup.reset()                                                   # cfg.terms all None
    assert "anharmonic" in ledger.gate_steps("thermodynamic", "QHA", ph)
    assert "anharmonic" not in ledger.gate_steps("thermodynamic", "QHA", None)


def test_plan_stores_every_term_and_conflicts_are_flagged():
    cfg.terms.vibrational = "QHA"
    opts = C.plan_options()
    assert opts["terms.vibrational"] == "QHA" and set(opts) == {f"terms.{k}" for k in C.TERMS}
    existing = ledger.Plan(steps=["relax", "qha"], options={"terms.vibrational": "HA", "terms.configurational": "ideal"})
    conf = ledger.plan_conflicts(existing, ["relax", "qha"], {"terms.vibrational": "QHA", "terms.configurational": "ideal"})
    assert conf == {"terms.vibrational": ("HA", "QHA")}


# ---------------------------------------------------------------- assembling ----------------------------------
def test_assembler_contains_exactly_the_selected_terms():
    assert [n for n, _ in C.assemble()._contrib] == ["F_vib[HA]", "F_el", "F_mag", "F_conf"]      # today's behaviour
    cfg.terms.only(vibrational="QHA", configurational="enumerated", orientational="rotors", defects="pairs")
    assert [n for n, _ in C.assemble()._contrib] == ["F_vib[QHA]", "F_conf[enumerated]", "F_orientational", "F_defects"]
    cfg.terms.only()
    assert C.assemble()._contrib == []                                # G = H0: the 0 K hull
    assert [n for n, _ in thermo.build_assembler(method="QHA")._contrib] == ["F_vib[QHA]"]


def test_zero_point_only(tmp_path, monkeypatch):
    class Modes:
        def zpe(self):
            return 0.123
    monkeypatch.setattr(thermo, "load_ha", lambda p: {"modes": Modes(), "imag_fraction": 0.0})
    ph = make_phase(str(tmp_path), "M/z", -1.0)
    cfg.terms.only(vibrational="ZPE")
    res = thermo.free_energy(ph, [0.0, 500.0, 1000.0])
    assert np.allclose(res["G_eV_per_atom"], -1.0 + 0.123)            # H0 + ZPE, flat in T


def test_forced_method_drops_terms_that_lose_their_requirements(tmp_path, monkeypatch):
    monkeypatch.setattr(thermo, "f_vib_vdos", lambda p, T: np.zeros_like(T))
    cfg.terms.vibrational, cfg.terms.anharmonic = "QHA", "isochore"
    names = [n for n, _ in thermo.build_assembler(method="VDOS")._contrib]       # the hull's VDOS overlay
    assert "F_vib[VDOS]" in names and "F_anh" not in names


# ---------------------------------------------------------------- reporting -----------------------------------
def test_describe_and_inventory(tmp_path, capsys):
    rows = C.describe()
    assert [r["term"] for r in rows] == list(C.TERMS) and rows[0]["compute steps"] == "phonons"
    C.show(settings=True)
    out = capsys.readouterr().out
    assert "vibrational" in out and "cfg.phonon" in out and "compute steps for every phase: relax, phonons" in out
    cfg.terms.only(configurational="ideal")
    a = make_phase(str(tmp_path), "M/a", 0.0, files={"disorder/site_occupancies.json": {"sites": [{"species": {"A": 1.0}}]}})
    b = make_phase(str(tmp_path), "M/b", 0.0)
    inv = C.inventory([a, b])
    assert {r["phase"]: r["data"] for r in inv} == {"M/a": True, "M/b": False}
    assert all(r.get("inconsistent") for r in inv)                    # mixed presence is flagged


# ---------------------------------------------------------------- a hull of configurational entropy only -----
def test_hull_with_configurational_entropy_only(tmp_path):
    """A, B, an ordered AB (stable at 0 K) and a disordered AB (higher in energy, ideal mixing on one A/B site).
    With vibrational='off' the free energy is H0 - T S_conf: the disordered phase enters the hull at high T."""
    from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
    from pymatgen.core import Composition
    root = str(tmp_path)
    cfg.phases.database_root = root
    dis = {"disorder/site_occupancies.json": {"sites": [{"species": {"A": 0.5, "B": 0.5}}]}}
    phases = {"A": ("A/ref", 0.0, 1, {}), "B": ("B/ref", 0.0, 1, {}),
              "AB ordered": ("AB/ordered", -0.050, 2, {}), "AB disordered": ("AB/disordered", -0.020, 2, dis)}
    comp = {"A": "A", "B": "B", "AB ordered": "AB", "AB disordered": "AB"}
    ph = {k: make_phase(root, tag, h0, n, f) for k, (tag, h0, n, f) in phases.items()}
    cfg.terms.only(configurational="ideal")

    def e_above_hull(T):
        g = {k: float(thermo.free_energy(p, [T])["G_eV_per_atom"][0]) for k, p in ph.items()}
        entries = [PDEntry(Composition(comp[k]), g[k] * Composition(comp[k]).num_atoms, name=k) for k in ph]
        pd = PhaseDiagram(entries)
        return {e.name: pd.get_e_above_hull(e) for e in entries}, g

    d0, g0 = e_above_hull(0.0)
    assert g0["AB disordered"] == pytest.approx(-0.020)                       # T = 0: only H0 (static hull)
    assert d0["AB ordered"] == pytest.approx(0.0, abs=1e-12) and d0["AB disordered"] == pytest.approx(0.030, abs=1e-9)
    d300, _ = e_above_hull(300.0)
    assert d300["AB disordered"] > 0.005                                      # still above the hull at 300 K
    T1 = 1000.0
    d1, g1 = e_above_hull(T1)
    assert g1["AB disordered"] == pytest.approx(-0.020 - T1 * KB * np.log(2))
    assert d1["AB disordered"] == pytest.approx(0.0, abs=1e-12) and d1["AB ordered"] > 0.02   # the order is lost
    # the same database with the term switched off never changes
    cfg.terms.only()
    d_off, _ = e_above_hull(T1)
    assert d_off["AB ordered"] == pytest.approx(0.0, abs=1e-12) and d_off["AB disordered"] == pytest.approx(0.030, abs=1e-9)


def test_project_selection_follows_the_plans_and_flags_disagreement(tmp_path):
    root = str(tmp_path)
    a = make_phase(root, "M/a", 0.0, plan_options={"terms.vibrational": "QHA", "terms.anharmonic": "isochore"})
    b = make_phase(root, "M/b", 0.0, plan_options={"terms.vibrational": "QHA", "terms.anharmonic": "isochore"})
    sel, diff = C.project_selection([a, b])
    assert sel["vibrational"] == "QHA" and sel["anharmonic"] == "isochore" and diff == {}        # as prepared
    c = make_phase(root, "M/c", 0.0, plan_options={"terms.vibrational": "HA"})
    sel, diff = C.project_selection([a, b, c])
    assert diff == {"vibrational": ["HA", "QHA"], "anharmonic": ["isochore", "off"]}
    cfg.terms.vibrational = "ZPE"                                    # an explicit choice overrides every plan
    assert C.project_selection([a, b, c])[1].get("vibrational") is None
    rows = C.inventory([a])                                          # each phase with its OWN selection (cfg.terms still ZPE)
    assert {r["term"]: r["model"] for r in rows}["vibrational"] == "ZPE"


def test_show_reads_the_database(tmp_path, capsys):
    root = str(tmp_path)
    cfg.phases.database_root = root
    make_phase(root, "M/a", 0.0, plan_options={"terms.vibrational": "QHA"})
    make_phase(root, "M/b", 0.0, plan_options={"terms.vibrational": "HA"})
    C.show(settings=False)
    out = capsys.readouterr().out
    assert "different vibrational models" in out
