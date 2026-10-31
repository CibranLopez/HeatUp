"""End-to-end test on the EMT backend (no GPU, no MLIP): Cu3Au (L1_2) against Cu and Au.

prepare -> run_plan (each step in its own subprocess) -> status -> analysis -> gates, plus the properties the
restructuring is about: nothing is recomputed on a rerun, analysis never invalidates the MD, thresholds change
verdicts without recomputation, an unfinished/failed step is reported, analysis runs without a calculator.
"""
import json
import os
import sys

import pytest

pytest.importorskip("phonopy")
pytest.importorskip("pymatgen")

import heatup                                   # noqa: E402
from heatup import cfg, override               # noqa: E402
from heatup.core import ledger                 # noqa: E402
from heatup.prepare.phases import fcc, l1_2, write_phase  # noqa: E402


@pytest.fixture()
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    heatup.reset()
    cfg.calc.backend, cfg.calc.device = "emt", "cpu"
    cfg.phases.database_root, cfg.phases.candidates_root = "database", "input/candidates"
    cfg.phases.sources = ["database", "candidates"]
    cfg.phonon.min_length_A = 8.0
    cfg.md.n_steps, cfg.md.nblock, cfg.md.min_cell_A = 1000, 5, 8.0
    cfg.analysis.equil_ps, cfg.analysis.block_ps = 0.5, 1.0
    write_phase("input/candidates", "Cu3Au", "L1_2", l1_2("Cu", "Au", 3.75))
    write_phase("input/candidates", "Cu", "Fm-3m", fcc("Cu", 3.61))
    write_phase("input/candidates", "Au", "Fm-3m", fcc("Au", 4.08))
    yield [{"material": "Cu3Au", "symmetry": "L1_2", "temperature": 300}]
    heatup.reset()


def test_three_stages(project):
    from heatup.analyze.evaluate import evaluate, run_analysis, status
    from heatup.compute.execute import run_plan
    from heatup.prepare.plan import prepare_project, preview_plan

    prep = prepare_project(project)
    assert [t.tag for t in prep.targets] == ["Cu3Au/L1_2"] and len(prep.competitors) == 2
    target = prep.targets[0]
    plan = ledger.read_plan(target)
    assert plan.steps == ["relax", "elastic", "phonons", "md"] and plan.md_temperatures_K == [300.0]
    assert preview_plan(prep)[0]["md cell"]

    # nothing computed yet: everything pending, nothing "problematic" except pending
    rows = status()
    assert {r["state"] for r in rows} == {"pending"}

    # --- notebook 2 -------------------------------------------------------------------------
    first = run_plan()
    assert not [r for r in first if r["action"] == "failed"], first
    assert {r["action"] for r in first} <= {"ran", "pruned", "blocked"}
    assert target.has_relaxation()
    again = run_plan()                                  # idempotent: nothing is recomputed
    assert {r["action"] for r in again} <= {"done", "pruned"}, again

    # --- notebook 3 -------------------------------------------------------------------------
    assert all(r["state"] in ("done", "pending") for r in status())     # analysis steps pending
    out = run_analysis()
    assert all(r["ok"] for r in out), out
    assert {r["state"] for r in status() if r["step"] in ledger.ANALYSIS_STEPS} == {"done"}
    # the analysis must not invalidate the MD (v0.3 bug: shared manifest.json)
    assert all(r["action"] == "done" for r in run_plan() if r["step"] == "md")

    from heatup.analyze.evaluate import evaluate_all
    assert [f"{r['material']}/{r['label']}" for r in evaluate_all(save_figure=False)] == ["Cu3Au/L1_2"]   # competitors get no verdict
    rep = evaluate(target, save_figure=False)
    assert rep["mechanical"]["status"] == "ok" and rep["structural_transition"]["available"]
    assert rep["vibrational"]["available"] and rep["thermodynamic"]["available"]
    assert rep["thermodynamic"]["n_competing"] == 2 and rep["incomplete"] == {}
    assert rep["thermodynamic"]["e_hull_at_T_eV"] < 0.02            # Cu3Au is stable in EMT

    # thresholds change verdicts without recomputing (v0.3: Gate 0 verdict was cached)
    with override(**{"gates.transition_rms_A": 1e-4}):
        assert evaluate(target, gates=["structural_transition"], save_figure=False)["structural_transition"]["status"] == "fail"
    assert evaluate(target, gates=["structural_transition"], save_figure=False)["structural_transition"]["status"] != "fail"
    assert all(r["state"] == "done" for r in status() if r["step"] in ledger.ANALYSIS_STEPS)


def test_unfinished_and_failed_are_reported(project):
    from heatup.analyze.evaluate import evaluate, status
    from heatup.compute.execute import run_plan
    from heatup.prepare.plan import prepare_project

    target = prepare_project(project, competitors=False).targets[0]
    run_plan(steps=["relax"])
    # a started-but-unfinished MD run: trajectory without md.json
    sim = target.md_run(300)
    os.makedirs(sim, exist_ok=True)
    open(os.path.join(sim, "output.traj"), "wb").write(b"partial")
    st = {(r["step"]): r for r in status()}
    assert st["md"]["state"] == "unfinished" and "frames" in st["md"]["detail"]
    rep = evaluate(target, gates=["structural_transition"], save_figure=False)
    assert rep["structural_transition"]["status"] == "missing"
    assert any(f.startswith("[incomplete]") for f in rep["flags"])
    # a failing step leaves its error behind
    with override(**{"relax.max_steps": 1}):
        ledger.write_plan(target, ledger.Plan(role="target", steps=["relax"], options={"relax.max_steps": 1,
                          "relax.fmax": cfg.relax.fmax}, operating_T_K=300.0))
        for f in (target.energy_json, target.contcar):
            os.remove(f)
        rows = run_plan(steps=["relax"], include_competing=False)
        assert rows[0]["action"] == "failed"
    assert ledger.step_state(target, "relax")[0] == "failed"


def test_analysis_needs_no_calculator(project):
    import subprocess
    from heatup.compute.execute import run_plan
    from heatup.prepare.plan import prepare_project

    prepare_project(project)
    run_plan()
    code = ("import sys, heatup\nfrom heatup.analyze.evaluate import evaluate_all, run_analysis, status\n"
            "from heatup import cfg\ncfg.calc.backend='no-such-backend'\n"
            "cfg.analysis.equil_ps=0.5; cfg.analysis.block_ps=1.0\n"
            "run_analysis(); r = evaluate_all(save_figure=False)\n"
            "bad = [m for m in ('torch','mace','phonopy') if m in sys.modules]\nassert not bad, bad\n"
            "print(r[0]['overall'])\n")
    env = dict(os.environ, PYTHONPATH=os.path.dirname(os.path.dirname(os.path.abspath(heatup.__file__))))
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert res.returncode == 0, res.stderr[-800:]


def test_layout_and_plan_roundtrip(project):
    from heatup.prepare.plan import prepare_project
    target = prepare_project(project, competitors=False).targets[0]
    d = json.load(open(target.plan_json))
    assert d["role"] == "target" and "md.n_steps" in d["options"] and not any(k.startswith("calc.") for k in d["options"])


def test_screen_facade_pre_md(project):
    """The one-call entry point composes prepare -> run_plan -> evaluate for a single phase."""
    from heatup import screen
    write_phase("database", "Cu3Au", "L1_2", l1_2("Cu", "Au", 3.75))
    r = screen("database/Cu3Au/L1_2", stage="pre_md", gates=["mechanical", "thermodynamic"])
    assert r.verdict == "accept" and r.gates["mechanical"].status == "ok" and r.e_hull_0K_eV < 0.0, r
    assert os.path.exists(r.result_file) and r.complete
    # compute=False must not compute or fetch anything: a second phase has no relaxation -> inconclusive
    write_phase("database", "Cu3Au", "L1_2b", l1_2("Cu", "Au", 3.80))
    r2 = screen("database/Cu3Au/L1_2b", stage="pre_md", compute=False, gates=["mechanical"])
    assert r2.verdict == "inconclusive"
    assert not os.path.exists("database/Cu3Au/L1_2b/relaxation")


def test_qha_method(project):
    """QHA step -> hull with phonon_method='QHA'; a hull grid outside the QHA grid is refused, not clamped."""
    import numpy as np
    from heatup.analyze.evaluate import evaluate
    from heatup.compute.execute import run_plan
    from heatup.prepare.plan import prepare_project

    cfg.phonon.min_length_A = 6.0
    cfg.qha.n_volumes, cfg.qha.temperatures = 5, list(range(0, 601, 100))
    cfg.hull.temperatures, cfg.hull.phonon_method = list(range(0, 601, 100)), "QHA"
    cfg.plan.target_steps, cfg.plan.competitor_steps = ["relax", "qha"], ["relax", "qha"]
    target = prepare_project(project).targets[0]
    rows = run_plan()
    assert not [r for r in rows if r["action"] == "failed"], rows
    rep = evaluate(target, gates=["thermodynamic"], save_figure=False)
    th = rep["thermodynamic"]
    assert th["available"] and th["method"] == "QHA" and th["e_hull_at_T_eV"] < 0.02, th
    # G = H0 + contribution must reproduce G_QHA of the QHA file (+ nothing else at P = 0, no shifts)
    q = json.load(open(os.path.join(target.phonon_dir, "qha.json")))
    assert abs(np.interp(300, q["temperatures"], q["G_eV_per_atom"]) - np.interp(300, th["temperatures"], th["G_target_eV"])) < 1e-9
    cfg.hull.temperatures = list(range(0, 1501, 50))                # wider than the QHA grid
    th2 = evaluate(target, gates=["thermodynamic"], save_figure=False)["thermodynamic"]
    assert not th2["available"] and "QHA computed for" in th2["message"], th2


def test_gates_asked_for_drive_the_plan(project):
    """Ask for a gate -> its calculations are planned and run; notebook 3 reports exactly the gates asked for."""
    from heatup.analyze.evaluate import evaluate
    from heatup.compute.execute import run_plan
    from heatup.prepare.plan import prepare_project

    cfg.plan.gates = ["relax", "mechanical"]
    prep = prepare_project(project)
    target = prep.targets[0]
    assert ledger.read_plan(target).steps == ["relax", "elastic"] and prep.competitors == []   # no hull -> no competitors
    assert {r["action"] for r in run_plan()} == {"ran"} and os.path.exists(target.elastic_json)
    rep = evaluate(target, save_figure=False)
    assert rep["enabled"] == ["mechanical"] and rep["overall"] == "ok", rep["flags"]
    assert not os.path.exists(target.md_dir)


def test_protocol_is_just_calls(project):
    """steps= lets anyone build a staged protocol; the default run_plan() then finishes everything - no pruning by default."""
    from heatup.compute.execute import run_plan
    from heatup.prepare.plan import prepare_project

    prepare_project(project)
    first = run_plan(steps=["relax", "elastic"])
    assert {r["step"] for r in first} == {"relax", "elastic"} and all(r["action"] == "ran" for r in first)
    rest = run_plan()
    assert {r["action"] for r in rest if r["step"] in ("relax", "elastic")} == {"done"}
    assert {r["step"] for r in rest if r["action"] == "ran"} == {"phonons", "md"}
    assert not [r for r in rest if r["action"] == "pruned"]


def test_two_roots_only(project):
    """Structures live in candidates, everything computed in the database; candidates are never touched."""
    from heatup.compute.execute import run_plan
    from heatup.prepare.plan import prepare_project

    prep = prepare_project(project)
    assert sorted(os.listdir("database")) == ["Au", "Cu", "Cu3Au"] and not os.path.exists("competing_phases")
    roles = {p.tag: ledger.read_plan(p).role for p in map(heatup.PhaseDir, [f"database/{m}/{l}" for m, l in
             [("Cu3Au", "L1_2"), ("Cu", "Fm-3m"), ("Au", "Fm-3m")]])}
    assert roles == {"Cu3Au/L1_2": "target", "Cu/Fm-3m": "competitor", "Au/Fm-3m": "competitor"}
    run_plan(steps=["relax"])
    for root, _dirs, files in os.walk("input/candidates"):
        assert files in ([], ["POSCAR"]), (root, files)             # nothing was written next to the inputs


def test_database_competitors_are_reused_only_with_matching_settings(project):
    from heatup.analyze.evaluate import evaluate
    from heatup.analyze.hull import usable_competitors
    from heatup.compute.execute import run_plan
    from heatup.prepare.plan import prepare_project

    cfg.plan.gates = ["relax", "thermodynamic"]
    cfg.plan.target_steps = cfg.plan.competitor_steps = ["relax"]
    prep = prepare_project(project)
    run_plan()
    contcar = {t: os.path.getmtime(f"database/{t}/relaxation/CONTCAR") for t in ("Cu/Fm-3m", "Au/Fm-3m")}

    # same settings: reused as they are, nothing recomputed
    prep2 = prepare_project(project)
    assert prep2.refused == {} and len(prep2.competitors) == 2
    assert {r["action"] for r in run_plan()} == {"done"}

    # other settings: refused with a warning, left exactly as they are, never used in the hull
    cfg.relax.fmax = 0.02
    prep3 = prepare_project(project)
    assert set(prep3.refused) == {"Cu/Fm-3m", "Au/Fm-3m"} and prep3.competitors == []
    assert {t: os.path.getmtime(f"database/{t}/relaxation/CONTCAR") for t in contcar} == contcar
    assert ledger.read_plan("database/Cu/Fm-3m").options["relax.fmax"] == 0.005
    target = prep3.targets[0]
    assert usable_competitors(target)[0] == [] and set(usable_competitors(target)[1]) == set(prep3.refused)
    run_plan()                                                      # the target is recomputed with the new settings, they are not
    th = evaluate(target, gates=["thermodynamic"], save_figure=False)
    assert not th["thermodynamic"]["available"] and any(f.startswith("[note] thermodynamic: [reuse refused]") for f in th["flags"])
    assert {t: os.path.getmtime(f"database/{t}/relaxation/CONTCAR") for t in contcar} == contcar


def test_fc3_step_and_report(project):
    """fc3 (phono3py) as a planned step; its summary reaches the report as a diagnostic."""
    pytest.importorskip("phono3py")
    from heatup.analyze.evaluate import evaluate
    from heatup.analyze.thermo import anharmonic_summary, load_fc3
    from heatup.compute.execute import run_plan
    from heatup.prepare.plan import prepare_project

    cfg.phonon.fc_order, cfg.phonon.min_length_A = 3, 6.0
    cfg.fc3.supercell, cfg.fc3.cutoff_pair_A, cfg.fc3.kappa_mesh_length_A = (2, 2, 2), 3.0, 8.0
    cfg.fc3.temperatures, cfg.fc3.ratio_warn = [300.0], 1e-9              # any linewidth triggers the flag
    cfg.plan.gates = ["relax", "mechanical"]
    cfg.plan.target_steps = ["relax", "fc3"]
    target = prepare_project(project, competitors=False).targets[0]
    rows = run_plan()
    assert [r["action"] for r in rows] == ["ran", "ran"], rows
    res = load_fc3(target)
    assert res["kappa_W_mK"][0] > 0 and res["fc3_supercell"] == [2, 2, 2]
    assert anharmonic_summary(res, 300.0)["flag"]
    rep = evaluate(target, gates=["mechanical"], save_figure=False)
    assert rep["anharmonic"]["T_K"] == 300.0 and any("anharmonic" in f for f in rep["flags"])
    assert {r["action"] for r in run_plan()} == {"done"}                  # cached: manifest of the fc3 step is fresh


def test_prepare_generates_the_task_folders(project):
    """Notebook 1 creates the folder of every planned calculation, each saying what to compute (for DFT & co.)."""
    from heatup.analyze.evaluate import status
    from heatup.prepare.plan import prepare_project

    prepare_project(project)
    base = "database/Cu3Au/L1_2"
    for d in ("relaxation", "elastic", "phonons", "md/300K"):
        assert os.path.isdir(f"{base}/{d}"), d
    rec = json.load(open(f"{base}/md/300K/step_md.json"))
    md = rec["task"]
    assert rec["temperature_K"] == 300.0 and md["inputs"] == ["relaxation/CONTCAR"]
    assert md["outputs"] == ["md/300K/output.traj", "md/300K/md.json"]
    assert "options" not in md and md["settings"]["file"] == "../../plan.json" and "md.*" in md["settings"]["keys"]
    assert json.load(open(f"{base}/plan.json"))["options"]["md.n_steps"] == cfg.md.n_steps       # the one place for settings
    assert json.load(open(f"{base}/relaxation/step_relax.json"))["task"]["inputs"] == ["POSCAR"]
    assert open(f"{base}/relaxation/POSCAR").read() == open(f"{base}/POSCAR").read()               # input kept next to CONTCAR
    assert os.path.isdir("database/Cu/Fm-3m/phonons") and not os.path.exists("database/Cu/Fm-3m/md")   # competitor: relax + phonons
    assert {r["state"] for r in status()} == {"pending"}                       # empty folders are not "started"
    assert heatup.PhaseDir(base).md_temperatures(complete=False) == []
    # asking for less removes the tasks that are no longer planned
    cfg.plan.gates = ["relax", "mechanical"]
    prepare_project(project, competitors=False)
    assert os.path.exists(f"{base}/elastic/step_elastic.json") and not os.path.exists(f"{base}/md/300K/step_md.json")


def test_relax_not_requested_uses_the_structure_as_given(project):
    """'relax' left out of cfg.plan.gates: the structure is taken as already relaxed (single point only)."""
    import numpy as np
    from ase.io import read
    from heatup.analyze.evaluate import evaluate
    from heatup.compute.execute import run_plan
    from heatup.prepare.plan import prepare_project

    write_phase("input/candidates", "Cu3Au", "strained", l1_2("Cu", "Au", 3.90))      # not at equilibrium
    entry = [{"material": "Cu3Au", "symmetry": "strained", "temperature": 300}]
    cfg.plan.gates = ["mechanical"]                                    # no "relax" asked for
    target = prepare_project(entry, competitors=False).targets[0]
    assert ledger.read_plan(target).options["relax.optimize"] is False
    assert {r["action"] for r in run_plan()} == {"ran"}
    e = json.load(open(target.energy_json))
    assert e["optimized"] is False and e["converged"] is True and e["fmax_final_eV_A"] >= 0.0
    assert np.allclose(read(target.contcar).cell.array, read(target.poscar).cell.array)   # cell kept as given
    assert not os.path.exists(os.path.join(target.relax_dir, "run.traj"))
    rep = evaluate(target, save_figure=False)
    assert rep["mechanical"]["available"] and any("residual stress" in f for f in rep["flags"])   # strain is visible
    # asked for: the same structure is relaxed and the cell changes
    cfg.plan.gates = ["relax", "mechanical"]
    prepare_project(entry, competitors=False)
    assert ledger.read_plan(target).options["relax.optimize"] is True
    run_plan()
    assert not np.allclose(read(target.contcar).cell.array, read(target.poscar).cell.array, atol=1e-3)
    assert json.load(open(target.energy_json))["optimized"] is True


def test_md_for_competing_phases_when_asked(project):
    """Default: competitors get relax + phonons only; listing 'md' gives them md/<T>K too (targets' temperatures by default)."""
    from heatup.prepare.plan import prepare_project

    prepare_project(project)
    assert not os.path.exists("database/Cu/Fm-3m/md")
    cfg.plan.competitor_steps = ["relax", "phonons", "md"]
    cfg.plan.md_temperatures = None
    prepare_project(project)
    assert os.path.isdir("database/Cu/Fm-3m/md/300K") and os.path.exists("database/Au/Fm-3m/md/300K/step_md.json")
    assert ledger.read_plan("database/Cu/Fm-3m").md_temperatures_K == [300.0]


def test_free_energy_method_decides_what_competitors_need(project):
    """cfg.hull.phonon_method set in notebook 1 plans the right calculations (md only where the method needs it)."""
    from heatup.prepare.plan import prepare_project

    prepare_project(project)                                               # default HA: competitors relax + phonons
    assert ledger.read_plan("database/Cu/Fm-3m").steps == ["relax", "phonons"]
    assert not os.path.exists("database/Cu/Fm-3m/md")
    cfg.plan.competitor_steps = None
    cfg.hull.phonon_method = "VDOS"
    cfg.md.n_steps, cfg.md.nblock = 600, 5
    prep = prepare_project(project)
    assert ledger.read_plan("database/Au/Fm-3m").steps == ["relax", "phonons", "md"]    # phonons planned earlier stay (merge)
    assert ledger.read_plan("database/Au/Fm-3m").md_temperatures_K == [300.0]            # the queried material's temperature
    assert os.path.exists("database/Cu/Fm-3m/md/300K/step_md.json")
    assert ledger.read_plan(prep.targets[0]).options["hull.phonon_method"] == "VDOS"


def test_vdos_hull_end_to_end(project):
    """Gate 3 with anharmonic (MD-spectrum) free energies for every phase, followed from the plan by 'notebook 3'."""
    from heatup.analyze.evaluate import evaluate, run_analysis
    from heatup.compute.execute import run_plan
    from heatup.prepare.plan import prepare_project

    cfg.md.n_steps, cfg.md.nblock, cfg.md.min_cell_A = 1500, 5, 6.0
    cfg.analysis.equil_ps = 0.3
    cfg.plan.gates = ["relax", "thermodynamic"]
    cfg.hull.phonon_method = "VDOS"
    prep = prepare_project(project)
    target = prep.targets[0]
    assert ledger.read_plan(target).steps == ["relax", "md"]
    assert all(ledger.read_plan(p.phase_dir).steps == ["relax", "md"] for p in prep.competitors)
    rows = run_plan()
    assert not [r for r in rows if r["action"] == "failed"], rows
    assert {r["step"] for r in rows} == {"relax", "md"}
    cfg.hull.phonon_method = None                                           # notebook 3 knows nothing: the plan does
    assert all(r["ok"] for r in run_analysis())
    th = evaluate(target, save_figure=False)["thermodynamic"]
    assert th["available"] and th["method"] == "VDOS" and th["n_competing"] == 2, th
    assert th["e_hull_at_T_eV"] < 0.05 and th["overlay"] is None            # a consistent VDOS hull needs no overlay


def test_vdos_coverage_flags():
    from heatup.analyze.hull import vdos_coverage_flags
    assert vdos_coverage_flags(300, {"t": [300], "a": [300]}) == []
    assert vdos_coverage_flags(600, {"t": [300, 900], "a": [300, 900]}) == []              # interpolated, same for everybody
    f = vdos_coverage_flags(300, {"t": [300], "a": [300, 900]})
    assert len(f) == 1 and "differ" in f[0]
    f = vdos_coverage_flags(450, {"t": [300], "a": [300]})
    assert len(f) == 2 and all("outside" in x for x in f)


def test_export_inputs_and_external_roundtrip(project):
    """Notebook 1 writes the cells for any code; forces/stresses from an outside calculator come back and reproduce
    HeatUp's own numbers (EMT plays the role of DFT here: it is only ever called outside HeatUp)."""
    import glob
    import numpy as np
    from ase.calculators.emt import EMT
    from ase.io import read
    from ase.optimize import BFGS
    from heatup.compute.execute import run_plan
    from heatup.compute.phonons import collect_ha
    from heatup.compute.static import collect_elastic
    from heatup.core.config import EV_A3_TO_GPA
    from heatup.prepare.export import export_inputs
    from heatup.prepare.plan import prepare_project

    cfg.phonon.min_length_A = 6.0
    cfg.plan.gates = ["relax", "mechanical", "vibrational"]
    prep = prepare_project(project, competitors=False)
    target = prep.targets[0]
    assert export_inputs(prep)[0]["state"].startswith("skipped")           # not relaxed yet: nothing to build cells from
    run_plan(steps=["relax"])
    rows = {r["step"]: r for r in export_inputs(prep)}
    assert set(rows) == {"phonons", "elastic", "md"} and {r["state"] for r in rows.values()} == {"written"}
    base = target.path
    assert os.path.exists(f"{base}/phonons/ha/phonopy_disp.yaml") and os.path.exists(f"{base}/phonons/ha/SPOSCAR")
    assert len(glob.glob(f"{base}/elastic/strain/*/POSCAR")) == 12 and os.path.exists(f"{base}/md/POSCAR")
    assert len(json.load(open(f"{base}/elastic/step_elastic.json"))["export"]["files"]) == 13
    assert {r["state"] for r in export_inputs(prep)} == {"up to date"}
    syms = read(f"{base}/phonons/ha/disp-001/POSCAR").get_chemical_symbols()         # species in one block each: VASP-ready
    assert 1 + sum(a != b for a, b in zip(syms, syms[1:])) == len(set(syms))

    # a code HeatUp knows nothing about computes the exported cells
    def forces(path):
        a = read(path)
        a.calc = EMT()
        return a.get_forces()
    disp = sorted(glob.glob(f"{base}/phonons/ha/disp-*"))
    collect_ha(target, np.array([forces(f"{d}/POSCAR") for d in disp]), perfect_forces=forces(f"{base}/phonons/ha/SPOSCAR"))
    assert ledger.step_state(target, "phonons")[0] == "external"                      # no manifest: accepted as it is
    ext_zpe = json.load(open(f"{base}/phonons/ha.json"))["zpe_eV_per_atom"]
    stresses = {}
    for d in sorted(glob.glob(f"{base}/elastic/strain/*/")):
        a = read(d + "POSCAR")
        a.calc = EMT()
        BFGS(a, logfile=None).run(fmax=cfg.elastic.fmax, steps=300)                   # ions relaxed at fixed cell
        stresses[os.path.basename(d.rstrip("/"))] = a.get_stress(voigt=True) * EV_A3_TO_GPA
    a0 = read(target.contcar)
    a0.calc = EMT()
    collect_elastic(target, stresses, a0.get_stress(voigt=True) * EV_A3_TO_GPA)
    ext_C = np.array(json.load(open(target.elastic_json))["C_GPa"])
    assert ledger.step_state(target, "elastic")[0] == "external"

    # HeatUp's own steps, forced, give the same numbers: both used the same cells
    with override(**{"run.force": ["phonons", "elastic"]}):
        run_plan(steps=["phonons", "elastic"])
    assert abs(json.load(open(f"{base}/phonons/ha.json"))["zpe_eV_per_atom"] - ext_zpe) < 1e-9
    assert np.allclose(json.load(open(target.elastic_json))["C_GPa"], ext_C, atol=1e-6)


def test_given_structures_get_contcar_without_any_calculation(project):
    """'relax' not asked for: CONTCAR is the given structure, the cells are exported, no calculator is involved."""
    from ase.io import read
    from heatup.prepare.export import export_inputs
    from heatup.prepare.plan import prepare_project

    cfg.plan.gates = ["vibrational"]
    cfg.phonon.min_length_A = 6.0
    prep = prepare_project(project, competitors=False)
    target = prep.targets[0]
    rows = export_inputs(prep)
    assert {r["step"] for r in rows} == {"phonons", "md"}
    assert os.path.exists(target.contcar) and not os.path.exists(target.energy_json)
    assert (read(target.contcar).positions == read(target.poscar).positions).all() and not target.has_relaxation()


def test_fc3_displacements_are_exported(project):
    pytest.importorskip("phono3py")
    import glob
    from heatup.compute.execute import run_plan
    from heatup.prepare.export import export_inputs
    from heatup.prepare.plan import prepare_project

    cfg.phonon.fc_order, cfg.phonon.min_length_A = 3, 6.0
    cfg.fc3.supercell, cfg.fc3.cutoff_pair_A = (2, 2, 2), 3.0
    cfg.plan.gates, cfg.plan.target_steps = ["relax"], ["relax", "fc3"]
    prep = prepare_project(project, competitors=False)
    run_plan(steps=["relax"])
    row = export_inputs(prep)[0]
    d = prep.targets[0].path + "/phonons/fc3"
    assert row["state"] == "written" and os.path.exists(f"{d}/phono3py_disp.yaml")
    assert len(glob.glob(f"{d}/disp-*/POSCAR")) >= 1


def test_one_record_per_step_and_no_duplicated_settings(project):
    from heatup.compute.execute import run_plan
    from heatup.prepare.plan import prepare_project

    prepare_project(project, competitors=False)
    run_plan(steps=["relax"])
    base = "database/Cu3Au/L1_2"
    d = f"{base}/relaxation"
    assert sorted(f for f in os.listdir(d) if f.endswith(".json")) == ["energy.json", "step_relax.json"]   # ONE metadata file
    rec = json.load(open(f"{d}/step_relax.json"))
    assert set(rec) >= {"task", "run", "result"} and rec["run"]["state"] == "done" and "options" not in rec["task"]
    plan = json.load(open(f"{base}/plan.json"))
    # settings: planned (plan.json) and used (the result's fingerprint) - nowhere else
    assert plan["options"]["relax.fmax"] == cfg.relax.fmax == rec["result"]["fingerprint"]["relax.fmax"]
    assert open(f"{d}/POSCAR").read() == open(f"{base}/POSCAR").read() and os.path.exists(f"{d}/CONTCAR")


def test_poscar_in_every_calculation_folder(project):
    from heatup.compute.execute import run_plan
    from heatup.prepare.export import export_inputs
    from heatup.prepare.plan import prepare_project

    cfg.phonon.min_length_A = 6.0
    cfg.md.min_cell_A = 6.0
    prep = prepare_project(project, competitors=False)
    target = prep.targets[0]
    run_plan(steps=["relax"])
    export_inputs(prep)
    base = target.path
    contcar = open(target.contcar).read()
    assert open(f"{base}/elastic/POSCAR").read() == contcar and open(f"{base}/phonons/POSCAR").read() == contcar
    assert open(f"{base}/md/300K/POSCAR").read() == open(f"{base}/md/POSCAR").read()
    assert open(f"{base}/relaxation/POSCAR").read() == open(target.poscar).read()


def test_failed_export_leaves_no_empty_folder(project):
    from heatup.compute.execute import run_plan
    from heatup.prepare.export import export_inputs
    from heatup.prepare.plan import prepare_project

    cfg.phonon.max_atoms = 1                                              # no phonon supercell fits
    prep = prepare_project(project, competitors=False)
    run_plan(steps=["relax"])
    rows = {r["step"]: r for r in export_inputs(prep)}
    assert rows["phonons"]["state"].startswith("failed") and rows["elastic"]["state"] == "written"
    assert not os.path.exists(f"{prep.targets[0].path}/phonons/ha")


def test_files_of_earlier_versions_are_still_read(project):
    """Per-step manifest_/status_ files (earlier 0.4 builds, v0.3 manifest.json) are read and folded into the step record."""
    from heatup.compute.execute import run_plan
    from heatup.prepare.plan import prepare_project

    target = prepare_project(project, competitors=False).targets[0]
    run_plan(steps=["relax"])
    rec = json.load(open(f"{target.relax_dir}/step_relax.json"))
    os.remove(f"{target.relax_dir}/step_relax.json")
    json.dump({"step": "relax", **rec["result"]}, open(f"{target.relax_dir}/manifest_relax.json", "w"))
    json.dump({"step": "relax", **rec["run"]}, open(f"{target.relax_dir}/status_relax.json", "w"))
    assert ledger.step_state(target, "relax", calc=True)[0] == "done"
    ledger.mark(target, "relax", None, "done")                           # the next write folds them in
    assert os.path.exists(f"{target.relax_dir}/step_relax.json")
    assert not os.path.exists(f"{target.relax_dir}/manifest_relax.json") and not os.path.exists(f"{target.relax_dir}/status_relax.json")
    assert ledger.step_state(target, "relax", calc=True)[0] == "done"
