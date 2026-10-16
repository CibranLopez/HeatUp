"""End-to-end runs with the EMT test potential (Ag/Au/Cu).  Slow (minutes): ``pytest -m slow``."""
import os

import numpy as np
import pytest
from ase.build import bulk
from ase.io import write

from heatup import cfg, run_pipeline, run_step
from heatup import manifest
from heatup.paths import PhaseDir

pytestmark = pytest.mark.slow


def put(root, mat, lab, atoms):
    d = os.path.join(root, mat, lab)
    os.makedirs(d, exist_ok=True)
    write(os.path.join(d, "POSCAR"), atoms, format="vasp", direct=True, sort=True)
    return d


def _apply(root):
    cfg.calc.backend = "emt"
    cfg.md.n_steps, cfg.md.equil_ps, cfg.md.min_cell_A = 6000, 1.0, 8.0
    cfg.gates.block_ps, cfg.gates.min_production_ps = 1.0, 2.0
    cfg.phases.sources = ["database"]
    cfg.phases.database_root = root
    cfg.phases.candidates_root = os.path.join(root, "_cand")
    cfg.hull.temperatures = list(range(0, 701, 100))
    cfg.hull.operating_T = 300.0


@pytest.fixture(scope="module")
def ag_au_cu(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("db"))
    put(root, "Ag", "Fm-3m", bulk("Ag", "fcc", a=4.09, cubic=True))
    put(root, "Au", "Fm-3m", bulk("Au", "fcc", a=4.08, cubic=True))
    put(root, "Cu", "Fm-3m", bulk("Cu", "fcc", a=3.61, cubic=True))
    cuau = bulk("Cu", "fcc", a=3.85, cubic=True)
    cuau.set_chemical_symbols(["Cu", "Au", "Au", "Cu"])
    put(root, "CuAu", "P4-mmm", cuau)
    c3a = bulk("Cu", "fcc", a=3.75, cubic=True)
    c3a[0].symbol = "Au"
    tgt = put(root, "Cu3Au", "Pm-3m", c3a)
    return root, tgt


@pytest.fixture(autouse=True)
def _emt_config(ag_au_cu):          # conftest resets cfg before every test; re-apply the EMT setup
    _apply(ag_au_cu[0])


def test_cu3au_passes_all_gates_and_is_on_the_hull(ag_au_cu):
    root, tgt = ag_au_cu
    r = run_pipeline(tgt, operating_T=300.0, device="cpu")
    assert r["overall"] == "ok", r["flags"]
    assert r["mechanical"]["born_stable"] and r["mechanical"]["bulk_GPa"] > 100
    assert r["structural_transition"]["status"] == "ok"
    assert r["vibrational"]["status"] == "ok"
    th = r["thermodynamic"]
    assert th["status"] == "ok" and th["e_hull_at_T_eV"] < 1e-3          # EMT: ordered Cu3Au is below Cu-Au
    ha = PhaseDir(tgt).join("phonons", "ha.json")
    assert os.path.exists(ha)
    from heatup.phonons import load_ha
    assert load_ha(tgt)["modes"].total_weight == pytest.approx(3.0)        # primitive-cell reduction guard
    assert os.path.exists(PhaseDir(tgt).join("stability", "report_post_md.json"))


def test_ha_and_vdos_free_energies_agree(ag_au_cu):
    from heatup.free_energy import free_energy
    _, tgt = ag_au_cu
    T = np.array([100.0, 300.0, 600.0])
    ha = free_energy(tgt, T, "HA")["G_eV_per_atom"]
    vd = free_energy(tgt, T, "VDOS")["G_eV_per_atom"]
    assert np.abs(ha - vd).max() < 5e-3                                   # harmonic crystal: MD spectrum ~ HA


def test_cache_is_reused_and_invalidated_by_config(ag_au_cu):
    _, tgt = ag_au_cu
    ph = PhaseDir(tgt)
    before = os.path.getmtime(ph.elastic_json)
    run_step("elastic", tgt, device="cpu")
    assert os.path.getmtime(ph.elastic_json) == before                    # fresh -> not recomputed
    cfg.elastic.strain = cfg.elastic.strain * 2
    assert not manifest.is_fresh(ph.elastic_dir, "elastic", ph.material, [ph.contcar])
    cfg.elastic.strain = cfg.elastic.strain / 2
    assert manifest.is_fresh(ph.elastic_dir, "elastic", ph.material, [ph.contcar])


def test_born_failure_does_not_stop_other_gates(ag_au_cu):
    root, _ = ag_au_cu
    d = put(root, "Cu_bcc", "Im-3m", bulk("Cu", "bcc", a=2.87, cubic=True))
    r = run_pipeline(d, operating_T=300.0, device="cpu", gates=["mechanical", "vibrational"])
    assert r["mechanical"]["status"] == "fail" and r["overall"] == "fail"
    assert r["vibrational"] is not None and r["vibrational"]["status"] != "missing"   # still evaluated
    assert r["skipped"] == ["structural_transition", "thermodynamic"] and "stopped_at" not in r


def test_immiscible_agcu_sits_above_the_hull_and_decomposes(ag_au_cu, tmp_path):
    from heatup.thermodynamic import assess_thermodynamic_stability
    root = str(tmp_path)
    put(root, "Ag", "Fm-3m", bulk("Ag", "fcc", a=4.09, cubic=True))
    put(root, "Cu", "Fm-3m", bulk("Cu", "fcc", a=3.61, cubic=True))
    x = bulk("Cu", "fcc", a=3.85, cubic=True)
    x.set_chemical_symbols(["Cu", "Ag", "Ag", "Cu"])
    t = put(root, "AgCu", "P4-mmm", x)
    cfg.phases.database_root = root
    cfg.hull.temperatures = [0.0, 300.0, 600.0]
    assert run_step("relax", t, device="cpu")
    r = assess_thermodynamic_stability(t, device="cpu")
    assert r["status"] == "warn" and 0.010 < r["e_hull_at_T_eV"] < 0.050   # EMT Ag-Cu mixing is endothermic
    assert set(r["decomposition"]) == {"Ag/Fm-3m", "Cu/Fm-3m"}
    assert sum(r["decomposition"].values()) == pytest.approx(1.0)
