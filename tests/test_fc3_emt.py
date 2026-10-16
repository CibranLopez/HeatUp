"""fc3 / phono3py end-to-end on EMT copper (a few seconds)."""
import pytest

pytest.importorskip("phono3py")

from ase.build import bulk  # noqa: E402

from heatup import cfg, run_step  # noqa: E402
from heatup.fc3 import anharmonic_summary, load_fc3  # noqa: E402
from heatup.builders import write_phase  # noqa: E402


@pytest.fixture
def cu(tmp_path):
    cfg.calc.backend = "emt"
    cfg.phases.database_root = str(tmp_path)
    cfg.phonon.fc_order = 3
    cfg.phonon.min_length_A = 7.0
    cfg.fc3.supercell = (2, 2, 2)
    cfg.fc3.kappa_mesh_length_A = 25.0
    cfg.fc3.temperatures = [100.0, 300.0]
    d = write_phase(str(tmp_path), "Cu", "Fm-3m", bulk("Cu", "fcc", a=3.61, cubic=True))
    assert run_step("relax", d, device="cpu", isolated=False)
    return d


def test_fc3_step_gives_kappa_and_linewidths(cu):
    assert run_step("fc3", cu, device="cpu", isolated=False)
    res = load_fc3(cu)
    assert res["fc3_supercell"] == [2, 2, 2] and res["temperatures_K"] == [100.0, 300.0]
    k100, k300 = res["kappa_W_mK"]
    assert 5 < k300 < k100 < 200                                          # lattice kappa of a metal-like EMT: finite, falls with T
    assert 0 < res["gamma_over_nu_mean"][1] < 0.5 and res["gamma_over_nu_mean"][1] > res["gamma_over_nu_mean"][0]
    assert anharmonic_summary(res, 300)["T_K"] == 300.0


def test_fc3_is_cached_and_independent_of_harmonic_cache(cu, caplog):
    import logging
    assert run_step("fc3", cu, device="cpu", isolated=False)
    assert run_step("phonons", cu, device="cpu", isolated=False)
    with caplog.at_level(logging.INFO, logger="heatup"):
        assert run_step("fc3", cu, device="cpu", isolated=False)          # fresh despite phonons writing in the same folder
        assert run_step("phonons", cu, device="cpu", isolated=False)
    assert caplog.text.count("[cached]") == 2


def test_cutoff_pair_runs(cu):
    cfg.fc3.cutoff_pair_A = 3.0
    assert run_step("fc3", cu, device="cpu", isolated=False, force=True)
    assert load_fc3(cu)["cutoff_pair_A"] == 3.0
