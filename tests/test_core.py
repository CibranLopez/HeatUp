"""Pure-function tests: config forwarding, ModeSet thermodynamics, elastic moduli, supercells, hull."""
import json

import numpy as np
import pytest

from heatup import cfg, override
from heatup.config import load_from_env, to_env, reset
from heatup.constants import KB_MEV
from heatup.elastic import moduli
from heatup.structure import perpendicular_widths, supercell_reps
from heatup.thermodynamic import hull_distance, hull_scan
from heatup.vibthermo import ModeSet, interpolate_f_vib


# ---- config ----------------------------------------------------------------
def test_config_roundtrip_through_env(monkeypatch):
    cfg.md.n_steps = 1234
    cfg.gates.mobile_species = ["Li"]
    env = to_env("X")
    reset()
    assert cfg.md.n_steps != 1234
    monkeypatch.setenv(*next(iter(env.items())))
    assert load_from_env()
    assert cfg.md.n_steps == 1234 and cfg.gates.mobile_species == ["Li"]


def test_override_restores():
    with override(**{"md.n_steps": 7}):
        assert cfg.md.n_steps == 7
    assert cfg.md.n_steps != 7


def test_unknown_key_rejected():
    with pytest.raises(Exception):
        cfg.md.not_an_option = 1


def test_per_material_model_resolved_for_children():
    cfg.calc.model_overrides = {"BaTiO3": "/x/BTO.model"}
    assert json.loads(next(iter(to_env("BaTiO3").values())))["calc"]["model"] == "/x/BTO.model"
    assert json.loads(next(iter(to_env("Si").values())))["calc"]["model"] != "/x/BTO.model"


# ---- ModeSet ---------------------------------------------------------------
def test_einstein_f_vib_and_zpe():
    w, T = 30.0, 400.0
    ms = ModeSet([w], [3.0])
    x = w / (KB_MEV * T)
    exact = 3 * (0.5 * w + KB_MEV * T * np.log1p(-np.exp(-x))) / 1000.0
    assert ms.f_vib(T) == pytest.approx(exact, rel=1e-12)
    assert ms.zpe() == pytest.approx(3 * 0.5 * w / 1000.0)
    assert ms.f_vib(0.0) == pytest.approx(ms.zpe())


def test_classical_limit_equipartition():
    ms = ModeSet([1.0], [3.0])               # kT >> hw: F = 3 kT ln(hw/kT)
    T = 3000.0
    assert ms.f_vib(T) == pytest.approx(3 * KB_MEV * T * np.log(1.0 / (KB_MEV * T)) / 1000.0, rel=2e-3)
    assert ms.cv(T) == pytest.approx(3 * KB_MEV / 1000.0, rel=1e-3)   # 3 kB per atom


def test_from_frequencies_weights_sum_to_three():
    om = np.random.default_rng(0).uniform(5, 40, size=(50, 12))      # 4-atom cell, 12 branches
    assert ModeSet.from_frequencies(om, np.ones(50), n_atoms=4).total_weight == pytest.approx(3.0)


def test_from_dos_normalised_to_three():
    g = np.linspace(0, 40, 400)
    ms = ModeSet.from_dos(g, g ** 2)
    assert ms.total_weight == pytest.approx(3.0)


def test_imaginary_fraction_and_skip():
    ms = ModeSet([-10.0, 10.0, 20.0, 30.0], [0.75] * 4)
    assert ms.imag_fraction(0.5) == pytest.approx(0.25)
    assert ms.max_imag_meV() == pytest.approx(10.0)
    assert np.isfinite(ms.f_vib(300.0))


def test_interpolate_f_vib_endpoints():
    a, b = ModeSet([20.0], [3.0]), ModeSet([15.0], [3.0])
    T = np.array([100.0, 300.0, 500.0])
    out = interpolate_f_vib([(300.0, a), (500.0, b)], T)
    assert out[1] == pytest.approx(a.f_vib(300.0)) and out[2] == pytest.approx(b.f_vib(500.0))
    assert out[0] == pytest.approx(a.f_vib(100.0))                    # constant extrapolation of the spectrum


# ---- elastic ---------------------------------------------------------------
def _cubic(c11, c12, c44):
    C = np.zeros((6, 6))
    for i in range(3):
        for j in range(3):
            C[i, j] = c11 if i == j else c12
    for i in range(3, 6):
        C[i, i] = c44
    return C


def test_cubic_moduli_analytic():
    c11, c12, c44 = 168.0, 121.0, 75.0                                # Cu
    m = moduli(_cubic(c11, c12, c44))
    assert m["bulk_hill_GPa"] == pytest.approx((c11 + 2 * c12) / 3)
    assert m["shear_voigt_GPa"] == pytest.approx((c11 - c12 + 3 * c44) / 5)
    assert m["shear_reuss_GPa"] == pytest.approx(5 * (c11 - c12) * c44 / (4 * c44 + 3 * (c11 - c12)))
    iso = moduli(_cubic(200.0, 100.0, 50.0))                          # Zener = 1 -> Voigt == Reuss
    assert iso["universal_anisotropy"] == pytest.approx(0.0, abs=1e-9)


# ---- supercells --------------------------------------------------------------
def test_supercell_uses_perpendicular_widths():
    a = 4.0                                                           # hexagonal-like (60 deg) cell
    cell = np.array([[a, 0, 0], [a / 2, a * np.sqrt(3) / 2, 0], [0, 0, 6.0]])
    reps = supercell_reps(cell, 10.0)
    w = perpendicular_widths(cell) * np.array(reps)
    assert (w >= 10.0 - 1e-9).all()
    assert reps == (3, 3, 2)               # perpendicular widths 4, 3.46, 6 (|a| alone would give 3,3,2 only by luck)
    with pytest.raises(ValueError):
        supercell_reps(cell, 20.0, max_atoms=50, n_atoms_cell=4)


# ---- hull --------------------------------------------------------------------
ELEMENTS = {"A": ({"A": 1.0}, 0.0), "B": ({"B": 1.0}, 0.0)}


def test_hull_distance_above_and_below():
    comp = {"A": 0.5, "B": 0.5}
    r = hull_distance(comp, 0.03, ELEMENTS)                           # competitors: pure elements only
    assert r["e_hull_eV"] == pytest.approx(0.03)
    r = hull_distance(comp, -0.10, ELEMENTS)
    assert r["e_hull_eV"] == pytest.approx(-0.10)                     # negative distance retained
    comps = dict(ELEMENTS, AB=({"A": 0.5, "B": 0.5}, -0.2))
    assert hull_distance(comp, -0.15, comps)["e_hull_eV"] == pytest.approx(0.05)


def test_hull_decomposition_lever_rule():
    comps = dict(ELEMENTS, A3B=({"A": 0.75, "B": 0.25}, -0.1), AB3=({"A": 0.25, "B": 0.75}, -0.1))
    r = hull_distance({"A": 0.5, "B": 0.5}, 0.0, comps)
    assert r["e_hull_eV"] == pytest.approx(0.1)                       # hull at x=.5 is -0.1 (A3B-AB3 tie line)
    assert sum(r["decomposition"].values()) == pytest.approx(1.0)


def test_hull_missing_elemental_reference():
    with pytest.raises(ValueError):
        hull_distance({"A": 0.5, "B": 0.5}, 0.0, {"A": ({"A": 1.0}, 0.0)})


def test_hull_scan_temperature_crossing():
    T = np.array([0.0, 100.0, 200.0])
    G_target = np.array([-0.05, -0.05, -0.05])
    comps = dict(ELEMENTS, AB=({"A": 0.5, "B": 0.5}, -0.10 + 0.0004 * T))   # competitor loses stability with T
    comps = {k: (c, np.full(3, g) if np.isscalar(g) else g) for k, (c, g) in comps.items()}
    s = hull_scan({"A": 0.5, "B": 0.5}, G_target, comps)
    assert s["e_hull_eV"][0] == pytest.approx(0.05)
    assert s["e_hull_eV"][2] == pytest.approx(-0.05 + 0.02)  # competitor at -0.02 -> target below it
    assert s["e_hull_eV"][2] < 0 < s["e_hull_eV"][0]


def test_upper_triangular_is_exact_even_for_float_noise():
    from ase import Atoms
    from heatup.structure import upper_triangular
    cell = np.diag([4.0, 4.0, 5.0])
    cell[1, 0] = cell[0, 1] = 1.2e-15                                 # relaxation noise: ASE's NPT needs exact zeros
    a = Atoms("Cu2", cell=cell, scaled_positions=[[0, 0, 0], [.5, .5, .5]], pbc=True)
    b = upper_triangular(a)
    assert b.cell[1, 0] == 0.0 and b.cell[2, 0] == 0.0 and b.cell[2, 1] == 0.0
    assert np.allclose(b.cell @ b.cell.T, a.cell @ a.cell.T)           # same lattice


def test_normalize_force_and_forced():
    from heatup.config import forced, normalize_force
    assert normalize_force(True) == ["all"] and normalize_force(False) == [] and normalize_force(None) == []
    assert normalize_force("md") == ["md"] and normalize_force(["relax", "md"]) == ["relax", "md"]
    cfg.run.force = ["md"]
    assert forced("md") and not forced("relax")
    assert forced("relax", True) and not forced("md", False)             # explicit argument wins
    cfg.run.force = ["all"]
    assert all(forced(s) for s in ("relax", "elastic", "phonons", "qha", "md", "analysis"))


def test_force_unknown_step_or_gate_rejected():
    from heatup import override
    with pytest.raises((ValueError, KeyError)):
        with override(**{"gates.enabled": ["mechanical", "nonsense"]}):
            from heatup.pipeline import run_pipeline
            run_pipeline("/nonexistent/X/a", compute=False)


def test_layout_lists_steps_and_sizes(tmp_path):
    from heatup.report import layout
    d = tmp_path / "Cu" / "Fm-3m" / "relaxation"
    d.mkdir(parents=True)
    (d / "CONTCAR").write_text("x" * 2048)
    (tmp_path / "Cu" / "Fm-3m" / "POSCAR").write_text("y")
    for i in range(12):
        (d / f"f{i}.dat").write_text("z")
    txt = layout(str(tmp_path), max_files=3)
    assert "relaxation/" in txt and "CONTCAR  (2.0 kB)" in txt and "more file(s)" in txt
    assert "does not exist" in layout(str(tmp_path / "nope"))


def test_fc3_config_and_manifest_isolation(tmp_path):
    from heatup import manifest
    assert cfg.phonon.fc_order == 2 and cfg.fc3.delta_A == 0.03
    f = tmp_path / "CONTCAR"; f.write_text("x")
    manifest.write_manifest(str(tmp_path), "phonons", None, [str(f)])
    manifest.write_manifest(str(tmp_path), "qha", None, [str(f)])
    manifest.write_manifest(str(tmp_path), "fc3", None, [str(f)])
    # qha / fc3 used to overwrite phonons/manifest.json and invalidate each other
    assert all(manifest.is_fresh(str(tmp_path), s, None, [str(f)]) for s in ("phonons", "qha", "fc3"))
    cfg.fc3.delta_A = 0.05
    assert manifest.is_fresh(str(tmp_path), "phonons", None, [str(f)])        # fc3 option: harmonic cache untouched
    assert not manifest.is_fresh(str(tmp_path), "fc3", None, [str(f)])


def test_linewidth_statistics_and_summary():
    import numpy as np
    from heatup.fc3 import anharmonic_summary, linewidth_statistics
    freq = np.array([[0.0, 4.0, 8.0], [2.0, 4.0, 8.0]])
    gamma = np.array([[0.0, 0.4, 0.8], [0.2, 0.4, 0.8]])                 # gamma/nu = 0.1 everywhere (except the Gamma modes)
    s = linewidth_statistics(freq, gamma, np.array([1.0, 3.0]), min_freq_THz=0.5)
    assert s["gamma_over_nu_mean"] == pytest.approx(0.1) and s["frac_ioffe_regel"] == 0.0
    assert s["tau_median_ps"] == pytest.approx(1 / (4 * np.pi * 0.4), rel=0.3)
    gamma[1] = [1.5, 3.0, 5.0]                                             # second point: gamma/nu = 0.75 -> beyond Ioffe-Regel
    s = linewidth_statistics(freq, gamma, np.array([1.0, 1.0]), 0.5)
    assert s["frac_ioffe_regel"] > 0.4
    res = {"temperatures_K": [100.0, 300.0], "kappa_W_mK": [30.0, 10.0], "gamma_over_nu_mean": [0.01, 0.08],
           "frac_ioffe_regel": [0.0, 0.0], "tau_median_ps": [5.0, 1.0]}
    assert anharmonic_summary(res, 120)["flag"] is None
    assert "anharmonic" in anharmonic_summary(res, 290)["flag"] and anharmonic_summary(None, 300) is None
