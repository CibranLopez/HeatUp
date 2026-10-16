import os

import pytest
from ase.build import bulk
from ase.io import write

from heatup import cfg
from heatup.generation import cell_multiplicities, enumerate_stoichiometries, generate, register_generator
from heatup.phases import find_competing_phases, sanitize_label, subsystems


def put(root, mat, lab, atoms):
    d = os.path.join(root, mat, lab)
    os.makedirs(d, exist_ok=True)
    write(os.path.join(d, "POSCAR"), atoms, format="vasp", direct=True, sort=True)
    return d


def test_sanitize_label_has_no_path_separator():
    assert "/" not in sanitize_label("P4/mmm") and sanitize_label("Fm-3m") == "Fm-3m"


def test_subsystems_of_ternary():
    s = subsystems(["Ba", "Ti", "O"])
    assert len(s) == 7 and ("Ba",) in s and ("Ba", "O", "Ti") in s


def test_stoichiometry_enumeration_reduced_and_bounded():
    s = enumerate_stoichiometries(["Ba", "O"], 5, 4)
    assert {"Ba": 1, "O": 1} in s and {"Ba": 2, "O": 2} not in s
    assert all(sum(x.values()) <= 5 for x in s)
    assert enumerate_stoichiometries(["Cu"], 5) == [{"Cu": 1}]
    assert cell_multiplicities(3, 10) == [1, 2, 3]


def test_competing_phases_exclude_target_and_duplicates(tmp_path):
    db = str(tmp_path / "db")
    cu = bulk("Cu", "fcc", a=3.61, cubic=True)
    au = bulk("Au", "fcc", a=4.08, cubic=True)
    cu3au = bulk("Cu", "fcc", a=3.75, cubic=True)
    cu3au[0].symbol = "Au"
    tgt = put(db, "Cu3Au", "Pm-3m", cu3au)
    put(db, "Cu", "a", cu)
    put(db, "Cu", "dup", bulk("Cu", "fcc", a=3.62, cubic=True))            # same structure within tolerance
    put(db, "Au", "a", au)
    put(db, "Cu3Au", "copy_of_target", cu3au)                               # target's twin must be excluded
    put(db, "Ag", "x", bulk("Ag", "fcc", a=4.09, cubic=True))               # element not in the target -> ignored
    cfg.phases.sources = ["database"]
    cfg.phases.database_root = db
    comp = find_competing_phases(tgt)
    tags = sorted(p.tag for p in comp)
    assert tags == ["Au/a", "Cu/a"], tags


def test_pyxtal_generation_is_seeded_and_deduplicated(tmp_path):
    pytest.importorskip("pyxtal")
    cfg.generation.max_atoms = 8
    a = generate({"Ba": 1, "O": 1}, n=5, root=str(tmp_path / "a"), seed=3)
    b = generate({"Ba": 1, "O": 1}, n=5, root=str(tmp_path / "b"), seed=3)
    assert a and sorted(p.formula for p in a) == sorted(p.formula for p in b)
    assert sorted(p.phase_dir.label for p in a) == sorted(p.phase_dir.label for p in b)
    assert all("/" not in p.phase_dir.label for p in a)


def test_custom_generator_registry(tmp_path):
    from pymatgen.core import Lattice, Structure
    register_generator("fixed", lambda counts, n, seed: [Structure(Lattice.cubic(3.6), ["Cu"], [[0, 0, 0]])])
    ph = generate({"Cu": 1}, n=3, root=str(tmp_path), method="fixed")
    assert len(ph) == 1 and ph[0].formula == "Cu"
