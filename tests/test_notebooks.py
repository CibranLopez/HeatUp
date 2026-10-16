"""Notebooks hold configuration and visible loops only - logic lives in the library."""
import ast
import glob
import json
import os

import pytest

NB = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "notebooks", "*.ipynb")))


@pytest.mark.parametrize("path", NB)
def test_notebook_defines_nothing(path):
    nb = json.load(open(path))
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        tree = ast.parse("".join(c["source"]))
        bad = [type(n).__name__ for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda))]
        assert not bad, f"{os.path.basename(path)} cell {i} defines {bad}: move it into heatup/"


@pytest.mark.parametrize("path", NB)
def test_notebook_has_no_stored_outputs(path):
    nb = json.load(open(path))
    assert all(not c.get("outputs") for c in nb["cells"] if c["cell_type"] == "code")
