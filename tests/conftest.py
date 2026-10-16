import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from heatup import cfg  # noqa: E402
from heatup.config import reset  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: end-to-end runs with the EMT test potential (minutes)")


@pytest.fixture(autouse=True)
def _fresh_config():
    reset()
    yield
    reset()


@pytest.fixture
def emt():
    """EMT backend (Cu/Ag/Au only) + short-run settings, for MD-free/light tests."""
    cfg.calc.backend = "emt"
    cfg.md.n_steps = 6000
    cfg.md.equil_ps = 1.0
    cfg.md.min_cell_A = 8.0
    cfg.gates.block_ps = 1.0
    cfg.gates.min_production_ps = 2.0
    return cfg
