"""heatup.structure_utils
==========================
Backward-compatibility shim.

All structure-preparation utilities previously defined here have been moved to
:mod:`heatup.structure_pipeline`, which is now the canonical location.
This module re-exports them so that existing code continues to work::

    # Old import (still works):
    from heatup.structure_utils import run_relaxation_subprocess, FMAX

    # New canonical import:
    from heatup.structure_pipeline import run_relaxation_subprocess
    from heatup import config; config.RELAX_FMAX

.. deprecated::
    Import directly from :mod:`heatup.structure_pipeline` and
    :mod:`heatup.config` in new code.
"""

from __future__ import annotations

from heatup.structure_pipeline import (
    run_relaxation,
    run_relaxation_subprocess,
    run_phonons,
    run_phonons_subprocess,
    run_elastic,
    run_elastic_subprocess,
    prepare_aimd_folders,
    _make_cell_upper_triangular,
    _compute_supercell_repetitions,
    _write_space_group as write_space_group_file,
)
from heatup import config

# ---------------------------------------------------------------------------
# Legacy constant aliases (values now come from config.py)
# ---------------------------------------------------------------------------

#: Alias for ``config.RELAX_FMAX``.
FMAX: float = config.RELAX_FMAX

#: Alias for ``config.RELAX_MAX_STEPS``.
MAX_RELAX_STEPS: int = config.RELAX_MAX_STEPS

#: Alias for ``config.RELAX_CELL``.
RELAX_CELL: bool = config.RELAX_CELL

#: Alias for ``config.PHONON_SUPERCELL``.
PHONON_SUPERCELL: tuple = config.PHONON_SUPERCELL

#: Alias for ``config.PHONON_DELTA``.
PHONON_DELTA: float = config.PHONON_DELTA

#: Alias for ``config.AIMD_MIN_CELL_ANG``.
AIMD_MIN_CELL_ANG: float = config.AIMD_MIN_CELL_ANG

#: Alias for ``config.ELASTIC_DELTA``.
ELASTIC_DELTA: float = config.ELASTIC_DELTA

__all__ = [
    "run_relaxation", "run_relaxation_subprocess",
    "run_phonons",    "run_phonons_subprocess",
    "run_elastic",    "run_elastic_subprocess",
    "prepare_aimd_folders",
    "write_space_group_file",
    "FMAX", "MAX_RELAX_STEPS", "RELAX_CELL",
    "PHONON_SUPERCELL", "PHONON_DELTA", "AIMD_MIN_CELL_ANG", "ELASTIC_DELTA",
]
