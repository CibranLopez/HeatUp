"""heatup.md_utils
==================
Backward-compatibility shim.

The database-scanning utilities and MD constants previously defined here have
been moved to :mod:`heatup.md_pipeline`, which is now the canonical location.
This module re-exports them so that existing code continues to work without
modification::

    # Old import (still works):
    from heatup.md_utils import scan_database, print_database_summary, NBLOCK

    # New canonical import:
    from heatup.md_pipeline import scan_database, print_database_summary, MD_NBLOCK

.. deprecated::
    Import directly from :mod:`heatup.md_pipeline` in new code.
    This shim will be removed in a future version.
"""

from __future__ import annotations

from heatup.md_pipeline import (
    scan_database,
    load_analysis,
    print_database_summary,
)
from heatup import config

# ---------------------------------------------------------------------------
# Legacy constant aliases
# These previously lived as module-level constants in md_utils.  They are now
# driven by config.py so they can be changed at runtime without editing source.
# The aliases below keep old notebooks working.
# ---------------------------------------------------------------------------

#: Alias for ``config.MD_TIMESTEP_FS``.  Use ``config.MD_TIMESTEP_FS`` in new code.
TIMESTEP_FS: float = config.MD_TIMESTEP_FS

#: Alias for ``config.MD_N_STEPS``.
N_STEPS: int = config.MD_N_STEPS

#: Alias for ``config.MD_NBLOCK``.
NBLOCK: int = config.MD_NBLOCK

#: Alias for ``config.MD_TTIME_FS``.
TTIME_FS: float = config.MD_TTIME_FS

#: Alias for ``config.MD_PTIME_FS``.
PTIME_FS: float = config.MD_PTIME_FS

#: Alias for ``config.MD_PRESSURE_GPA``.
PRESSURE_GPA: float = config.MD_PRESSURE_GPA

#: Alias for ``config.MD_STEP_SKIP``.
STEP_SKIP: int = config.MD_STEP_SKIP

#: Alias for ``config.MD_STEP_EQUIV``.
STEP_EQUIV: int = config.MD_STEP_EQUIV

#: Re-export MACE_MODEL from config for notebooks that imported it from here.
MACE_MODEL: str = config.MACE_MODEL

__all__ = [
    "scan_database", "load_analysis", "print_database_summary",
    "TIMESTEP_FS", "N_STEPS", "NBLOCK", "TTIME_FS", "PTIME_FS",
    "PRESSURE_GPA", "STEP_SKIP", "STEP_EQUIV", "MACE_MODEL",
]
