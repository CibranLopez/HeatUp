"""HeatUp - temperature-aware stability screening with machine-learning potentials.

Three stages, three notebooks, one folder tree (see ``heatup.core.layout``)::

    prepare   heatup.prepare   structures -> database, competing phases, plan.json per phase  (notebook 1, no calculator)
    compute   heatup.compute   run what the plans say with an interatomic potential  (notebook 2, optional)
    analyze   heatup.analyze   analysis, gates, hull, reports - reads files only      (notebook 3, no calculator)

``heatup.screen`` composes the three for one phase (the entry point for other software).  Everything heavy is
imported lazily: ``import heatup`` pulls in neither torch nor phonopy, and ``heatup.analyze`` never imports a
calculator.
"""

__version__ = "0.4.0"

import importlib
import logging
import sys

from heatup.core.config import cfg, load_json, override, reset  # noqa: E402,F401
from heatup.core.layout import PhaseDir  # noqa: E402,F401

def verbose(level: int = logging.INFO) -> None:
    """Switch on console logging (call once in a notebook)."""
    logger = logging.getLogger("heatup")
    logger.setLevel(level)
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logger.handlers):
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(h)


#: public name -> module that defines it (resolved on first access)
_LAZY = {
    # prepare
    "prepare_project": "heatup.prepare.plan", "preview_plan": "heatup.prepare.plan", "save_project": "heatup.prepare.plan",
    "write_phase": "heatup.prepare.phases", "locate_phase": "heatup.prepare.phases",
    "export_inputs": "heatup.prepare.export", "build_md_cell": "heatup.prepare.export", "collect_ha": "heatup.compute.phonons", "collect_elastic": "heatup.compute.static",
    # compute
    "run_plan": "heatup.compute.execute", "run_step": "heatup.compute.runtime",
    "build_calculator": "heatup.compute.calculator", "default_model": "heatup.compute.calculator",
    "relax_atoms": "heatup.compute.static", "elastic_tensor": "heatup.compute.static", "moduli": "heatup.compute.static",
    # analyze
    "status": "heatup.analyze.evaluate", "run_analysis": "heatup.analyze.evaluate",
    "evaluate": "heatup.analyze.evaluate", "evaluate_all": "heatup.analyze.evaluate",
    "summary_rows": "heatup.analyze.evaluate",
    "assess_mechanical_stability": "heatup.analyze.gates", "assess_structural_transition": "heatup.analyze.gates",
    "assess_vibrational_stability": "heatup.analyze.gates",
    "assess_thermodynamic_stability": "heatup.analyze.hull", "free_energy": "heatup.analyze.thermo",
    # facade
    "screen": "heatup.screen", "ScreenOptions": "heatup.screen", "ScreenResult": "heatup.screen",
    "GateResult": "heatup.screen", "screen_many": "heatup.screen", "generate_and_screen": "heatup.screen",
}
__all__ = ["__version__", "cfg", "load_json", "override", "reset", "verbose", "PhaseDir", *_LAZY]


def __getattr__(name: str):
    if name in _LAZY:
        value = getattr(importlib.import_module(_LAZY[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'heatup' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
