"""HeatUp — Temperature-Aware Phase Stability.

HeatUp predicts whether a crystal structure is thermodynamically, mechanically,
and vibrationally stable at finite temperature by combining a machine-learning
interatomic potential (MACE-MP by default) with three sequential stability gates.

Quick start::

    from heatup import run_stability_pipeline
    report = run_stability_pipeline(
        sym_dir     = "database/LGPS/P42-nmc",
        operating_T = 800.0,
    )
    print(report["overall"])   # "ok" | "warn" | "fail"

Phonon modes
------------
The vibrational free energy can be computed three ways.  Set before running::

    import heatup.config as cfg

    cfg.PHONON_MODE = "HA"    # harmonic approximation (finite displacement)
    cfg.PHONON_MODE = "QHA"   # quasi-harmonic (phonons at multiple volumes)
    cfg.PHONON_MODE = "VDOS"  # anharmonic VDOS from AIMD (default)

Calculator backends
-------------------
Any ASE-compatible calculator can be used::

    cfg.CALC_BACKEND = "mace-mp"   # default
    cfg.CALC_BACKEND = "chgnet"
    cfg.CALC_BACKEND = "m3gnet"
    cfg.CALC_BACKEND = "custom"
    cfg.CUSTOM_CALC_FACTORY = lambda device: MyCalc(device=device)

Public API
----------
"""

from heatup.pipeline    import run_stability_pipeline, run_stability_pipeline_batch
from heatup.mechanical  import assess_mechanical_stability
from heatup.vibrational import assess_vibrational_stability
from heatup.thermodynamic import assess_thermodynamic_stability
from heatup.plotting    import plot_stability_report
from heatup.phonons     import run_phonons, get_free_energy, run_vdos_for_sim
from heatup.calculator  import build_calculator, release_calculator, calculator_label
from heatup.manifest    import write_manifest, load_manifest, check_manifest_match, manifest_summary
from heatup import config

__version__ = "0.2.0"

__all__ = [
    # Pipeline
    "run_stability_pipeline",
    "run_stability_pipeline_batch",
    # Gates
    "assess_mechanical_stability",
    "assess_vibrational_stability",
    "assess_thermodynamic_stability",
    # Plotting
    "plot_stability_report",
    # Phonons (mode-agnostic)
    "run_phonons",
    "get_free_energy",
    "run_vdos_for_sim",
    # Calculator factory
    "build_calculator",
    "release_calculator",
    "calculator_label",
    # Manifest / traceability
    "write_manifest",
    "load_manifest",
    "check_manifest_match",
    "manifest_summary",
    # Config
    "config",
]
