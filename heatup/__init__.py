"""HeatUp — Temperature-Aware Phase Stability with Generalised Gibbs Free Energies.

Public API
----------
>>> from heatup import run_stability_pipeline, run_stability_pipeline_batch
>>> from heatup import assess_mechanical_stability
>>> from heatup import assess_vibrational_stability
>>> from heatup import assess_thermodynamic_stability
>>> from heatup import plot_stability_report
>>> from heatup.free_energy import GibbsAssembler, build_default_assembler
>>> from heatup.free_energy import (
...     harmonic_f_vib, anharmonic_f_vib,
...     electronic_f_el, magnetic_f_mag,
...     configurational_f_conf, pv_contribution,
... )
"""

from heatup.pipeline import (
    run_stability_pipeline,
    run_stability_pipeline_batch,
)
from heatup.mechanical    import assess_mechanical_stability
from heatup.vibrational   import assess_vibrational_stability
from heatup.thermodynamic import assess_thermodynamic_stability
from heatup.plotting      import plot_stability_report
from heatup import config

__version__ = "0.1.0"
__all__ = [
    "run_stability_pipeline",
    "run_stability_pipeline_batch",
    "assess_mechanical_stability",
    "assess_vibrational_stability",
    "assess_thermodynamic_stability",
    "plot_stability_report",
    "config",
]
