"""ThermoPhase — Temperature-Aware Phase Stability with Generalised Gibbs Free Energies.

Public API
----------
>>> from thermophasepy import run_stability_pipeline, run_stability_pipeline_batch
>>> from thermophasepy import assess_mechanical_stability
>>> from thermophasepy import assess_vibrational_stability
>>> from thermophasepy import assess_thermodynamic_stability
>>> from thermophasepy import plot_stability_report
>>> from thermophasepy.free_energy import GibbsAssembler, build_default_assembler
>>> from thermophasepy.free_energy import (
...     harmonic_f_vib, anharmonic_f_vib,
...     electronic_f_el, magnetic_f_mag,
...     configurational_f_conf, pv_contribution,
... )
"""

from thermophasepy.pipeline import (
    run_stability_pipeline,
    run_stability_pipeline_batch,
)
from thermophasepy.mechanical    import assess_mechanical_stability
from thermophasepy.vibrational   import assess_vibrational_stability
from thermophasepy.thermodynamic import assess_thermodynamic_stability
from thermophasepy.plotting      import plot_stability_report
from thermophasepy import config

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
