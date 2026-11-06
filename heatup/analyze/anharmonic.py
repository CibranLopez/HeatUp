"""heatup.analyze.anharmonic  --  ``phonons/anharmonic/``: the phonons that come from MD, and where they come from
================================================================================================================
ANHARMONIC level (``heatup.core.provenance``): at every MD temperature the velocity autocorrelation gives an effective-harmonic
spectrum (frequencies renormalised by the anharmonicity of the true potential at that temperature and cell).

Like the other levels, everything of this one is in its own folder, ``phonons/anharmonic/``.  Unlike them it has NO calculation of
its own - the trajectories are the ``md/<T>K`` runs - so the folder holds the post-processing only, and says so::

    phonons/anharmonic/
        source.json          what this level is and which md folders it reads (written as soon as the MD is planned)
        anharmonic.json      summary per temperature (below)
        <T>K/vdos.json       the VDOS of md/<T>K (analysis step ``vdos``)          <T>K/vdos.png   its figure

``anharmonic.json``::

    {"meta": {...},                                    # what approximation this is, see provenance.KINDS["anharmonic"]
     "source": {"folder": "md", "runs": {"300": "md/300K/output.traj", ...}},
     "runs": [{"temperature_K", "mean_omega_meV", "mean_ln_omega", "imag_fraction", "resolution_meV", "production_ps",
               "T_kin_mean_K", "vdos": "phonons/anharmonic/<T>K/vdos.json", "source": "md/<T>K/output.traj"}, ...]}

``mean_ln_omega`` is the quantity the classical vibrational free energy depends on (F_cl = 3 kT <ln(hw/kT)>).
Written by ``evaluate.run_analysis`` after the per-temperature analysis steps (:func:`collect_md_phonons`).
"""
from __future__ import annotations

import logging
import os

import numpy as np

from heatup.analyze.trajectory import load_vdos
from heatup.core import ledger, provenance
from heatup.core.config import cfg
from heatup.core.layout import PhaseDir, write_json

log = logging.getLogger("heatup")


def write_source_marker(phase_dir, temperatures=None) -> str:
    """``phonons/anharmonic/source.json``: this level has no calculation, its phonons are taken from ``md/<T>K``."""
    phase = PhaseDir.of(phase_dir)
    temps = sorted(int(round(t)) for t in (temperatures if temperatures is not None else ledger.md_temperatures(phase)))
    path = os.path.join(phase.anharmonic_dir, "source.json")
    write_json(path, {
        "kind": "anharmonic", "calculation": "none",
        "note": "no calculation in this folder: the phonons are taken from the MD runs of the md/ folder "
                "(velocity autocorrelation -> effective-harmonic VDOS); this folder holds the post-processing only",
        "source_folder": "md", "runs": {str(T): f"md/{T}K/output.traj" for T in temps},
        "meta": provenance.stamp("anharmonic")})
    return path


def collect_md_phonons(phase_dir) -> dict | None:
    """Write ``phonons/anharmonic/anharmonic.json`` (+ ``source.json``) from the finished VDOS analyses; None if there is none."""
    phase = PhaseDir.of(phase_dir)
    runs = []
    for T in ledger.md_temperatures(phase):
        v = load_vdos(phase, T)
        if not v:
            continue
        o, w = v["modes"].physical()
        runs.append({"temperature_K": float(T),
                     "mean_omega_meV": float((w * o).sum() / w.sum()),
                     "mean_ln_omega": float((w * np.log(o)).sum() / w.sum()),     # omega in meV
                     "imag_fraction": float(v["modes"].imag_fraction(cfg.harmonic.imag_tol_meV)),
                     "resolution_meV": v.get("resolution_meV"), "production_ps": v.get("production_ps"),
                     "T_kin_mean_K": v.get("T_kin_mean_K"),
                     "vdos": f"phonons/anharmonic/{int(T)}K/vdos.json", "source": f"md/{int(T)}K/output.traj"})
    if not runs:
        return None
    write_source_marker(phase, [r["temperature_K"] for r in runs])
    res = {"meta": provenance.stamp("anharmonic"),
           "source": {"folder": "md", "runs": {str(int(r["temperature_K"])): r["source"] for r in runs}}, "runs": runs}
    write_json(os.path.join(phase.anharmonic_dir, "anharmonic.json"), res)
    return res
