"""heatup.manifest
====================
Run-manifest utilities for full result traceability.

Every significant output produced by HeatUp (phonon DOS, VDOS, hull results,
stability reports …) is accompanied by a ``manifest.json`` file that records
the exact configuration used to compute it.  This guarantees:

- **Reproducibility** — re-running with the same manifest recreates the result.
- **Auditability** — comparing two ``manifest.json`` files immediately reveals
  which parameters differed between runs.
- **Error prevention** — if a cached result was produced with a different
  ``PHONON_MODE`` or ``MD_ENSEMBLE``, the pipeline can detect the mismatch
  before using stale data.

Manifest schema
---------------
Every manifest is a JSON object with at minimum these top-level keys:

    "heatup_version"    str   — package version string.
    "timestamp"         str   — ISO-8601 UTC timestamp of the write.
    "calculator"        str   — e.g. "mace-mp:mace-mpa-0-medium".
    "phonon_mode"       str   — "HA" | "QHA" | "VDOS".
    "force_constant_order" int — 2 | 3 (relevant for HA/QHA).
    "phonon_backend"    str   — "ase" | "phonopy".
    "md_ensemble"       str   — "NPT" | "NVT" (relevant for VDOS).
    "md_timestep_fs"    float
    "md_n_steps"        int
    "md_nblock"         int
    "md_step_equiv"     int
    "md_ttime_fs"       float
    "md_ptime_fs"       float  (NPT only)
    "md_pressure_GPa"   float  (NPT only)
    "relax_fmax"        float
    "relax_cell"        bool
    "qha_n_volumes"     int    (QHA only)
    "qha_volume_range"  float  (QHA only)
    "qha_eos"           str    (QHA only)
    "phonon_supercell"  list   (HA/QHA only)
    "phonon_delta"      float  (HA/QHA only)
    "vib_zero_window_mev"  float
    "vib_zero_frac_warn"   float
    "vib_zero_frac_fail"   float
    "thermo_hull_warn_eV"  float
    "competing_phase_sources" list
    "output_path"       str    — path of the file this manifest accompanies.
    "step"              str    — which pipeline step wrote this (e.g. "phonons").

Usage::

    from heatup.manifest import write_manifest, check_manifest_match

    # After writing phonons/dos.json:
    write_manifest(
        output_path = os.path.join(ph_dir, "dos.json"),
        step        = "phonons",
    )

    # Before reusing a cached result:
    ok, diff = check_manifest_match(cached_manifest_path)
    if not ok:
        print(f"Config mismatch: {diff}")
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from heatup import config


def _heatup_version() -> str:
    try:
        from heatup import __version__
        return __version__
    except Exception:
        return "unknown"


def build_manifest(
        output_path: str,
        step: str,
        extra: dict[str, Any] | None = None,
) -> dict:
    """Build a manifest dict capturing the current :mod:`heatup.config` state.

    Args:
        output_path: Absolute path of the file this manifest accompanies.
        step:        Human-readable step name (e.g. ``"phonons"``, ``"vdos"``,
                     ``"hull"``, ``"relaxation"``).
        extra:       Any additional key–value pairs to include (e.g. per-run
                     parameters not in config).

    Returns:
        Manifest dict ready for JSON serialisation.
    """
    m: dict[str, Any] = {
        # Identification
        "heatup_version"    : _heatup_version(),
        "timestamp_utc"     : datetime.now(timezone.utc).isoformat(),
        "step"              : step,
        "output_path"       : os.path.abspath(output_path),

        # Calculator
        "calculator"        : _safe_calc_label(),
        "calc_backend"      : config.CALC_BACKEND,
        "mace_model"        : config.MACE_MODEL,

        # Phonon mode
        "phonon_mode"          : config.PHONON_MODE,
        "force_constant_order" : config.FORCE_CONSTANT_ORDER,
        "phonon_backend"       : config.PHONON_BACKEND,
        "phonon_supercell"     : list(config.PHONON_SUPERCELL),
        "phonon_delta"         : config.PHONON_DELTA,

        # QHA
        "qha_n_volumes"    : config.QHA_N_VOLUMES,
        "qha_volume_range" : config.QHA_VOLUME_RANGE,
        "qha_eos"          : config.QHA_EOS,

        # MD
        "md_ensemble"    : config.MD_ENSEMBLE,
        "md_timestep_fs" : config.MD_TIMESTEP_FS,
        "md_n_steps"     : config.MD_N_STEPS,
        "md_nblock"      : config.MD_NBLOCK,
        "md_step_equiv"  : config.MD_STEP_EQUIV,
        "md_ttime_fs"    : config.MD_TTIME_FS,
        "md_ptime_fs"    : config.MD_PTIME_FS,
        "md_pressure_GPa": config.MD_PRESSURE_GPA,

        # Relaxation
        "relax_fmax"            : config.RELAX_FMAX,
        "relax_cell"            : config.RELAX_CELL,
        "relax_constant_volume" : config.RELAX_CONSTANT_VOLUME,

        # Vibrational stability thresholds
        "vib_zero_window_mev" : config.VIB_ZERO_WINDOW_MEV,
        "vib_zero_frac_warn"  : config.VIB_ZERO_FRAC_WARN,
        "vib_zero_frac_fail"  : config.VIB_ZERO_FRAC_FAIL,
        "vib_min_frames"      : config.VIB_MIN_FRAMES,

        # Thermodynamic stability thresholds
        "thermo_hull_warn_eV"          : config.THERMO_HULL_WARN_EV,
        "thermo_hull_stable_eV"        : config.THERMO_HULL_STABLE_EV,
        "thermo_fe_consistency_thresh" : config.THERMO_FE_CONSISTENCY_THRESHOLD,
        "competing_phase_sources"      : list(config.COMPETING_PHASE_SOURCES),
    }

    if extra:
        m.update(extra)

    return m


def write_manifest(
        output_path: str,
        step: str,
        extra: dict[str, Any] | None = None,
) -> str:
    """Write a ``manifest.json`` file next to *output_path*.

    The manifest is saved as ``<parent_dir>/<output_stem>_manifest.json``,
    or as ``<parent_dir>/manifest.json`` when *output_path* is a directory.

    If ``config.WRITE_MANIFEST`` is False, this function is a no-op.

    Args:
        output_path: Path of the file / directory this manifest accompanies.
        step:        Pipeline step name.
        extra:       Additional key–value pairs to include.

    Returns:
        Path of the written manifest file, or empty string if skipped.
    """
    if not config.WRITE_MANIFEST:
        return ""

    m = build_manifest(output_path, step, extra)

    abs_out = os.path.abspath(output_path)
    if os.path.isdir(abs_out):
        manifest_path = os.path.join(abs_out, "manifest.json")
    else:
        base, _ = os.path.splitext(abs_out)
        manifest_path = base + "_manifest.json"

    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w") as fh:
        json.dump(m, fh, indent=2)
    return manifest_path


def load_manifest(manifest_path: str) -> dict | None:
    """Load and return a manifest dict, or None if the file is absent.

    Args:
        manifest_path: Path to the ``*_manifest.json`` file.

    Returns:
        Parsed manifest dict, or None.
    """
    if not os.path.exists(manifest_path):
        return None
    with open(manifest_path) as fh:
        return json.load(fh)


def check_manifest_match(
        manifest_path: str,
        keys_to_check: list[str] | None = None,
) -> tuple[bool, dict[str, tuple]]:
    """Compare a saved manifest against the current config.

    Useful before reusing a cached result: if the config has changed in a
    way that would invalidate the cache, the caller can decide to recompute.

    Args:
        manifest_path:  Path to the saved ``*_manifest.json``.
        keys_to_check:  Specific keys to compare.  If None, all keys present
                        in both the manifest and the current build are compared
                        (excludes ``"timestamp_utc"`` and ``"output_path"``).

    Returns:
        Tuple ``(match: bool, diffs: dict)``.  *diffs* maps key →
        ``(saved_value, current_value)`` for every key that changed.
    """
    saved = load_manifest(manifest_path)
    if saved is None:
        return False, {"_missing": (None, "manifest file not found")}

    current = build_manifest(
        output_path = saved.get("output_path", ""),
        step        = saved.get("step", ""),
    )

    # Keys that are allowed to differ (metadata, not physics).
    skip = {"timestamp_utc", "output_path", "heatup_version"}

    check = set(keys_to_check) if keys_to_check else (set(saved) & set(current)) - skip

    diffs: dict[str, tuple] = {}
    for key in check:
        sv = saved.get(key)
        cv = current.get(key)
        if sv != cv:
            diffs[key] = (sv, cv)

    return len(diffs) == 0, diffs


def manifest_summary(manifest_path: str) -> str:
    """Return a compact human-readable summary of a saved manifest.

    Args:
        manifest_path: Path to the manifest JSON file.

    Returns:
        Multi-line string suitable for notebook display.
    """
    m = load_manifest(manifest_path)
    if m is None:
        return f"[no manifest at {manifest_path}]"

    lines = [
        f"Step          : {m.get('step', '?')}",
        f"Timestamp     : {m.get('timestamp_utc', '?')}",
        f"HeatUp        : {m.get('heatup_version', '?')}",
        f"Calculator    : {m.get('calculator', '?')}",
        f"Phonon mode   : {m.get('phonon_mode', '?')}  "
        f"(IFC order {m.get('force_constant_order', '?')}, "
        f"backend={m.get('phonon_backend', '?')})",
        f"MD ensemble   : {m.get('md_ensemble', '?')}  "
        f"{m.get('md_n_steps', '?')} steps × {m.get('md_timestep_fs', '?')} fs",
        f"Phonon mode details:",
    ]
    if m.get("phonon_mode") == "QHA":
        lines += [
            f"  QHA volumes : {m.get('qha_n_volumes', '?')}  "
            f"range=±{m.get('qha_volume_range', '?') * 100:.0f}%  "
            f"EOS={m.get('qha_eos', '?')}",
        ]
    lines += [
        f"  Supercell   : {m.get('phonon_supercell', '?')}  "
        f"Δ={m.get('phonon_delta', '?')} Å",
        f"Relax fmax    : {m.get('relax_fmax', '?')} eV/Å  "
        f"relax_cell={m.get('relax_cell', '?')}",
        f"Hull warn     : {m.get('thermo_hull_warn_eV', 0) * 1000:.0f} meV/atom",
        f"Competing srcs: {m.get('competing_phase_sources', '?')}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _safe_calc_label() -> str:
    """Return calculator label without importing MACE (avoids slow import)."""
    try:
        from heatup.calculator import calculator_label
        return calculator_label()
    except Exception:
        return f"{config.CALC_BACKEND}:{config.MACE_MODEL}"
