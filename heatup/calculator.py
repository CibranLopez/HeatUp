"""heatup.calculator
=====================
Pluggable ASE calculator factory.

All MACE / CHGNet / M3GNet / custom calculator construction goes through
:func:`build_calculator`.  Nothing else in HeatUp ever calls ``mace_mp``
directly — this is the single choke point that makes backend migration trivial.

Usage::

    from heatup.calculator import build_calculator
    calc = build_calculator(device="cuda")
    atoms.calc = calc

To switch backend across the whole library::

    import heatup.config as cfg
    cfg.CALC_BACKEND = "chgnet"

To use a completely custom calculator::

    cfg.CALC_BACKEND        = "custom"
    cfg.CUSTOM_CALC_FACTORY = lambda device: MyCalc(device=device)

Backends
--------
"mace-mp"
    MACE-MP universal potential (Batatia et al. 2023).
    Requires ``mace-torch``.  Model file / name set by ``cfg.MACE_MODEL``.
    D3 dispersion controlled by ``cfg.MACE_DISPERSION``.

"chgnet"
    CHGNet universal potential (Deng et al. 2023).
    Requires ``chgnet``.  Model version follows the installed package.

"m3gnet"
    M3GNet universal potential (Chen & Ong 2022).
    Requires ``matgl``.

"custom"
    Arbitrary ASE-compatible calculator returned by ``cfg.CUSTOM_CALC_FACTORY``.
    Factory signature: ``(device: str) -> ase.calculators.calculator.Calculator``.

GPU memory management
---------------------
:func:`release_calculator` performs a best-effort GPU memory release
(``del calc``, ``gc.collect()``, ``torch.cuda.empty_cache()``).  For full
isolation between materials when running on CUDA, use the subprocess wrappers
in :mod:`heatup.structure_pipeline` and :mod:`heatup.md_pipeline`, which
exit the child process and let the OS reclaim CUDA context entirely.
"""

from __future__ import annotations

import gc
from typing import Any

from heatup import config


def build_calculator(
        device: str | None = None,
        backend: str | None = None,
) -> Any:
    """Build and return an ASE-compatible calculator.

    All parameters have module-level defaults in :mod:`heatup.config` so the
    call ``build_calculator()`` is always valid without any arguments.

    Args:
        device:  Compute device string, e.g. ``'cpu'`` or ``'cuda'``.
                 Defaults to ``config.DEFAULT_DEVICE``.
        backend: Calculator backend name.  Defaults to ``config.CALC_BACKEND``.
                 Supported: ``"mace-mp"``, ``"chgnet"``, ``"m3gnet"``,
                 ``"custom"``.

    Returns:
        A configured ASE calculator instance.

    Raises:
        ValueError: If the requested backend is unknown.
        ImportError: If the required package for the backend is not installed.
        RuntimeError: If ``CALC_BACKEND == "custom"`` but
                      ``CUSTOM_CALC_FACTORY`` is None.
    """
    if device  is None:
        device  = config.DEFAULT_DEVICE
    if backend is None:
        backend = config.CALC_BACKEND

    if backend == "mace-mp":
        return _build_mace_mp(device)
    if backend == "chgnet":
        return _build_chgnet(device)
    if backend == "m3gnet":
        return _build_m3gnet(device)
    if backend == "custom":
        return _build_custom(device)

    raise ValueError(
        f"Unknown calculator backend: {backend!r}.  "
        f"Supported: 'mace-mp', 'chgnet', 'm3gnet', 'custom'."
    )


def calculator_label(backend: str | None = None) -> str:
    """Return a short human-readable label for the active calculator.

    Useful for labelling output files and JSON records so they remain
    self-documenting when the backend is changed.

    Args:
        backend: Override ``config.CALC_BACKEND`` for this call.

    Returns:
        A compact string such as ``"mace-mp:mace-mpa-0-medium"`` or
        ``"chgnet:0.3.0"``.
    """
    if backend is None:
        backend = config.CALC_BACKEND

    if backend == "mace-mp":
        return f"mace-mp:{config.MACE_MODEL}"
    if backend == "chgnet":
        try:
            from chgnet.model import CHGNet as _C
            return f"chgnet:{_C().version}"
        except Exception:
            return "chgnet:unknown"
    if backend == "m3gnet":
        try:
            import matgl as _m
            return f"m3gnet:{_m.__version__}"
        except Exception:
            return "m3gnet:unknown"
    if backend == "custom":
        return "custom"
    return backend


def release_calculator(calc: Any | None) -> None:
    """Best-effort GPU memory release for a calculator object.

    For true isolation between CUDA runs, prefer subprocess execution.
    This function is a lightweight fallback for CPU or single-run scenarios.

    Args:
        calc: Calculator instance to release.  Passing ``None`` is safe.
    """
    if calc is None:
        return
    try:
        import torch
        del calc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Private backend constructors
# ---------------------------------------------------------------------------

def _build_mace_mp(device: str) -> Any:
    """Build a MACE-MP calculator.

    Deferred import avoids slow MACE load at module import time — MACE
    initialises its model registry on import (~1 s on first call).

    Args:
        device: Compute device string.

    Returns:
        Configured :class:`mace.calculators.MACECalculator`.
    """
    try:
        from mace.calculators import mace_mp
    except ImportError as exc:
        raise ImportError(
            "The 'mace-torch' package is required for the 'mace-mp' backend. "
            "Install it with:  pip install mace-torch"
        ) from exc

    return mace_mp(
        model        = config.MACE_MODEL,
        device       = device,
        dispersion   = config.MACE_DISPERSION,
        default_dtype= config.MACE_DEFAULT_DTYPE,
    )


def _build_chgnet(device: str) -> Any:
    """Build a CHGNet calculator.

    Args:
        device: Compute device string.

    Returns:
        :class:`chgnet.model.dynamics.CHGNetCalculator`.
    """
    try:
        from chgnet.model import CHGNet
        from chgnet.model.dynamics import CHGNetCalculator
    except ImportError as exc:
        raise ImportError(
            "The 'chgnet' package is required for the 'chgnet' backend. "
            "Install it with:  pip install chgnet"
        ) from exc

    model = CHGNet.load()
    return CHGNetCalculator(model=model, use_device=device)


def _build_m3gnet(device: str) -> Any:
    """Build an M3GNet calculator via the matgl package.

    Args:
        device: Compute device string.

    Returns:
        :class:`matgl.ext.ase.M3GNetCalculator`.
    """
    try:
        import matgl
        from matgl.ext.ase import M3GNetCalculator
    except ImportError as exc:
        raise ImportError(
            "The 'matgl' package is required for the 'm3gnet' backend. "
            "Install it with:  pip install matgl"
        ) from exc

    pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
    return M3GNetCalculator(potential=pot, stress_weight=1.0)


def _build_custom(device: str) -> Any:
    """Invoke the user-supplied factory to build a custom calculator.

    Args:
        device: Passed verbatim to ``config.CUSTOM_CALC_FACTORY``.

    Raises:
        RuntimeError: If ``config.CUSTOM_CALC_FACTORY`` is None.
    """
    factory = config.CUSTOM_CALC_FACTORY
    if factory is None:
        raise RuntimeError(
            "cfg.CALC_BACKEND == 'custom' but cfg.CUSTOM_CALC_FACTORY is None. "
            "Set it to a callable that accepts a device string and returns an "
            "ASE calculator:\n"
            "    cfg.CUSTOM_CALC_FACTORY = lambda device: MyCalc(device=device)"
        )
    return factory(device)
