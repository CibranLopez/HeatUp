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
import types
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


def _mace_forward_schema(model: Any) -> str:
    """Return the TorchScript forward schema string for a MACE model, or ''."""
    try:
        return str(model._get_method("forward").schema)
    except Exception:
        pass
    try:
        return str(model.forward.schema)
    except Exception:
        pass
    return ""


class _LegacyMaceWrapper:
    """Callable wrapper that strips unsupported kwargs from TorchScript model calls.

    Older TorchScript MACE models (compiled before ``compute_edge_forces`` was
    added to the schema) raise a ``RuntimeError`` when newer ``MACECalculator``
    versions pass that kwarg.  This wrapper detects the mismatch at construction
    time and silently drops the unsupported argument on every forward call.
    """

    def __init__(self, model: Any) -> None:
        self._model = model
        schema = _mace_forward_schema(model)
        self._strip_edge_forces = "compute_edge_forces" not in schema
        self._strip_atomic_stresses = "compute_atomic_stresses" not in schema

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._strip_edge_forces:
            kwargs.pop("compute_edge_forces", None)
        if self._strip_atomic_stresses:
            kwargs.pop("compute_atomic_stresses", None)
        return self._model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)


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

    if str(device).lower().startswith("cuda"):
        # PyTorch 2.10+ NNC/TensorExpressions kernel fuser tries to compile
        # fused CUDA kernels after a few warmup calls.  When the TorchScript
        # graph contains torch.autograd.grad (used by MACE to compute forces
        # from energy), the fuser's CUDA codegen fails with
        # "Global alloc not supported yet" because it cannot handle dynamic
        # allocations inside the fused region.
        # Disabling GPU fusion globally fixes this with no accuracy impact.
        try:
            import torch
            torch._C._jit_override_can_fuse_on_gpu(False)
        except Exception:
            pass

    calc = mace_mp(
        model        = config.MACE_MODEL,
        device       = device,
        dispersion   = config.MACE_DISPERSION,
        default_dtype= config.MACE_DEFAULT_DTYPE,
    )
    return _wrap_mace_models(calc)


def _wrap_mace_models(calc: Any) -> Any:
    """Wrap models in a MACE calculator for legacy TorchScript compatibility.

    Replaces each entry in ``calc.models`` with a :class:`_LegacyMaceWrapper`
    when the underlying model's TorchScript schema does not include
    ``compute_edge_forces`` (added in newer MACE releases).  Models that
    already support the argument are returned unwrapped.

    Args:
        calc: A ``MACECalculator`` instance.

    Returns:
        The same calculator with ``calc.models`` patched in-place.
    """
    try:
        # Switch to eval mode so the TorchScript forward pass uses
        # create_graph=False inside torch.autograd.grad.  In training mode
        # (the default after torch.jit.load) create_graph=True is passed,
        # which triggers "Global alloc not supported yet" on CUDA with
        # PyTorch >= 2.10.
        for m in calc.models:
            try:
                m.eval()
            except Exception:
                pass
        calc.models = [_LegacyMaceWrapper(m) for m in calc.models]
    except Exception:
        pass  # If models aren't accessible, leave calculator unchanged
    _patch_mace_result_tensors(calc)
    return calc


def _patch_mace_result_tensors(calc: Any) -> None:
    """Patch MACE result tensor assembly for legacy/single-head model outputs.

    Some model + calculator version combinations return a 1D tensor from
    ``atomic_energies_fn`` while the calculator indexes it as if it were 2D,
    which triggers ``IndexError: too many indices for tensor of dimension 1``.
    """
    orig = getattr(calc, "_create_result_tensors", None)
    if orig is None:
        return

    def _patched(self, num_models: int, num_atoms: int, batch: Any, out: dict):
        try:
            return orig(num_models, num_atoms, batch, out)
        except IndexError as exc:
            if "too many indices for tensor of dimension 1" not in str(exc):
                raise

            # Rebuild tensors without node_energy path, then compute node_e0
            # with a shape-aware fallback for single-head models.
            out_fallback = dict(out)
            out_fallback.pop("node_energy", None)
            ret_tensors, node_e0 = orig(num_models, num_atoms, batch, out_fallback)

            try:
                import torch

                if out.get("node_energy") is not None:
                    node_attrs = batch["node_attrs"][:num_atoms]
                    ae = self.models[0].atomic_energies_fn(node_attrs)
                    if getattr(ae, "ndim", 0) == 1:
                        node_e0 = ae.detach().cpu().numpy()
                    else:
                        node_heads = batch["head"][batch["batch"]][:num_atoms]
                        idx = torch.arange(num_atoms, device=node_heads.device)
                        node_e0 = ae[idx, node_heads].detach().cpu().numpy()
            except Exception:
                node_e0 = None

            return ret_tensors, node_e0

    try:
        calc._create_result_tensors = types.MethodType(_patched, calc)
    except Exception:
        pass


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
