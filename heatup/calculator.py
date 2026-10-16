"""heatup.calculator
=====================
The single choke point where an ASE calculator is built.

Backends: ``mace-mp`` (default), ``chgnet``, ``m3gnet``, ``emt`` (unit tests /
Ag-Au-Cu validation without a GPU), ``custom`` (``cfg.custom_calc_factory``).

The per-material model (``cfg.calc.model_overrides``) is resolved with
``build_calculator(material=...)`` - no global mutate-and-restore context
manager as in v0.2.
"""

from __future__ import annotations

import gc
import logging
import types
from typing import Any

from heatup.config import cfg

log = logging.getLogger("heatup")


def default_model(material: str | None = None) -> str:
    """The one model name every HeatUp call (and every caller that imports it) uses for *material*."""
    return cfg.calc.model_for(material)


def resolve_device(device: str | None = None) -> str:
    """``"auto"``/None -> ``"cuda"`` if a CUDA tensor can really be allocated, else ``"cpu"``."""
    device = device or cfg.calc.device
    if device != "auto":
        return device
    try:
        import torch
        if torch.cuda.is_available():
            torch.zeros(1, device="cuda")
            return "cuda"
    except Exception as exc:  # noqa: BLE001
        log.warning("CUDA probe failed (%s) - using CPU", exc)
    return "cpu"


def calculator_label(material: str | None = None, backend: str | None = None) -> str:
    backend = backend or cfg.calc.backend
    if backend == "mace-mp":
        return f"mace-mp:{cfg.calc.model_for(material)}:{cfg.calc.dtype}"
    return backend


def build_calculator(device: str | None = None, material: str | None = None,
                     backend: str | None = None) -> Any:
    backend = backend or cfg.calc.backend
    device = resolve_device(device)
    if backend == "mace-mp":
        return _build_mace_mp(device, cfg.calc.model_for(material))
    if backend == "chgnet":
        return _build_chgnet(device)
    if backend == "m3gnet":
        return _build_m3gnet(device)
    if backend == "emt":
        from ase.calculators.emt import EMT
        return EMT()
    if backend == "custom":
        if cfg.custom_calc_factory is None:
            raise RuntimeError("calc.backend='custom' but cfg.custom_calc_factory is None")
        return cfg.custom_calc_factory(device)
    raise ValueError(f"Unknown calculator backend {backend!r}")


def release_calculator(calc: Any | None) -> None:
    """Best-effort GPU memory release (full release needs a subprocess, see runtime)."""
    if calc is None:
        return
    try:
        del calc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# MACE (+ compatibility patches for legacy TorchScript models)
# ---------------------------------------------------------------------------

def _build_mace_mp(device: str, model: str) -> Any:
    try:
        from mace.calculators import mace_mp
    except ImportError as exc:
        raise ImportError("backend 'mace-mp' needs `pip install mace-torch`") from exc

    if device.startswith("cuda"):
        # PyTorch >= 2.10 NNC fuser fails on MACE's autograd graph
        # ("Global alloc not supported yet"); disabling GPU fusion is exact.
        try:
            import torch
            torch._C._jit_override_can_fuse_on_gpu(False)
        except Exception:  # noqa: BLE001
            pass

    calc = mace_mp(model=model, device=device, dispersion=cfg.calc.dispersion,
                   default_dtype=cfg.calc.dtype)
    return _wrap_mace_models(calc)


def _forward_schema(model: Any) -> str:
    for getter in (lambda m: m._get_method("forward").schema, lambda m: m.forward.schema):
        try:
            return str(getter(model))
        except Exception:  # noqa: BLE001
            continue
    return ""


class _LegacyMaceWrapper:
    """Drop kwargs that an old TorchScript model's forward() does not accept."""

    def __init__(self, model: Any) -> None:
        self._model = model
        schema = _forward_schema(model)
        self._drop = [k for k in ("compute_edge_forces", "compute_atomic_stresses")
                      if schema and k not in schema]

    def __call__(self, *a: Any, **kw: Any) -> Any:
        for k in self._drop:
            kw.pop(k, None)
        return self._model(*a, **kw)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)


def _wrap_mace_models(calc: Any) -> Any:
    try:
        for m in calc.models:
            try:
                m.eval()   # eval mode avoids create_graph=True in the TorchScript graph
            except Exception:  # noqa: BLE001
                pass
        if any(_forward_schema(m) and "compute_edge_forces" not in _forward_schema(m)
               for m in calc.models):
            calc.models = [_LegacyMaceWrapper(m) for m in calc.models]
    except Exception:  # noqa: BLE001
        pass
    _patch_result_tensors(calc)
    return calc


def _patch_result_tensors(calc: Any) -> None:
    """Work around ``IndexError: too many indices for tensor of dimension 1``
    raised by some model/calculator version pairs for single-head models."""
    orig = getattr(calc, "_create_result_tensors", None)
    if orig is None:
        return

    def _patched(self, num_models, num_atoms, batch, out):  # noqa: ANN001
        try:
            return orig(num_models, num_atoms, batch, out)
        except IndexError as exc:
            if "too many indices for tensor of dimension 1" not in str(exc):
                raise
            out2 = dict(out)
            out2.pop("node_energy", None)
            ret, node_e0 = orig(num_models, num_atoms, batch, out2)
            try:
                import torch
                if out.get("node_energy") is not None:
                    ae = self.models[0].atomic_energies_fn(batch["node_attrs"][:num_atoms])
                    if getattr(ae, "ndim", 0) == 1:
                        node_e0 = ae.detach().cpu().numpy()
                    else:
                        heads = batch["head"][batch["batch"]][:num_atoms]
                        idx = torch.arange(num_atoms, device=heads.device)
                        node_e0 = ae[idx, heads].detach().cpu().numpy()
            except Exception:  # noqa: BLE001
                node_e0 = None
            return ret, node_e0

    try:
        calc._create_result_tensors = types.MethodType(_patched, calc)
    except Exception:  # noqa: BLE001
        pass


def _build_chgnet(device: str) -> Any:
    try:
        from chgnet.model import CHGNet
        from chgnet.model.dynamics import CHGNetCalculator
    except ImportError as exc:
        raise ImportError("backend 'chgnet' needs `pip install chgnet`") from exc
    return CHGNetCalculator(model=CHGNet.load(), use_device=device)


def _build_m3gnet(device: str) -> Any:
    try:
        import matgl
        from matgl.ext.ase import PESCalculator
    except ImportError as exc:
        raise ImportError("backend 'm3gnet' needs `pip install matgl`") from exc
    return PESCalculator(matgl.load_model("M3GNet-MP-2021.2.8-PES"))
