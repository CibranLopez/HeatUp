"""heatup.runtime
==================
Step registry, CUDA-isolated subprocess execution and logging.

Every expensive step is a plain function ``fn(phase_dir, device, force, **kw)``
that **raises** on failure.  :func:`run_step` executes it

* in a child process (default; the OS then reclaims all CUDA memory - del /
  gc / empty_cache do not), forwarding the whole config as one JSON env var;
* retrying on CPU when the child exits with :data:`EXIT_RETRY_CPU` (75, a
  *known* CUDA runtime failure) - and **only** then.  v0.2 retried on CPU after
  *any* non-zero exit, which re-ran genuine bugs (and wasted hours) before
  reporting them.

Child entry point: ``python -m heatup._worker <step> <phase_dir> [--device D]
[--force] [--kwargs JSON]``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import subprocess
import sys
import traceback
from typing import Any

from heatup import config
from heatup.calculator import resolve_device
from heatup.paths import PhaseDir

log = logging.getLogger("heatup")

EXIT_OK, EXIT_FAIL, EXIT_RETRY_CPU = 0, 1, 75

#: step name -> "module:function"
STEPS: dict[str, str] = {
    "relax":    "heatup.relax:run_relaxation",
    "phonons":  "heatup.phonons:run_ha",
    "qha":      "heatup.phonons:run_qha",
    "fc3":      "heatup.fc3:run_fc3",
    "elastic":  "heatup.elastic:run_elastic",
    "md":       "heatup.md:run_md",
}


def verbose(level: int = logging.INFO) -> None:
    """Switch on console logging (call once in a notebook)."""
    logger = logging.getLogger("heatup")
    logger.setLevel(level)
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in logger.handlers):
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(h)


def is_retryable_cuda_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return (
        "global alloc not supported yet" in msg
        or "nvml_success" in msg
        or "cudacachingallocator" in msg
        or ("cuda" in msg and "out of memory" in msg)
        or "cudaerrormemoryallocation" in msg
        or ("cuda" in msg and "no kernel image" in msg)
    )


def _resolve(step: str):
    mod, fn = STEPS[step].split(":")
    return getattr(importlib.import_module(mod), fn)


def _call(step: str, phase: PhaseDir, device: str, force: bool, kwargs: dict) -> None:
    _resolve(step)(phase, device=device, force=force, **kwargs)


def _python() -> str:
    return os.environ.get("HEATUP_PYTHON", sys.executable)


def _child_env(material: str | None) -> dict[str, str]:
    env = os.environ.copy()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.update(config.to_env(material))
    return env


def run_step(step: str, phase_dir: "str | PhaseDir", device: str | None = None,
             force: bool | None = None, isolated: bool = True, **kwargs: Any) -> bool:
    """Run one step; returns True on success. Never raises for compute errors.

    ``force=None`` -> the step is recomputed iff it is listed in ``cfg.run.force`` (or "all");
    ``True``/``False`` override the config for this call.
    """
    phase = PhaseDir.of(phase_dir)
    force = config.forced(step, force)
    device = resolve_device(device)
    tag = f"{phase.tag}:{step}"

    if not isolated:
        try:
            _call(step, phase, device, force, kwargs)
            return True
        except Exception as exc:  # noqa: BLE001
            if device.startswith("cuda") and is_retryable_cuda_error(exc):
                log.warning("[%s] CUDA failure (%s) - retrying on CPU", tag, exc)
                try:
                    _call(step, phase, "cpu", True, kwargs)
                    return True
                except Exception as exc2:  # noqa: BLE001
                    log.error("[%s] failed: %s", tag, exc2)
                    log.debug(traceback.format_exc())
                    return False
            log.error("[%s] failed: %s", tag, exc)
            log.debug(traceback.format_exc())
            return False

    def _spawn(dev: str, frc: bool) -> int:
        cmd = [_python(), "-m", "heatup._worker", step, phase.path, "--device", dev,
               "--kwargs", json.dumps(kwargs)]
        if frc:
            cmd.append("--force")
        return subprocess.run(cmd, env=_child_env(phase.material), check=False).returncode

    rc = _spawn(device, force)
    if rc == EXIT_RETRY_CPU and device.startswith("cuda"):
        log.warning("[%s] CUDA runtime failure - retrying on CPU", tag)
        rc = _spawn("cpu", True)
    if rc != EXIT_OK:
        log.error("[%s] subprocess exited with code %d", tag, rc)
    return rc == EXIT_OK


def main(argv: list[str] | None = None) -> int:
    config.load_from_env()
    verbose()
    ap = argparse.ArgumentParser(prog="heatup.runtime")
    ap.add_argument("step", choices=sorted(STEPS))
    ap.add_argument("phase_dir")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--kwargs", default="{}")
    a = ap.parse_args(argv)
    try:
        _call(a.step, PhaseDir(a.phase_dir), a.device, a.force, json.loads(a.kwargs))
        return EXIT_OK
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        if a.device.startswith("cuda") and is_retryable_cuda_error(exc):
            return EXIT_RETRY_CPU
        return EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
