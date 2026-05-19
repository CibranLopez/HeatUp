"""heatup.md_pipeline
======================
NPT / NVT molecular-dynamics simulation and trajectory analysis.

All computation goes through :func:`run_md_simulation`, which:

1. Copies the supercell ``POSCAR`` into the temperature sub-folder.
2. Writes ``simulation-input.json`` (all parameters, before any computation).
3. Runs the MD trajectory with the configured ensemble (NPT or NVT) and
   calculator backend (see :mod:`heatup.config` and :mod:`heatup.calculator`).
4. Analyses the trajectory (thermodynamic convergence, diffusivity, MSD).

Database layout::

    database/
      <material>/
        <symmetry>/
          POSCAR                     ← original structure, never modified
          aimd/
            POSCAR                   ← upper-triangular supercell
            <T>K/
              simulation-input.json  ← all parameters, written before MD
              POSCAR                 ← copy of aimd/POSCAR for this run
              output.traj            ← ASE trajectory (one frame per NBLOCK steps)
              npt.log  / nvt.log     ← thermostat / barostat log
              CONTCAR                ← final frame in VASP format
              analysis.json          ← diffusion, thermo stats (written after MD)

All fixed simulation and analysis parameters are read from
:mod:`heatup.config`; no module-level constants are redefined here.

MD ensembles
------------
Controlled by ``config.MD_ENSEMBLE``:

NPT (default)
    Uses :class:`ase.md.npt.NPT` (Martyna-Tobias-Klein).  Cell shape and
    volume fluctuate.  Parameters: timestep, ttime (thermostat), ptime
    (barostat), externalstress (pressure).

NVT
    Uses :class:`ase.md.nvtberendsen.NVTBerendsen` or
    :class:`ase.md.langevin.Langevin` (Nosé-Hoover chain via
    :class:`ase.md.npt.NPT` with pfactor=None).  Cell fixed.

GPU memory isolation
--------------------
:func:`run_md_subprocess` spawns a child process; the OS releases the CUDA
context on exit.  Use it for batch runs.  :func:`run_md_simulation` runs
in the current process and attempts a best-effort release via
:func:`~heatup.calculator.release_calculator`.

Typical usage::

    from heatup.md_pipeline import run_md_simulation, run_md_subprocess

    # Single run in current process (CPU or single-material GPU):
    run_md_simulation("database/AgI/P6_3mc", temperature=600, device="cpu")

    # CUDA-isolated subprocess (recommended for GPU batch runs):
    run_md_subprocess("database/AgI/P6_3mc", temperature=600, device="cuda")
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ase import units
from ase.io import read as ase_read
from ase.io.trajectory import Trajectory
from ase.io.vasp import read_vasp, write_vasp
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from pymatgen.analysis.diffusion.analyzer import DiffusionAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor

from heatup import config
from heatup.calculator import build_calculator, release_calculator, calculator_label


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _sim_dir(sym_dir: str, temperature: float) -> str:
    """Return the canonical simulation sub-folder path.

    Returns ``<sym_dir>/aimd/<int(temperature)>K``.
    """
    return os.path.join(sym_dir, "aimd", f"{int(temperature)}K")


def _tag(sym_dir: str, temperature: float | None = None) -> str:
    symmetry = os.path.basename(os.path.abspath(sym_dir))
    material = os.path.basename(os.path.dirname(os.path.abspath(sym_dir)))
    tag = f"{material}/{symmetry}"
    if temperature is not None:
        tag += f"/{int(temperature)}K"
    return tag


def _cuda_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    return env


def _build_md_params(
        material: str,
        symmetry: str,
        temperature: float,
        device: str,
) -> dict:
    """Assemble the parameter record written to ``simulation-input.json``."""
    return {
        "material"     : material,
        "symmetry"     : symmetry,
        "temperature_K": float(temperature),
        "calculator"   : calculator_label(),
        "device"       : device,
        "ensemble"     : config.MD_ENSEMBLE,
        "timestep_fs"  : config.MD_TIMESTEP_FS,
        "n_steps"      : config.MD_N_STEPS,
        "ttime_fs"     : config.MD_TTIME_FS,
        "ptime_fs"     : config.MD_PTIME_FS,     # NPT only
        "pressure_GPa" : config.MD_PRESSURE_GPA, # NPT only
        "nblock"       : config.MD_NBLOCK,
        "step_skip"    : config.MD_STEP_SKIP,
        "step_equiv"   : config.MD_STEP_EQUIV,
    }


# ---------------------------------------------------------------------------
# Trajectory loading
# ---------------------------------------------------------------------------

def _load_trajectory(
        sim_dir: str,
        step_skip: int,
        step_equiv: int,
        traj_timestep_fs: float,
) -> tuple[list, list, int, int]:
    """Load the trajectory and split into equilibration and production parts.

    Args:
        sim_dir:          Path to the temperature sub-folder.
        step_skip:        Read stride (1 = every frame).
        step_equiv:       Number of frames treated as equilibration.
        traj_timestep_fs: Effective time between read frames in fs.

    Returns:
        Tuple ``(equil_frames, prod_frames, n_prod, n_atoms)``.

    Raises:
        FileNotFoundError: If the trajectory file is absent.
        ValueError:        If the file is empty or unreadable.
    """
    traj_path = os.path.join(sim_dir, "output.traj")
    if not os.path.exists(traj_path):
        raise FileNotFoundError(f"Trajectory not found: {traj_path}")
    if os.path.getsize(traj_path) == 0:
        raise ValueError(
            f"Trajectory is empty (0 bytes): {traj_path}.  "
            "The MD run was interrupted before any frames were written."
        )

    try:
        full = ase_read(traj_path, index=f"::{step_skip}")
    except Exception as exc:
        raise ValueError(f"Cannot read trajectory {traj_path}: {exc}") from exc

    print(f"  Loaded {len(full)} frames (stride={step_skip}).")
    n_total = len(full)
    equil   = full[:step_equiv]
    prod    = full[step_equiv:]
    n_prod  = len(prod)
    n_atoms = len(prod[0]) if n_prod else 0

    total_ps = n_total * traj_timestep_fs / 1000.0
    equil_ps = step_equiv * traj_timestep_fs / 1000.0
    prod_ps  = n_prod * traj_timestep_fs / 1000.0
    print(f"  Total {total_ps:.1f} ps | Equil {equil_ps:.1f} ps | Prod {prod_ps:.1f} ps")
    return equil, prod, n_prod, n_atoms


# ---------------------------------------------------------------------------
# Thermodynamic observables
# ---------------------------------------------------------------------------

def _extract_thermo(frames: list, traj_timestep_fs: float,
                    time_offset: float = 0.0) -> tuple:
    """Extract temperature, pressure, and volume time series.

    Args:
        frames:           List of ASE Atoms objects.
        traj_timestep_fs: Effective time between frames in fs.
        time_offset:      Starting time in ps (for labelling production frames).

    Returns:
        Tuple ``(times_ps, temperatures_K, pressures_GPa, volumes_A3)``.
    """
    times = np.array([i * traj_timestep_fs / 1000.0 + time_offset
                      for i in range(len(frames))])
    T_K   = np.array([a.get_temperature()       for a in frames])
    P_GPa = np.array([-np.trace(a.get_stress()) / 3.0 * 160.21766208
                      for a in frames])   # eV/Å³ → GPa, isotropic component
    V_A3  = np.array([a.get_volume()            for a in frames])
    return times, T_K, P_GPa, V_A3


def _plot_convergence(
        equil_t: np.ndarray, prod_t: np.ndarray,
        equil_y: np.ndarray, prod_y: np.ndarray,
        name: str, ylabel: str, out_path: str,
) -> None:
    """Save a thermodynamic convergence plot (equilibration + production).

    Args:
        equil_t:  Time axis for equilibration frames (ps).
        prod_t:   Time axis for production frames (ps).
        equil_y:  Observable during equilibration (shown faded).
        prod_y:   Observable during production (mean shown as dashed line).
        name:     Observable name (``"temperature"``/``"pressure"``/``"volume"``).
        ylabel:   Y-axis label string.
        out_path: Full output path for the PDF.
    """
    palette = {"temperature": "tab:green", "pressure": "tab:blue",
               "volume": "tab:purple"}
    color = palette.get(name, "k")
    mean  = float(np.mean(prod_y))

    plt.figure(figsize=(7, 3))
    plt.plot(equil_t, equil_y, color=color, alpha=0.35, label="Equilibration")
    plt.plot(prod_t,  prod_y,  color=color, alpha=1.0,  label="Production")
    plt.axhline(mean, ls="--", color="k", lw=1, label=f"Mean = {mean:.3g}")
    plt.xlabel("t (ps)"); plt.ylabel(ylabel)
    plt.legend(fontsize=7); plt.grid(alpha=0.4); plt.tight_layout()
    plt.savefig(out_path, dpi=80, bbox_inches="tight")
    plt.close()


def _plot_msd(diffusion_results: dict, out_path: str) -> None:
    """Save MSD vs. time for all species.

    Args:
        diffusion_results: Mapping of element symbol → result dict from
            :func:`_compute_diffusivity_one_species`.
        out_path: Full output path for the PDF.
    """
    plt.figure(figsize=(7, 4))
    for sp, vals in diffusion_results.items():
        plt.plot(vals["times_ps"], vals["msd"], label=sp)
    plt.xlabel("Time (ps)"); plt.ylabel("MSD (Å²)")
    plt.xlim(left=0); plt.ylim(bottom=0)
    plt.legend(); plt.grid(True, alpha=0.4); plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Diffusion analysis
# ---------------------------------------------------------------------------

def _compute_diffusivity_one_species(
        structures: list,
        species: str,
        temperature: float,
        traj_timestep_fs: float,
) -> dict:
    """Compute diffusivity and ionic conductivity for one element.

    Uses :class:`pymatgen.analysis.diffusion.analyzer.DiffusionAnalyzer`
    with the mean-squared displacement (MSD) method.

    Args:
        structures:       Pymatgen Structure objects for the **production** window
                          only (equilibration must be stripped by the caller).
        species:          Element symbol, e.g. ``"Li"``.
        temperature:      MD temperature in Kelvin.
        traj_timestep_fs: Effective time between frames in fs.

    Returns:
        Dict with keys ``diffusivity_cm2s``, ``diffusivity_cm2s_std``,
        ``conductivity_mScm``, ``conductivity_mScm_std``,
        ``msd`` (list of floats, Å²), ``times_ps`` (list of floats).
    """
    analyzer = DiffusionAnalyzer.from_structures(
        structures=structures,
        specie=species,
        temperature=temperature,
        time_step=traj_timestep_fs,
        step_skip=1,
        smoothed="max",
    )
    msd      = analyzer.msd
    times_ps = np.arange(len(msd)) * traj_timestep_fs / 1000.0
    return {
        "diffusivity_cm2s"      : float(analyzer.diffusivity),
        "diffusivity_cm2s_std"  : float(analyzer.diffusivity_std_dev),
        "conductivity_mScm"     : float(analyzer.conductivity),
        "conductivity_mScm_std" : float(analyzer.conductivity_std_dev),
        "msd"                   : msd.tolist(),
        "times_ps"              : times_ps.tolist(),
    }


# ---------------------------------------------------------------------------
# Trajectory analysis
# ---------------------------------------------------------------------------

def analyse_simulation(sim_dir: str, force: bool = False) -> dict | None:
    """Analyse a completed trajectory and write ``analysis.json``.

    All parameters are read from ``simulation-input.json`` inside *sim_dir*
    so results are fully reproducible from the stored record alone.

    Writes::

        <sim_dir>/analysis.json
        <sim_dir>/temperature-convergence.pdf
        <sim_dir>/pressure-convergence.pdf
        <sim_dir>/volume-convergence.pdf
        <sim_dir>/diffusion.pdf

    Args:
        sim_dir: Path to a temperature sub-folder (e.g. ``database/AgI/P6_3mc/aimd/600K``).
        force:   Re-run even if ``analysis.json`` already exists.

    Returns:
        Analysis result dict, or ``None`` if the trajectory is missing.
    """
    analysis_path = os.path.join(sim_dir, "analysis.json")
    input_path    = os.path.join(sim_dir, "simulation-input.json")

    if not force and os.path.exists(analysis_path):
        print(f"  [skip] {sim_dir}: analysis.json exists.")
        return load_analysis(sim_dir)

    if not os.path.exists(input_path):
        print(f"  [skip] {sim_dir}: simulation-input.json missing.")
        return None

    with open(input_path) as fh:
        params = json.load(fh)

    timestep_fs      = params["timestep_fs"]
    nblock           = params["nblock"]
    step_skip        = params["step_skip"]
    step_equiv       = params["step_equiv"]
    temperature_dir  = params["temperature_K"]
    traj_timestep_fs = timestep_fs * nblock * step_skip

    try:
        equil, prod, n_prod, _ = _load_trajectory(
            sim_dir, step_skip, step_equiv, traj_timestep_fs)

        if n_prod < 10:
            print(f"  [skip] {sim_dir}: only {n_prod} production frames (need ≥ 10).")
            return None

        # --- Thermodynamic convergence ---
        equil_times, equil_T, equil_P, equil_V = _extract_thermo(
            equil, traj_timestep_fs)
        t_off = step_equiv * traj_timestep_fs / 1000.0
        prod_times,  prod_T,  prod_P,  prod_V  = _extract_thermo(
            prod, traj_timestep_fs, time_offset=t_off)

        for name, ylabel, eq_y, pr_y in (
            ("temperature", "T (K)",     equil_T, prod_T),
            ("pressure",    "P (GPa)",   equil_P, prod_P),
            ("volume",      "V (Å³)",    equil_V, prod_V),
        ):
            _plot_convergence(
                equil_times, prod_times, eq_y, pr_y,
                name, ylabel,
                os.path.join(sim_dir, f"{name}-convergence.pdf"),
            )

        temperature_K = float(np.mean(prod_T))
        thermo = {
            "temperature_K" : {"mean": float(np.mean(prod_T)), "std": float(np.std(prod_T))},
            "pressure_GPa"  : {"mean": float(np.mean(prod_P)), "std": float(np.std(prod_P))},
            "volume_A3"     : {"mean": float(np.mean(prod_V)), "std": float(np.std(prod_V))},
        }

        # --- Diffusivity per species ---
        adaptor    = AseAtomsAdaptor()
        structures = [adaptor.get_structure(a) for a in prod]
        elements   = sorted(structures[0].composition.get_el_amt_dict().keys())
        print(f"  Species detected: {elements}")

        diffusion: dict = {}
        for sp in elements:
            print(f"    Analysing {sp} ...")
            try:
                diffusion[sp] = _compute_diffusivity_one_species(
                    structures, sp, temperature_K, traj_timestep_fs)
            except Exception as exc:
                print(f"    [warn] {sp} skipped: {exc}")

        _plot_msd(diffusion, os.path.join(sim_dir, "diffusion.pdf"))

        # --- Serialise (exclude bulky MSD arrays from JSON) ---
        output = {
            "sim_dir"            : sim_dir,
            "temperature_K"      : temperature_dir,
            "traj_timestep_fs"   : traj_timestep_fs,
            "n_production_frames": n_prod,
            "thermo"             : thermo,
            "diffusion"          : {
                sp: {k: v for k, v in vals.items()
                     if k not in ("msd", "times_ps")}
                for sp, vals in diffusion.items()
            },
        }
        with open(analysis_path, "w") as fh:
            json.dump(output, fh, indent=4)
        print(f"  Analysis → {analysis_path}")

        print("-" * 48)
        print(f"T (K):   {thermo['temperature_K']['mean']:.2f} ± {thermo['temperature_K']['std']:.2f}")
        print(f"P (GPa): {thermo['pressure_GPa']['mean']:.2f} ± {thermo['pressure_GPa']['std']:.2f}")
        print(f"V (Å³):  {thermo['volume_A3']['mean']:.2f}  ± {thermo['volume_A3']['std']:.2f}")
        for sp, vals in diffusion.items():
            print(f"  {sp}: D = {vals['diffusivity_cm2s']:.2e} cm²/s  "
                  f"σ = {vals['conductivity_mScm']:.4f} mS/cm")
        print("-" * 48)
        return output

    except Exception as exc:
        print(f"  [error] {sim_dir}: {exc}")
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# MD simulation runner
# ---------------------------------------------------------------------------

def _run_npt(atoms, temperature: float, traj_path: str, log_path: str) -> None:
    """Run ASE NPT (Martyna-Tobias-Klein) MD.

    Cell shape and volume fluctuate.  Uses the Parrinello-Rahman barostat
    with the time constant ``config.MD_PTIME_FS`` and external stress set by
    ``config.MD_PRESSURE_GPA``.

    Args:
        atoms:       ASE Atoms with calculator attached.
        temperature: Target temperature in Kelvin.
        traj_path:   Output trajectory path.
        log_path:    Thermostat / barostat log path.
    """
    from ase.md.npt import NPT

    dt       = config.MD_TIMESTEP_FS * units.fs
    ttime    = config.MD_TTIME_FS   * units.fs
    ptime    = config.MD_PTIME_FS   * units.fs
    stress   = config.MD_PRESSURE_GPA / 160.21766208  # GPa → eV/Å³
    pfactor  = units.GPa * ptime**2

    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
    dyn = NPT(atoms,
              timestep       = dt,
              temperature_K  = temperature,
              ttime          = ttime,
              pfactor        = pfactor,
              externalstress = stress,
              logfile        = log_path)

    traj = Trajectory(traj_path, "w", atoms)
    dyn.attach(traj.write, interval=config.MD_NBLOCK)
    dyn.run(config.MD_N_STEPS)
    traj.close()


def _run_nvt(atoms, temperature: float, traj_path: str, log_path: str) -> None:
    """Run ASE NVT (Nosé-Hoover) MD.

    Cell is kept fixed.  Implemented via ASE's NPT with ``pfactor=None``
    (no barostat), which gives a Nosé-Hoover chain NVT ensemble.

    Args:
        atoms:       ASE Atoms with calculator attached.
        temperature: Target temperature in Kelvin.
        traj_path:   Output trajectory path.
        log_path:    Thermostat log path.
    """
    from ase.md.npt import NPT

    dt    = config.MD_TIMESTEP_FS * units.fs
    ttime = config.MD_TTIME_FS   * units.fs

    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
    dyn = NPT(atoms,
              timestep      = dt,
              temperature_K = temperature,
              ttime         = ttime,
              pfactor       = None,   # no barostat → NVT
              logfile       = log_path)

    traj = Trajectory(traj_path, "w", atoms)
    dyn.attach(traj.write, interval=config.MD_NBLOCK)
    dyn.run(config.MD_N_STEPS)
    traj.close()


def run_md_simulation(
        sym_dir: str,
        temperature: float,
        device: str = "cuda",
        force_rerun: bool = False,
) -> dict | None:
    """Run the full MD → analysis pipeline for one structure at one temperature.

    The ensemble (NPT or NVT) is selected by ``config.MD_ENSEMBLE``.
    The calculator backend is selected by ``config.CALC_BACKEND``.

    The cell used for MD is the upper-triangular supercell built by
    :func:`heatup.structure_pipeline.prepare_aimd_folders`
    (``aimd/POSCAR``).  If that is absent, falls back to ``sym_dir/POSCAR``.

    Args:
        sym_dir:     Path to the symmetry directory
                     (e.g. ``"database/AgI/P6_3mc"``).
        temperature: Target temperature in Kelvin.
        device:      Compute device (``"cpu"`` or ``"cuda"``).
        force_rerun: Redo both MD and analysis even if ``analysis.json`` exists.

    Returns:
        Analysis result dict (same schema as :func:`analyse_simulation`),
        or ``None`` on failure.
    """
    symmetry = os.path.basename(os.path.abspath(sym_dir))
    material = os.path.basename(os.path.dirname(os.path.abspath(sym_dir)))
    sim      = _sim_dir(sym_dir, temperature)
    tag      = _tag(sym_dir, temperature)

    analysis_path = os.path.join(sim, "analysis.json")
    if not force_rerun and os.path.exists(analysis_path):
        print(f"[done] {tag} — analysis.json exists (use force_rerun=True to redo).")
        return load_analysis(sim)

    os.makedirs(sim, exist_ok=True)

    # Source POSCAR: prefer the prepared supercell.
    aimd_poscar   = os.path.join(sym_dir, "aimd", "POSCAR")
    source_poscar = aimd_poscar if os.path.exists(aimd_poscar) else os.path.join(sym_dir, "POSCAR")
    if not os.path.exists(source_poscar):
        print(f"  [error] {tag}: POSCAR not found. Run prepare_aimd_folders() first.")
        return None
    shutil.copy2(source_poscar, os.path.join(sim, "POSCAR"))

    params = _build_md_params(material, symmetry, temperature, device)
    with open(os.path.join(sim, "simulation-input.json"), "w") as fh:
        json.dump(params, fh, indent=4)
    print(f"Parameters → {sim}/simulation-input.json")

    traj_path = os.path.join(sim, "output.traj")
    log_path  = os.path.join(sim, f"{config.MD_ENSEMBLE.lower()}.log")
    contcar   = os.path.join(sim, "CONTCAR")

    calc  = None
    atoms = None
    if os.path.exists(traj_path) and not force_rerun:
        print(f"  [skip MD] trajectory already exists: {traj_path}")
    else:
        total_ps = config.MD_N_STEPS * config.MD_TIMESTEP_FS / 1000.0
        eff_fs   = config.MD_TIMESTEP_FS * config.MD_NBLOCK
        print(f"Running {config.MD_ENSEMBLE} MD: {tag}  "
              f"({config.MD_N_STEPS} steps × {config.MD_TIMESTEP_FS} fs = {total_ps:.1f} ps, "
              f"writing every {eff_fs:.0f} fs, "
              f"backend={calculator_label()}) ...")

        try:
            atoms      = read_vasp(file=os.path.join(sim, "POSCAR"))
            calc       = build_calculator(device=device)
            atoms.calc = calc

            if config.MD_ENSEMBLE.upper() == "NPT":
                _run_npt(atoms, temperature, traj_path, log_path)
            elif config.MD_ENSEMBLE.upper() == "NVT":
                _run_nvt(atoms, temperature, traj_path, log_path)
            else:
                raise ValueError(
                    f"Unknown MD_ENSEMBLE: {config.MD_ENSEMBLE!r}.  "
                    f"Supported: 'NPT', 'NVT'."
                )

            write_vasp(contcar, atoms, direct=True, sort=True)
            print(f"  MD finished → {traj_path}")

        except Exception as exc:
            print(f"  [error] {tag} MD: {exc}")
            traceback.print_exc()
            return None

        finally:
            if atoms is not None:
                atoms.calc = None
            release_calculator(calc)

    print(f"Analysing trajectory for {tag} ...")
    return analyse_simulation(sim, force=force_rerun)


# ---------------------------------------------------------------------------
# Subprocess wrapper
# ---------------------------------------------------------------------------

def _self_path() -> str:
    return os.path.abspath(__file__)


def run_md_subprocess(
        sym_dir: str,
        temperature: float,
        device: str = "cuda",
        force_rerun: bool = False,
) -> bool:
    """Run MD in a CUDA-isolated subprocess.

    Spawns a child process that calls :func:`run_md_simulation`.  When the
    child exits, the OS reclaims all CUDA memory unconditionally.  Recommended
    for GPU batch runs where multiple materials are processed sequentially.

    Args:
        sym_dir:     Symmetry directory path.
        temperature: Target temperature in Kelvin.
        device:      Compute device string.
        force_rerun: Pass ``--force`` to the subprocess.

    Returns:
        ``True`` if the subprocess exits with code 0.
    """
    cmd = [
        sys.executable, _self_path(),
        "--_run",
        "--sym_dir",     sym_dir,
        "--temperature", str(int(temperature)),
        "--device",      device,
    ]
    if force_rerun:
        cmd.append("--force")

    proc = subprocess.run(cmd, env=_cuda_env(), check=False)
    if proc.returncode != 0:
        print(f"  [error] MD subprocess exited with code {proc.returncode}")
        return False
    return True


# ---------------------------------------------------------------------------
# Database utilities
# ---------------------------------------------------------------------------

def load_analysis(sim_dir: str) -> dict | None:
    """Load ``analysis.json`` from a simulation sub-folder.

    Args:
        sim_dir: Path to a temperature sub-folder.

    Returns:
        Parsed dict, or ``None`` if the file does not exist.
    """
    path = os.path.join(sim_dir, "analysis.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def scan_database(database_root: str) -> list[dict]:
    """Walk *database_root* and return all completed simulation records.

    A simulation is considered complete when ``aimd/<T>K/analysis.json``
    exists and contains at least one entry under ``"diffusion"``.

    Each returned record is the parsed ``analysis.json`` dict extended with:

    - ``"material"``  (str): material folder name.
    - ``"symmetry"``  (str): symmetry sub-folder name.
    - ``"sym_dir"``   (str): full path to the symmetry directory.
    - ``"sim_dir"``   (str): full path to the temperature sub-folder.

    Args:
        database_root: Root of the materials database.

    Returns:
        List of completed simulation records; empty if none found.
    """
    records: list[dict] = []
    if not os.path.isdir(database_root):
        return records

    for material in sorted(os.listdir(database_root)):
        mat_dir = os.path.join(database_root, material)
        if not os.path.isdir(mat_dir):
            continue
        for symmetry in sorted(os.listdir(mat_dir)):
            sym_dir  = os.path.join(mat_dir, symmetry)
            aimd_dir = os.path.join(sym_dir, "aimd")
            if not os.path.isdir(aimd_dir):
                continue
            for temp_folder in sorted(os.listdir(aimd_dir)):
                if not temp_folder.endswith("K"):
                    continue
                sim = os.path.join(aimd_dir, temp_folder)
                data = load_analysis(sim)
                if data is None or not data.get("diffusion"):
                    continue
                data["material"] = material
                data["symmetry"] = symmetry
                data["sym_dir"]  = sym_dir
                data["sim_dir"]  = sim
                records.append(data)
    return records


def print_database_summary(database_root: str) -> None:
    """Print a formatted table of all completed simulations.

    Args:
        database_root: Root of the materials database.
    """
    records = scan_database(database_root)
    if not records:
        print(f"No completed simulations in {database_root}")
        return

    hdr = (f'{"Material":<18} {"Symmetry":<16} {"T (K)":>7}'
           f' {"Sp":<4} {"D (cm²/s)":>14} {"σ (mS/cm)":>14}')
    print(f"\n{hdr}\n{'-' * len(hdr)}")
    for rec in records:
        temp = int(rec.get("temperature_K", 0))
        for sp, vals in rec.get("diffusion", {}).items():
            d = vals.get("diffusivity_cm2s", float("nan"))
            s = vals.get("conductivity_mScm", float("nan"))
            print(f'{rec["material"]:<18} {rec["symmetry"]:<16} {temp:>7d}'
                  f' {sp:<4} {d:>14.3e} {s:>14.4f}')


# ---------------------------------------------------------------------------
# __main__ entry point — used by run_md_subprocess
# ---------------------------------------------------------------------------

def _main_subprocess() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="HeatUp MD runner.")
    parser.add_argument("--_run",        action="store_true")
    parser.add_argument("--sym_dir",     required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--device",      default="cuda")
    parser.add_argument("--force",       action="store_true")
    args = parser.parse_args()

    result = run_md_simulation(
        args.sym_dir, args.temperature,
        device=args.device, force_rerun=args.force,
    )
    return 0 if result is not None else 1


if __name__ == "__main__":
    sys.exit(_main_subprocess())
