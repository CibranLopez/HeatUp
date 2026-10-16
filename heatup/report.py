"""heatup.report
=================
Tables and figures for notebooks: pass the ``reports`` dict ``{tag: run_pipeline(...) result}``
(or the individual gate results) and get a styled DataFrame / a matplotlib figure back.
Everything here is read-only - no calculation is triggered.
"""

from __future__ import annotations

import os

import numpy as np

import matplotlib.pyplot as plt  # noqa: E402

from heatup.pipeline import summary_rows  # noqa: E402

STATUS_BG = {"ok": "#d7f0e0", "warn": "#fdf0c4", "fail": "#f6d0cb", "missing": "#e6e8ec"}


def style_status(df):
    """Colour every cell holding ``ok/warn/fail/missing``."""
    return df.style.map(lambda v: f"background-color: {STATUS_BG.get(v, '')}" if isinstance(v, str) else "")


def gate_table(reports: dict):
    """One row per phase: overall verdict, the four gate statuses, B, G, hull distance."""
    import pandas as pd
    return pd.DataFrame(summary_rows(reports)).set_index("phase")


def vibrational_table(reports: dict, tags=None):
    import pandas as pd
    rows = []
    for tag in tags or reports:
        v = (reports[tag] or {}).get("vibrational") or {}
        if not v.get("available"):
            continue
        rows.append({"phase": tag, "status": v["status"], "evidence": v["evidence"],
                     "soft-mode weight %": 100 * max((r["zero_frac"] for r in v.get("vdos", [])), default=np.nan),
                     "HA imaginary %": 100 * v["ha"]["imag_fraction"] if v.get("ha") else np.nan,
                     "message": v["message"]})
    return pd.DataFrame(rows).set_index("phase").round(3)


def hull_table(reports: dict, tags=None):
    import pandas as pd
    rows = []
    for tag in tags or reports:
        th = (reports[tag] or {}).get("thermodynamic") or {}
        rows.append({"phase": tag, "status": th.get("status", "missing"),
                     "E_hull (meV/atom)": 1e3 * th["e_hull_at_T_eV"] if th.get("available") else np.nan,
                     "decomposes to": ", ".join(f"{v:.2f} {k.split('/')[0]}"
                                                for k, v in (th.get("decomposition") or {}).items()),
                     "flags": "; ".join(th.get("flags", [])) if th.get("flags") else ""})
    return pd.DataFrame(rows).set_index("phase").round(2)


def show_stability_figure(report: dict, stage: str = "post_md"):
    """The saved one-page report image (IPython ``Image``), or None."""
    from IPython.display import Image
    png = os.path.join(report["phase_dir"], "stability", f"report_{stage}.png")
    return Image(png) if os.path.exists(png) else None


def plot_moduli(table, ax=None):
    """Bars: computed vs experimental B and G (``moduli_table`` output)."""
    fig, axes = (plt.subplots(1, 2, figsize=(9, 3.4)) if ax is None else (ax[0].figure, ax))
    x, w = np.arange(len(table)), 0.38
    for a, (k, ke, lab) in zip(axes, [("B (GPa)", "B_exp", "bulk modulus"), ("G (GPa)", "G_exp", "shear modulus")]):
        a.bar(x - w / 2, table[k], w, label="HeatUp")
        a.bar(x + w / 2, table[ke], w, label="experiment")
        a.set_xticks(x, [i.split("/")[0] for i in table.index])
        a.set_ylabel(lab + " (GPa)")
        a.legend()
    fig.tight_layout()
    return fig


def plot_hull_vs_T(reports: dict, tags=None, ax=None, overlay: bool = True):
    """E_hull(T) of the given phases (consistent method, dashed = anharmonic overlay)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    for tag in tags or reports:
        th = (reports[tag] or {}).get("thermodynamic") or {}
        if not th.get("available"):
            continue
        T = np.asarray(th["temperatures"])
        line, = ax.plot(T, 1e3 * np.asarray(th["e_hull_eV"]), label=f"{tag} ({th['method']})")
        if overlay and th.get("overlay"):
            ax.plot(T, 1e3 * np.asarray(th["overlay"]["e_hull_eV"]), "--", color=line.get_color(), lw=1)
    ax.axhline(0, c="k", lw=0.5)
    ax.set(xlabel="T (K)", ylabel="E_hull (meV/atom)")
    ax.legend()
    return ax.figure


def plot_free_energy_terms(report: dict):
    """Left: hull distance vs T (consistent + overlay); right: the free-energy terms of the target."""
    th = report.get("thermodynamic") or {}
    if not th.get("available"):
        print("hull not evaluated:", th.get("message"))
        return None
    T = np.asarray(th["temperatures"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(T, 1e3 * np.asarray(th["e_hull_eV"]), label=f"{th['method']} (consistent)")
    if th.get("overlay"):
        ax[0].plot(T, 1e3 * np.asarray(th["overlay"]["e_hull_eV"]), "--", label="+ anharmonic overlay")
    ax[0].axhline(0, c="k", lw=0.5)
    ax[0].axvline(th["operating_T_K"], c="k", lw=0.5, ls=":")
    ax[0].set(xlabel="T (K)", ylabel="E_hull (meV/atom)", title=f"{report['material']}/{report['label']}")
    ax[0].legend()
    for k, v in th["contributions"].items():
        ax[1].plot(T, 1e3 * (np.asarray(v) - np.asarray(v)[0]), label=k)
    ax[1].set(xlabel="T (K)", ylabel="G - G(0) per term (meV/atom)", title="free-energy terms of the target")
    ax[1].legend()
    fig.tight_layout()
    return fig


def plot_vdos(report: dict, ax=None):
    """HA DOS and MD VDOS of one report on common axes."""
    from heatup.plotting import _vdos
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.5))
    _vdos(ax, report)
    return ax.figure


# ---------------------------------------------------------------------------
# database layout
# ---------------------------------------------------------------------------

def _size(n: int) -> str:
    for unit in ("B", "kB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n} B"


def layout(root: str, max_depth: int = 5, max_files: int = 8, hide: tuple = ("__pycache__", ".ipynb_checkpoints")) -> str:
    """Text tree of a database / competing-phase directory with file sizes - shows what exists and
    where each step wrote its data (``relaxation/``, ``elastic/``, ``phonons/``, ``md/<T>K/``, ``stability/``)."""
    if not os.path.isdir(root):
        return f"{root}/  (does not exist yet)"
    lines = [root.rstrip("/") + "/"]

    def walk(path: str, prefix: str, depth: int) -> None:
        try:
            names = sorted(n for n in os.listdir(path) if n not in hide)
        except OSError:
            return
        dirs = [n for n in names if os.path.isdir(os.path.join(path, n))]
        files = [n for n in names if n not in dirs]
        shown_files = files[:max_files]
        entries = [(d, True) for d in dirs] + [(f, False) for f in shown_files]
        extra = len(files) - len(shown_files)
        for i, (name, is_dir) in enumerate(entries):
            last = i == len(entries) - 1 and extra <= 0
            branch = "└── " if last else "├── "
            full = os.path.join(path, name)
            if is_dir:
                lines.append(f"{prefix}{branch}{name}/")
                if depth < max_depth:
                    walk(full, prefix + ("    " if last else "│   "), depth + 1)
            else:
                lines.append(f"{prefix}{branch}{name}  ({_size(os.path.getsize(full))})")
        if extra > 0:
            lines.append(f"{prefix}└── ... +{extra} more file(s)")

    walk(root, "", 1)
    return "\n".join(lines)


def print_layout(root: str, **kw) -> None:
    print(layout(root, **kw))
