"""heatup.analyze.report
========================
Tables and figures for notebooks (read-only - no calculation is triggered): pass the ``reports`` dict
``{tag: evaluate(...) result}`` (or the individual gate results) and get a styled DataFrame / a matplotlib figure
back; plus the status table of the project and the one-page dashboard figure of a report.
"""

from __future__ import annotations

import os
from collections import Counter

import numpy as np

import matplotlib.pyplot as plt  # noqa: E402

from heatup.analyze.evaluate import summary_rows  # noqa: E402
from heatup.analyze.thermo import load_fc3, load_ha  # noqa: E402
from heatup.core import ledger  # noqa: E402
from heatup.core.config import STATUS_FAIL, STATUS_MISSING, STATUS_OK, STATUS_WARN, cfg  # noqa: E402


STATUS_BG = {"ok": "#d7f0e0", "warn": "#fdf0c4", "fail": "#f6d0cb", "missing": "#e6e8ec",
             # step states (ledger)
             "done": "#d7f0e0", "stale": "#fdf0c4", "pending": "#e6e8ec", "running": "#d6e4f7",
             "crashed": "#f6d0cb", "unfinished": "#fbdcb8", "failed": "#f6d0cb", "pruned": "#f1f1f1",
             "external": "#e1f0f7"}


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
    """Text tree of the database (or candidates) directory with file sizes - shows what exists and
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


def plot_fc3(report: dict, axes=None):
    """kappa_RTA(T) and <gamma/nu>(T) of the target (needs ``cfg.phonon.fc_order = 3``)."""
    res = load_fc3(report["phase_dir"])
    if not res:
        raise ValueError("no fc3 result: set cfg.phonon.fc_order = 3 and run the 'fc3' step")
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    T = res["temperatures_K"]
    axes[0].plot(T, res["kappa_W_mK"], "o-")
    axes[0].set_xlabel("T (K)"); axes[0].set_ylabel("kappa_RTA (W/m/K)"); axes[0].set_yscale("log")
    ratio = [np.nan if r is None else r for r in res["gamma_over_nu_mean"]]
    axes[1].plot(T, ratio, "o-", color="C3")
    axes[1].axhline(cfg_ratio_warn(), color="0.5", ls="--", lw=0.8, label="warn")
    axes[1].set_xlabel("T (K)"); axes[1].set_ylabel("<gamma / nu>"); axes[1].legend()
    return axes


def cfg_ratio_warn() -> float:
    return cfg.fc3.ratio_warn

# ---------------------------------------------------------------------------
# project status (what is computed, what is not)
# ---------------------------------------------------------------------------

def status_table(rows: list[dict], problems_only: bool = False):
    """DataFrame of ``heatup.status()`` rows (index: phase), coloured with :func:`style_status`."""
    import pandas as pd
    if problems_only:
        rows = [r for r in rows if r["state"] not in (*ledger.GOOD_STATES, ledger.PRUNED)]
    cols = ["phase", "role", "step", "T_K", "state", "detail", "calculator"]
    df = pd.DataFrame(rows, columns=cols)
    return df.set_index("phase") if len(df) else df


def status_report(rows: list[dict], max_lines: int = 40) -> str:
    """Plain-text summary of ``heatup.status()``: counts per state and one warning line per unfinished piece."""
    if not rows:
        return "no planned calculations found (run notebook 1: prepare_project)"
    n = Counter(r["state"] for r in rows)
    lines = ["states: " + ", ".join(f"{k} {v}" for k, v in sorted(n.items()))]
    bad = [r for r in rows if r["state"] not in (*ledger.GOOD_STATES, ledger.PRUNED)]
    order = [ledger.FAILED, ledger.CRASHED, ledger.UNFINISHED, ledger.RUNNING, ledger.STALE, ledger.PENDING]
    bad.sort(key=lambda r: (order.index(r["state"]) if r["state"] in order else 9, r["phase"], r["step"]))
    for r in bad[:max_lines]:
        at = "" if r["T_K"] is None else f"@{r['T_K']:g}K"
        lines.append(f"  [{r['state'].upper():10s}] {r['phase']}: {r['step']}{at}" + (f" - {r['detail']}" if r["detail"] else ""))
    if len(bad) > max_lines:
        lines.append(f"  ... {len(bad) - max_lines} more")
    labels = sorted({r["calculator"] for r in rows if r["calculator"]})
    if len(labels) > 1:
        lines.append(f"  [WARNING] phases were computed with different calculators: {labels} - "
                     "hull distances are only comparable within one")
    return "\n".join(lines)


def show_status(rows: list[dict] | None = None) -> None:
    """Print :func:`status_report` (computing ``heatup.status()`` if no rows are given)."""
    if rows is None:
        from heatup.analyze.evaluate import status
        rows = status()
    print(status_report(rows))



# ---------------------------------------------------------------------------
# one-page dashboard figure of a report
# ---------------------------------------------------------------------------

COLORS = {STATUS_OK: "#2a9d5c", STATUS_WARN: "#e0a100", STATUS_FAIL: "#c0392b", STATUS_MISSING: "#8a8f98"}
MARK = {STATUS_OK: "pass", STATUS_WARN: "warn", STATUS_FAIL: "FAIL", STATUS_MISSING: "n/a"}
TITLES = {"structural_transition": "Gate 0  structure", "mechanical": "Gate 1  Born",
          "vibrational": "Gate 2  vibrational", "thermodynamic": "Gate 3  hull"}


def _wrap(s: str, n: int = 58) -> str:
    out, line = [], ""
    for w in s.split():
        if len(line) + len(w) + 1 > n:
            out.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    return "\n".join(out + [line])


def _scorecard(ax, report: dict) -> None:
    ax.axis("off")
    ax.set_title(f"{report['material']} / {report['label']}  -  {report['operating_T_K']:g} K  -  "
                 f"overall {report['overall'].upper()}", loc="left", fontsize=11,
                 color=COLORS[report["overall"]], fontweight="bold")
    y = 0.95
    for g, title in TITLES.items():
        r = report.get(g)
        st = r["status"] if r else STATUS_MISSING
        if g in report.get("skipped", []) and not r:
            st = STATUS_MISSING
        ax.text(0.0, y, f"{title}", fontsize=10, fontweight="bold", va="top", transform=ax.transAxes)
        ax.text(0.30, y, MARK[st], fontsize=10, color=COLORS[st], fontweight="bold", va="top",
                transform=ax.transAxes)
        msg = _wrap(r["message"] if r else ("skipped" if g in report.get("skipped", []) else "not evaluated"))
        ax.text(0.40, y, msg, fontsize=8, va="top", transform=ax.transAxes)
        y -= 0.07 + 0.055 * msg.count("\n") + 0.03


def _vdos(ax, report: dict) -> None:
    vib = report.get("vibrational") or {}
    ax.set_xlabel("energy (meV)")
    ax.set_ylabel("DOS (modes / atom / meV)")
    ax.set_title("Vibrational DOS", loc="left", fontsize=10)
    ha = load_ha(report["phase_dir"]) if report.get("phase_dir") else None
    if ha:
        sm = ha["modes"].smooth_dos(sigma_meV=1.0)
        ax.fill_between(sm[0], sm[1], color="#b0c4de", alpha=0.6, label="HA (0 K)")
    cols = plt.cm.viridis(np.linspace(0.15, 0.85, max(1, len(vib.get("vdos", [])))))
    for c, r in zip(cols, vib.get("vdos", [])):
        w, o = np.asarray(r["weight"]), np.asarray(r["omega_meV"])
        if len(o) > 2:
            ax.plot(o, w / np.gradient(o), color=c, lw=1.2, label=f"MD {r['temperature_K']:.0f} K")
            ax.axvspan(-r["zero_window_meV"], r["zero_window_meV"], color=COLORS[STATUS_FAIL], alpha=0.15)
    if ha or vib.get("vdos"):
        ax.legend(frameon=False, fontsize=8)
    else:
        ax.text(0.5, 0.5, "no spectra", ha="center", transform=ax.transAxes)


def _hull(ax, report: dict) -> None:
    th = report.get("thermodynamic") or {}
    ax.set_title("Distance to hull", loc="left", fontsize=10)
    ax.set_xlabel("T (K)")
    ax.set_ylabel("E$_{hull}$ (meV/atom)")
    if not th.get("available"):
        ax.text(0.5, 0.5, "hull not evaluated", ha="center", transform=ax.transAxes)
        return
    T, e = np.asarray(th["temperatures"]), 1e3 * np.asarray(th["e_hull_eV"])
    g = cfg.gates
    ax.axhspan(-1e3, 1e3 * g.hull_ok_eV, color=COLORS[STATUS_OK], alpha=0.12)
    ax.axhspan(1e3 * g.hull_ok_eV, 1e3 * g.hull_fail_eV, color=COLORS[STATUS_WARN], alpha=0.12)
    ax.axhspan(1e3 * g.hull_fail_eV, 1e4, color=COLORS[STATUS_FAIL], alpha=0.12)
    ax.plot(T, e, color="#1f3a6e", lw=1.8, label=f"{th['method']} (consistent)")
    if th.get("overlay"):
        ax.plot(T, 1e3 * np.asarray(th["overlay"]["e_hull_eV"]), "--", color="#7a3e9d",
                label="+ anharmonic overlay")
    ax.axvline(th["operating_T_K"], color="k", lw=0.6, ls=":")
    lo, hi = min(e.min(), -20), max(e.max(), 1e3 * g.hull_fail_eV * 1.3)
    ax.set_ylim(lo * 1.1 if lo < 0 else lo, hi)
    ax.legend(frameon=False, fontsize=8)


def _elastic(ax, report: dict) -> None:
    m = report.get("mechanical") or {}
    ax.set_title("Stiffness matrix C (GPa)", loc="left", fontsize=10)
    if not m.get("available"):
        ax.axis("off")
        ax.text(0.5, 0.5, "not computed", ha="center", transform=ax.transAxes)
        return
    C = np.array(m["C_GPa"])
    ax.imshow(C, cmap="RdBu_r", vmin=-abs(C).max(), vmax=abs(C).max())
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{C[i, j]:.0f}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(6), list("123456"))
    ax.set_yticks(range(6), list("123456"))
    ax.set_xlabel(f"B={m['bulk_GPa']:.0f}  G={m['shear_GPa']:.0f}  "
                  f"min eig={m['min_eigenvalue_GPa']:.0f} GPa", fontsize=8)


def plot_stability_report(report: dict, save_path: str | None = None) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), constrained_layout=True)
    _scorecard(axes[0, 0], report)
    _elastic(axes[0, 1], report)
    _vdos(axes[1, 0], report)
    _hull(axes[1, 1], report)
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
