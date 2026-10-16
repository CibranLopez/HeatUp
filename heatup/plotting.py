"""heatup.plotting - dashboard figure for a stability report."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from heatup.config import STATUS_FAIL, STATUS_MISSING, STATUS_OK, STATUS_WARN, cfg

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
    from heatup.phonons import load_ha
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
