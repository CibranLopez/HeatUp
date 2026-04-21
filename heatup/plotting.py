"""heatup.plotting
========================
Dashboard figure for a complete stability report.

Three panels:
    Left:   Anharmonic VDOS with the soft-mode window highlighted.
    Centre: E_above_hull vs temperature with stability regions.
    Right:  Scorecard summarising all three gates with thresholds.
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from heatup import config

# Colour palette (accessible, print-friendly).
_COL = {
    "ok"      : "#27ae60",
    "warn"    : "#f39c12",
    "fail"    : "#c0392b",
    "missing" : "#95a5a6",
    "blue"    : "#1a4a8a",
    "light_g" : "#d5f5e3",
    "light_y" : "#fef9e7",
    "light_r" : "#fadbd8",
}
_MARKER = {
    config.STATUS_OK      : "✓",
    config.STATUS_WARN    : "⚠",
    config.STATUS_FAIL    : "✗",
    config.STATUS_MISSING : "—",
}


def plot_stability_report(
        report: dict,
        save_path: str | None = None,
        show: bool = False,
) -> plt.Figure:
    """Generate the three-panel stability dashboard.

    Args:
        report:    Dict returned by :func:`heatup.pipeline.run_stability_pipeline`.
        save_path: Save the figure to this path if provided (PDF recommended).
        show:      Call ``plt.show()``; set ``False`` inside notebooks with
                   ``%matplotlib inline``.

    Returns:
        The :class:`matplotlib.figure.Figure` object.
    """
    mech  = report.get("mechanical",    {})
    vib   = report.get("vibrational",   {})
    therm = report.get("thermodynamic", {})

    fig, axes = plt.subplots(1, 3, figsize=(15, 5),
                             gridspec_kw={"wspace": 0.35})
    ax_vib, ax_hull, ax_score = axes

    _panel_vdos    (ax_vib,   vib)
    _panel_hull    (ax_hull,  therm)
    _panel_scorecard(ax_score, mech, vib, therm,
                    report.get("material", "?"), report.get("symmetry", "?"))

    overall = report.get("overall", config.STATUS_MISSING)
    ov_col  = _COL.get(overall, _COL["missing"])
    fig.suptitle(
        f"Stability report — {report.get('material','?')} / {report.get('symmetry','?')}",
        fontsize=12, fontweight="bold", color=ov_col, y=1.02,
    )

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Figure → {save_path}")

    if show:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# Individual panels
# ---------------------------------------------------------------------------

def _panel_vdos(ax: plt.Axes, vib: dict) -> None:
    st  = vib.get("status", config.STATUS_MISSING)
    col = _COL.get(st, _COL["missing"])
    ax.set_title(
        f"Vibrational VDOS  {_MARKER.get(st, '—')}",
        fontsize=10, fontweight="bold", color=col,
    )
    if vib.get("available") and vib.get("omega_mev") is not None:
        om = np.array(vib["omega_mev"])
        g  = np.array(vib["vdos"])
        ax.plot(om, g, color=_COL["blue"], lw=1.4, label="VDOS")
        win = config.VIB_ZERO_WINDOW_MEV
        ax.axvspan(-win, win, color=_COL["light_r"], alpha=0.55,
                   label=f"Soft-mode window\n|ω| < {win:.1f} meV")
        ax.axvline(0.0, color="k", lw=0.7, ls="--")
        ax.set_xlabel("ω (meV)", fontsize=9)
        ax.set_ylabel("VDOS (normalised)", fontsize=9)
        ax.set_xlim(om.min(), om.max())
        ax.set_ylim(bottom=0)
        n_src = vib.get("n_sources", 1)
        ax.legend(fontsize=7, loc="upper right",
                  title=f"Averaged over {n_src} MD run(s)", title_fontsize=6)
    else:
        ax.text(0.5, 0.5, "No VDOS data available",
                transform=ax.transAxes, ha="center", va="center",
                color=_COL["missing"])
        ax.set_axis_off()
    ax.grid(True, alpha=0.3, lw=0.5)


def _panel_hull(ax: plt.Axes, therm: dict) -> None:
    st  = therm.get("status", config.STATUS_MISSING)
    col = _COL.get(st, _COL["missing"])
    ax.set_title(
        f"Thermodynamic stability  {_MARKER.get(st, '—')}",
        fontsize=10, fontweight="bold", color=col,
    )
    if not therm.get("available") or not therm.get("hull_results"):
        ax.text(0.5, 0.5, "No hull data available",
                transform=ax.transAxes, ha="center", va="center",
                color=_COL["missing"])
        ax.set_axis_off()
        return

    valid = [(r["T"], r["e_above_hull_eV_atom"])
             for r in therm["hull_results"]
             if r.get("e_above_hull_eV_atom") is not None]
    if not valid:
        ax.text(0.5, 0.5, "Insufficient hull data",
                transform=ax.transAxes, ha="center", va="center",
                color=_COL["missing"])
        return

    T_arr = np.array([v[0] for v in valid])
    E_meV = np.array([v[1] for v in valid]) * 1000

    ax.fill_between(T_arr, -200, 0,
                    color=_COL["light_g"], alpha=0.5, zorder=0, label="Stable")
    ax.fill_between(T_arr, 0, config.THERMO_HULL_WARN_EV * 1000,
                    color=_COL["light_y"], alpha=0.5, zorder=0, label="Metastable")
    ax.fill_between(T_arr, config.THERMO_HULL_WARN_EV * 1000, 600,
                    color=_COL["light_r"], alpha=0.35, zorder=0, label="Unstable")

    ax.axhline(0, color=_COL["ok"],   lw=1.0, ls="--", zorder=1)
    ax.axhline(config.THERMO_HULL_WARN_EV * 1000,
               color=_COL["warn"], lw=1.0, ls="--", zorder=1)
    ax.plot(T_arr, E_meV, color=_COL["blue"], lw=1.5, zorder=3)

    op_T = therm.get("operating_T_K", 1200.0)
    e_op = therm.get("e_above_hull_at_T_eV")
    if e_op is not None:
        ax.axvline(op_T, color="grey", lw=1.0, ls=":", zorder=2)
        ax.annotate(
            f"{op_T:.0f} K\n{e_op * 1000:+.1f} meV/at",
            xy=(op_T, e_op * 1000),
            xytext=(op_T * 1.04, e_op * 1000),
            fontsize=7, color=col,
            arrowprops=dict(arrowstyle="->", color=col, lw=0.8),
        )

    ymax = max(abs(E_meV).max() * 1.15, config.THERMO_HULL_WARN_EV * 1200)
    ax.set_xlim(T_arr.min(), T_arr.max())
    ax.set_ylim(min(E_meV.min() - 20, -30), ymax)
    ax.set_xlabel(
        f"Temperature (K)  [{therm.get('n_competing', 0)} competing phases]",
        fontsize=9,
    )
    ax.set_ylabel("E above hull (meV/atom)", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3, lw=0.5)


def _panel_scorecard(
        ax: plt.Axes,
        mech: dict, vib: dict, therm: dict,
        material: str, symmetry: str,
) -> None:
    ax.set_axis_off()

    criteria = [
        ("1. Mechanical",    mech),
        ("2. Vibrational",   vib),
        ("3. Thermodynamic", therm),
    ]
    statuses = [c.get("status", config.STATUS_MISSING) for _, c in criteria]
    if config.STATUS_FAIL in statuses:
        overall, ov_col = "UNSTABLE", _COL["fail"]
    elif config.STATUS_WARN in statuses or config.STATUS_MISSING in statuses:
        overall, ov_col = "NEEDS ATTENTION", _COL["warn"]
    else:
        overall, ov_col = "STABLE", _COL["ok"]

    ax.text(0.5, 0.97, f"{material} / {symmetry}",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=11, fontweight="bold")
    ax.text(0.5, 0.88, overall,
            transform=ax.transAxes, ha="center", va="top",
            fontsize=13, fontweight="bold", color=ov_col,
            bbox=dict(boxstyle="round,pad=0.4", fc="white",
                      ec=ov_col, linewidth=2))

    row_y = 0.72
    for label, c in criteria:
        st  = c.get("status", config.STATUS_MISSING)
        mk  = _MARKER.get(st, "—")
        col = _COL.get(st, _COL["missing"])
        msg = c.get("message", "No data.")[:80]
        ax.text(0.04, row_y, f"{mk}  {label}",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=9, fontweight="bold", color=col)
        ax.text(0.08, row_y - 0.09, msg,
                transform=ax.transAxes, ha="left", va="top",
                fontsize=7.5, color="#333333")
        row_y -= 0.23

    ax.text(0.04, row_y - 0.02, "Thresholds:",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=7, color="#666666", style="italic")
    thresh_lines = [
        f"  Born:  all C eigenvalues > {config.MECH_BORN_EIGENVALUE_FAIL_GPa:.0f} GPa",
        f"  B:     warn < {config.MECH_BULK_WARN_GPa:.0f} GPa,  fail < {config.MECH_BULK_FAIL_GPa:.0f} GPa",
        f"  VDOS@0: warn > {config.VIB_ZERO_FRAC_WARN * 100:.0f}%,  fail > {config.VIB_ZERO_FRAC_FAIL * 100:.0f}%  (|ω|<{config.VIB_ZERO_WINDOW_MEV:.1f} meV)",
        f"  Hull:  warn > {config.THERMO_HULL_WARN_EV * 1000:.0f} meV/atom at operating T",
    ]
    for i, line in enumerate(thresh_lines):
        ax.text(0.04, row_y - 0.12 - i * 0.09, line,
                transform=ax.transAxes, ha="left", va="top",
                fontsize=6.5, color="#666666", family="monospace")
