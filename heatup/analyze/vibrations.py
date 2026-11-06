"""heatup.analyze.vibrations
===========================
Find and SHOW vibrational instabilities.  Reads files only (no calculator).  Used by the vibrational check
(:func:`heatup.analyze.gates.assess_vibrational_stability`) and by notebook 3, which displays the figure of every
material.  What is read and plotted is chosen by ``cfg.vibrational.sources``:

``harmonic``       phonon bands + DOS at the relaxed 0 K cell (``harmonic/harmonic.json``).  Imaginary branches are
                   listed one by one (branch index, minimum, q-point and nearest high-symmetry point, share of the path)
                   and drawn in red.
``quasiharmonic``  the same phonons at every volume of the quasi-harmonic grid: share of imaginary weight per volume and
                   the volumes the crystal actually visits, V_eq(T).
``anharmonic``     the VDOS of the MD runs (``md/<T>K/vdos.json``), every temperature in one panel plus a zoom on the
                   low-frequency end with the soft-mode window shaded and the contribution of every species.

What an MD spectrum can and cannot say
--------------------------------------
A VDOS from the velocity autocorrelation is a non-negative spectral density: it has no sign and no branches, so it
cannot show a *negative* frequency.  A soft or collapsing mode appears as weight piling up at w = 0 (a solid has
g(w) ~ w^2), and a mode that has already driven a phase change shows up in the structural-transition check instead.
Branch-resolved information from MD needs q-resolved spectra (spectral energy density) or effective force constants
fitted to the trajectory; neither is implemented.  The soft-mode report therefore gives the weight inside the window,
which species carries it, and the spectrum itself.
"""
from __future__ import annotations

import logging
import os

import numpy as np

from heatup.core.config import cfg
from heatup.core.layout import PhaseDir

log = logging.getLogger("heatup")

RED = "#c0392b"
LEVEL_ORDER = ("harmonic", "quasiharmonic", "anharmonic")        # increasing physical completeness


# ---------------------------------------------------------------------------
# harmonic: imaginary branches
# ---------------------------------------------------------------------------

def _runs(x: np.ndarray, mask: np.ndarray) -> list[list[float]]:
    """[x_start, x_end] of every contiguous run of True in *mask* along the path coordinate *x*."""
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if start is not None and (not m or i == len(mask) - 1):
            end = i if m else i - 1
            out.append([float(x[start]), float(x[end])])
            start = None
    return out


def imaginary_branches(band: dict, tol_meV: float | None = None) -> list[dict]:
    """Imaginary branches of a phonon band structure (``harmonic.json["band"]``: ``frequencies_meV`` (nk, nbands),
    ``x``, ``special_x``, ``labels``, optionally ``kpoints`` in fractional reciprocal coordinates).

    A branch is the band index in ascending order of frequency at each q, so branch 0 is the lowest.  One entry per
    branch whose minimum is below ``-tol``: ``min_meV``, the q-point of the minimum (``q_frac``) and the nearest
    high-symmetry point (``nearest_label``), the ``fraction_of_path`` that is imaginary and the imaginary ``segments``
    along the path coordinate."""
    tol = abs(cfg.harmonic.imag_tol_meV if tol_meV is None else tol_meV)
    f = np.asarray(band["frequencies_meV"], float)
    x = np.asarray(band["x"], float)
    kp = np.asarray(band.get("kpoints") or [], float)
    sx = np.asarray(band.get("special_x") or [], float)
    labels = list(band.get("labels") or [])
    out = []
    for b in range(f.shape[1]):
        col = f[:, b]
        neg = col < -tol
        if not neg.any():
            continue
        k = int(np.argmin(col))
        lab = None
        if len(sx):
            i = int(np.argmin(np.abs(sx - x[k])))
            lab = labels[i] if i < len(labels) else None
        out.append({"branch": b, "min_meV": float(col[k]), "q_index": k,
                    "q_frac": kp[k].tolist() if len(kp) == len(x) else None,
                    "nearest_label": None if lab is None else str(lab).replace("G", "Γ"),
                    "fraction_of_path": float(neg.mean()), "segments": _runs(x, neg)})
    return out


def describe_branches(branches: list[dict], limit: int = 4) -> str:
    if not branches:
        return "no imaginary branch along the path"
    parts = []
    for b in branches[:limit]:
        q = "" if b["q_frac"] is None else " q=(" + ", ".join(f"{c:.2f}" for c in b["q_frac"]) + ")"
        near = "" if b["nearest_label"] is None else f" near {b['nearest_label']}"
        parts.append(f"branch {b['branch']}: {b['min_meV']:.1f} meV{near}{q}, {100 * b['fraction_of_path']:.0f}% of the path")
    more = f"; +{len(branches) - limit} more" if len(branches) > limit else ""
    return f"{len(branches)} imaginary branch(es): " + "; ".join(parts) + more


def harmonic_report(ha: dict) -> dict:
    """Instability report of a loaded harmonic result (``thermo.load_harmonic``).  Without a band structure (results of
    other codes may only have the mesh) the report has the mesh quantities and no branches."""
    tol = ha.get("imag_tol_meV", cfg.harmonic.imag_tol_meV)
    band = ha.get("band")
    branches = imaginary_branches(band, tol) if band else []
    return {"imag_fraction": ha["imag_fraction"], "max_imag_meV": ha["max_imag_meV"], "tol_meV": tol,
            "has_band": bool(band), "branches": branches, "n_unstable_branches": len(branches),
            "branch_summary": describe_branches(branches) if band else "no band structure stored (mesh only)"}


# ---------------------------------------------------------------------------
# quasi-harmonic: imaginary weight at the volumes that are visited
# ---------------------------------------------------------------------------

def quasiharmonic_report(q: dict) -> dict:
    """Imaginary-weight share per volume of ``quasiharmonic.json`` and the volumes visited at V_eq(T)."""
    vols = np.asarray(q["volumes_A3_per_atom"], float)
    imag = np.asarray(q["imag_fraction"], float)
    veq = np.asarray(q["V_eq_A3_per_atom"], float)
    visited = sorted({int(np.argmin(np.abs(vols - v))) for v in veq})
    return {"volumes_A3_per_atom": vols.tolist(), "imag_fraction": imag.tolist(),
            "temperatures_K": np.asarray(q["temperatures"], float).tolist(), "V_eq_A3_per_atom": veq.tolist(),
            "visited_indices": visited, "visited_max_imag": float(imag[visited].max()),
            "V0_ref_A3_per_atom": q.get("V0_ref_A3_per_atom")}


# ---------------------------------------------------------------------------
# anharmonic (MD): soft-mode weight by species
# ---------------------------------------------------------------------------

def framework_zero_fraction(v: dict, window: float, mobile: list[str]) -> tuple[float, dict]:
    """Atom-weighted fraction of spectral weight with |w| < window, over the non-mobile species of a VDOS result;
    also the fraction of every species."""
    num = den = 0.0
    per = {}
    for sp, p in v["partial"].items():
        om = np.asarray(p["low_omega_meV"])
        w = np.asarray(p["low_weight"])
        z = float(w[np.abs(om) < window].sum() / p["total_weight"])
        per[sp] = z
        if sp in mobile:
            continue
        num += z * p["n_atoms"]
        den += p["n_atoms"]
    if den == 0:                       # everything declared mobile -> use all
        num = sum(per[sp] * p["n_atoms"] for sp, p in v["partial"].items())
        den = sum(p["n_atoms"] for p in v["partial"].values())
    return num / den, per


def _low_dos(v: dict, mobile: list[str] | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Low-frequency end of a VDOS: (omega, atom-weighted DOS in states/atom/meV, {species: DOS}); the weights are per bin."""
    mobile = cfg.analysis.mobile_species if mobile is None else mobile
    sp_all = v["partial"]
    om = np.asarray(next(iter(sp_all.values()))["low_omega_meV"], float)
    d = np.gradient(om) if len(om) > 1 else np.ones_like(om)
    tot, n, per = np.zeros_like(om), 0, {}
    for sp, p in sp_all.items():
        dos = np.asarray(p["low_weight"], float) / d
        per[sp] = dos
        if sp not in mobile:
            tot += dos * p["n_atoms"]
            n += p["n_atoms"]
    return om, tot / max(n, 1), per


def soft_report_md(v: dict, window: float, T: float | None = None) -> dict:
    """Soft-mode report of one MD spectrum: weight inside ``|w| < window``, per species, and the species that carries most."""
    z, per = framework_zero_fraction(v, window, cfg.analysis.mobile_species)
    framework = {sp: x for sp, x in per.items() if sp not in cfg.analysis.mobile_species} or per
    worst = max(framework, key=framework.get)
    return {"temperature_K": T, "window_meV": window, "zero_frac": z, "per_species": per, "dominant_species": worst}


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

def _plt():
    import matplotlib.pyplot as plt
    return plt


def plot_harmonic(ha: dict, ax_band=None, ax_dos=None, title: str | None = None) -> tuple:
    """Phonon bands with the imaginary parts in red, and the DOS beside them.  Axes created when not given."""
    plt = _plt()
    if ax_band is None:
        _, (ax_band, ax_dos) = plt.subplots(1, 2, figsize=(8, 4), gridspec_kw={"width_ratios": [3, 1]})
    tol = abs(ha.get("imag_tol_meV", cfg.harmonic.imag_tol_meV))
    band, modes = ha.get("band"), ha["modes"]
    rep = harmonic_report(ha)
    if band:
        f = np.asarray(band["frequencies_meV"], float)
        x = np.asarray(band["x"], float)
        ax_band.plot(x, np.where(f < -tol, np.nan, f), color="C0", lw=0.9)
        ax_band.plot(x, np.where(f < -tol, f, np.nan), color=RED, lw=1.6)
        for xx in band["special_x"]:
            ax_band.axvline(xx, color="0.75", lw=0.5)
        ax_band.set_xticks(band["special_x"])
        ax_band.set_xticklabels([str(lab).replace("G", "Γ") for lab in band["labels"]])
        ax_band.set_xlim(x.min(), x.max())
        for b in rep["branches"][:3]:
            ax_band.annotate(f"{b['min_meV']:.1f} meV", (x[b["q_index"]], b["min_meV"]),
                             textcoords="offset points", xytext=(4, -10), fontsize=7, color=RED)
    else:
        ax_band.text(0.5, 0.5, "no band structure stored", ha="center", va="center", transform=ax_band.transAxes)
    ax_band.axhline(0, color="k", lw=0.5)
    ax_band.set_ylabel("ω (meV)")
    ax_band.set_title(title or "harmonic", fontsize=9)
    g, d = modes.smooth_dos(cfg.harmonic.dos_sigma_meV)
    neg = g < 0
    ax_dos.fill_betweenx(g, 0, d, where=~neg, color="C0", alpha=0.6)
    ax_dos.fill_betweenx(g, 0, d, where=neg, color=RED, alpha=0.8)
    ax_dos.axhline(0, color="k", lw=0.5)
    ax_dos.set_xlabel("DOS")
    ax_dos.set_yticks([])
    if band:
        ax_dos.set_ylim(ax_band.get_ylim())
    ax_dos.set_title(f"{100 * rep['imag_fraction']:.2f}% imaginary", fontsize=8, color=RED if rep["imag_fraction"] > 0 else "k")
    return ax_band, ax_dos


def plot_quasiharmonic(rep: dict, ax=None):
    """Imaginary-weight share against volume, with the volumes visited between 0 K and the highest T marked."""
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3.6))
    v = np.asarray(rep["volumes_A3_per_atom"], float)
    im = 100 * np.asarray(rep["imag_fraction"], float)
    ax.plot(v, im, "o-", color="C0")
    vis = rep["visited_indices"]
    ax.plot(v[vis], im[vis], "o", color=RED if im[vis].max() > 0 else "C2", ms=7, mfc="none", label="visited at V_eq(T)")
    veq = np.asarray(rep["V_eq_A3_per_atom"], float)
    ax.axvspan(veq.min(), veq.max(), color="0.9", zorder=0)
    ax.set_xlabel("volume (Å³/atom)")
    ax.set_ylabel("imaginary weight (%)")
    ax.set_title("quasi-harmonic", fontsize=9)
    ax.legend(fontsize=7)
    return ax


def plot_quasiharmonic_result(q: dict, path: str, title: str = "") -> str:
    """``phonons/quasiharmonic/quasiharmonic.png``: imaginary weight per volume, and V_eq(T) with the thermal expansion."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    plt = _plt()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.6))
    plot_quasiharmonic(quasiharmonic_report(q), a1)
    a1.set_title(f"{title} quasi-harmonic".strip(), fontsize=9)
    T = np.asarray(q["temperatures"], float)
    a2.plot(T, np.asarray(q["V_eq_A3_per_atom"], float), "o-", color="C0")
    a2.set_xlabel("T (K)")
    a2.set_ylabel("V_eq (Å³/atom)", color="C0")
    a3 = a2.twinx()
    a3.plot(T, 1e6 * np.asarray(q["alpha_V_1_K"], float), "s--", color="C3", ms=3)
    a3.set_ylabel("α_V (1e-6 /K)", color="C3")
    a2.set_title("thermal expansion", fontsize=9)
    fig.tight_layout()
    try:
        fig.savefig(path, dpi=110)
    finally:
        plt.close(fig)
    return path


def plot_perturbative(res: dict, axes=None):
    """kappa_RTA(T) and <gamma/nu>(T) of a perturbative (fc3) result; the dashed line is ``cfg.perturbative.ratio_warn``."""
    plt = _plt()
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    T = res["temperatures_K"]
    axes[0].plot(T, res["kappa_W_mK"], "o-")
    axes[0].set_xlabel("T (K)")
    axes[0].set_ylabel("kappa_RTA (W/m/K)")
    axes[0].set_yscale("log")
    ratio = [np.nan if r is None else r for r in res["gamma_over_nu_mean"]]
    axes[1].plot(T, ratio, "o-", color="C3")
    axes[1].axhline(cfg.perturbative.ratio_warn, color="0.5", ls="--", lw=0.8, label="warn")
    axes[1].set_xlabel("T (K)")
    axes[1].set_ylabel("<gamma / nu>")
    axes[1].legend()
    return axes


def plot_perturbative_result(res: dict, path: str, title: str = "") -> str:
    """``phonons/perturbative/perturbative.png``."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    plot_perturbative(res, axes)
    axes[0].set_title(f"{title} perturbative (fc3)".strip(), fontsize=9)
    fig.tight_layout()
    try:
        fig.savefig(path, dpi=110)
    finally:
        plt.close(fig)
    return path


def plot_anharmonic(phase, temperatures=None, ax_full=None, ax_low=None, window: float | None = None):
    """VDOS of the MD runs: whole spectrum, and the low-frequency end with the soft-mode window and species."""
    from heatup.analyze.trajectory import load_vdos
    from heatup.core import ledger
    plt = _plt()
    phase = PhaseDir.of(phase)
    if ax_full is None:
        _, (ax_full, ax_low) = plt.subplots(1, 2, figsize=(8, 3.6))
    temps = list(temperatures) if temperatures is not None else ledger.md_temperatures(phase)
    cm = plt.get_cmap("viridis")
    last = None
    for i, T in enumerate(temps):
        v = load_vdos(phase, T)
        if not v:
            continue
        c = cm(i / max(1, len(temps) - 1))
        o, w = v["modes"].omega, v["modes"].weight
        g = w / np.gradient(o)
        ax_full.plot(o, g, color=c, lw=0.9, label=f"{T:g} K")
        om, tot, per = _low_dos(v)
        ax_low.plot(om, tot, color=c, lw=1.2)
        win = window or max(cfg.gates.zero_window_meV, 1.5 * v["resolution_meV"])
        last = (v, om, per, win, c)
    ax_full.set_xlabel("ω (meV)")
    ax_full.set_ylabel("VDOS (states/atom/meV)")
    ax_full.set_title("anharmonic: VDOS from MD", fontsize=9)
    if temps:
        ax_full.legend(fontsize=7)
    if last:
        v, om, per, win, c = last
        ax_low.axvspan(0, win, color=RED, alpha=0.15, label=f"soft-mode window ({win:.1f} meV)")
        if len(per) > 1:
            for sp, d in per.items():
                ax_low.plot(om, d, "--", lw=0.8, color=c, alpha=0.6)
                ax_low.annotate(sp, (om[int(np.argmax(d))], d.max()), fontsize=7, color="0.3")
        ax_low.legend(fontsize=7)
    ax_low.set_xlim(left=0)
    ax_low.set_xlabel("ω (meV)")
    ax_low.set_ylabel("low-ω zoom")
    ax_low.set_title("weight at ω→0 = soft modes", fontsize=9)
    return ax_full, ax_low


def plot_vdos_run(v: dict, T: float, path: str, title: str = "") -> str | None:
    """The figure written next to every ``md/<T>K/vdos.json``: whole spectrum (total and per species) and the low end."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    plt = _plt()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))
    o, w = v["modes"].omega, v["modes"].weight
    a1.fill_between(o, w / np.gradient(o), color="C0", alpha=0.35, label="all atoms")
    for sp, p in v["partial"].items():
        so, sw = np.asarray(p["omega_meV"]), np.asarray(p["weight"])
        if len(so) > 1:
            a1.plot(so, sw / np.gradient(so), lw=0.9, label=sp)
    a1.set_xlabel("ω (meV)")
    a1.set_ylabel("VDOS (states/atom/meV)")
    a1.legend(fontsize=7)
    a1.set_title(f"{title} {T:g} K".strip(), fontsize=9)
    om, tot, per = _low_dos(v)
    win = max(cfg.gates.zero_window_meV, 1.5 * v["resolution_meV"])
    z, _ = framework_zero_fraction(v, win, cfg.analysis.mobile_species)
    a2.plot(om, tot, color="C0", lw=1.3)
    for sp, d in per.items():
        a2.plot(om, d, "--", lw=0.8, label=sp)
    a2.axvspan(0, win, color=RED, alpha=0.15)
    a2.set_xlim(left=0)
    a2.set_xlabel("ω (meV)")
    a2.set_title(f"low-ω: {100 * z:.3f}% of the weight in |ω| < {win:.1f} meV", fontsize=9,
                 color=RED if z >= cfg.gates.zero_frac_warn else "k")
    a2.legend(fontsize=7)
    fig.tight_layout()
    try:
        fig.savefig(path, dpi=110)
    finally:
        plt.close(fig)
    return path


def plot_vibrational_phase(phase_dir, vib: dict | None = None, sources: list | None = None, save_path: str | None = None,
                           title: str | None = None):
    """One figure with the panels of the selected sources (``cfg.vibrational.sources`` by default), in the order
    harmonic, quasi-harmonic, anharmonic.  Panels whose data are missing say so.  Returns the figure."""
    from heatup.analyze.thermo import load_harmonic, load_quasiharmonic
    from heatup.core import ledger
    plt = _plt()
    phase = PhaseDir.of(phase_dir)
    sources = [s for s in LEVEL_ORDER if s in (cfg.vibrational.sources if sources is None else sources)]
    if not sources:
        raise ValueError("no vibrational source selected (cfg.vibrational.sources)")
    widths = {"harmonic": [3, 1], "quasiharmonic": [2], "anharmonic": [2, 2]}
    ratios = [w for s in sources for w in widths[s]]
    fig = plt.figure(figsize=(3.4 * sum(len(widths[s]) for s in sources) + 1.4, 3.9))
    gs = fig.add_gridspec(1, len(ratios), width_ratios=ratios, wspace=0.28)
    col = 0
    for s in sources:
        n = len(widths[s])
        axes = [fig.add_subplot(gs[0, col + i]) for i in range(n)]
        col += n
        try:
            if s == "harmonic":
                ha = load_harmonic(phase)
                if not ha:
                    raise FileNotFoundError("harmonic/harmonic.json")
                plot_harmonic(ha, *axes, title=f"harmonic: {harmonic_report(ha)['branch_summary'][:60]}")
            elif s == "quasiharmonic":
                q = load_quasiharmonic(phase)
                if not q:
                    raise FileNotFoundError("quasiharmonic/quasiharmonic.json")
                plot_quasiharmonic(quasiharmonic_report(q), axes[0])
            else:
                if not ledger.md_temperatures(phase):
                    raise FileNotFoundError("no finished MD run")
                plot_anharmonic(phase, None, *axes)
        except FileNotFoundError as exc:
            for a in axes:
                a.axis("off")
            axes[0].text(0.5, 0.5, f"{s}:\nno data\n({exc})", ha="center", va="center", transform=axes[0].transAxes,
                         fontsize=8, color="0.4")
    name = title or phase.tag
    status = "" if not vib else f"  -  {vib.get('status', '')}"
    fig.suptitle(f"{name}: vibrational{status}", fontsize=10)
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.14, top=0.84)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=110)
    return fig


def plot_vibrational(report: dict, sources: list | None = None):
    """The vibrational figure of one entry of ``evaluate_all()`` / ``evaluate()`` (what notebook 3 displays)."""
    vib = report.get("vibrational") or {}
    src = sources or vib.get("sources")
    return plot_vibrational_phase(report["phase_dir"], vib, src, title=f"{report.get('material', '')}/{report.get('label', '')}")


def save_vibrational_figure(phase_dir, vib: dict, sources: list) -> str | None:
    """Write ``stability/vibrational.png`` (called by the vibrational check when ``cfg.vibrational.save_figures``)."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    plt = _plt()
    phase = PhaseDir.of(phase_dir)
    path = os.path.join(phase.stability_dir, "vibrational.png")
    try:
        fig = plot_vibrational_phase(phase, vib, sources, path)
        plt.close(fig)
        return path
    except Exception as exc:  # noqa: BLE001
        log.warning("%s: vibrational figure failed: %s", phase.tag, exc)
        return None
