"""heatup.validation
=====================
Reference data and comparisons used by the validation notebook.

Room-temperature Voigt-averaged elastic moduli (GPa) as used in the earlier notebook
(Daniels & Smith, Phys. Rev. 111, 713 (1958); Hiki & Granato (1966); Simmons & Wang (1971)) - verify
before quoting.  PBE-trained MLIPs under-bind, hence the default 20 % tolerance.
"""

from __future__ import annotations

LITERATURE_ELASTIC: dict[str, dict[str, float]] = {
    "Ag/Fm-3m": {"B": 104.0, "G": 30.0},
    "Au/Fm-3m": {"B": 173.0, "G": 27.0},
    "Cu/Fm-3m": {"B": 140.0, "G": 48.0},
}
MODULI_TOLERANCE = 0.20


def moduli_table(reports: dict, literature: dict | None = None):
    """DataFrame: computed vs experimental B and G (GPa) with relative deviations (%)."""
    import pandas as pd
    literature = LITERATURE_ELASTIC if literature is None else literature
    rows = []
    for tag, lit in literature.items():
        m = (reports.get(tag) or {}).get("mechanical") or {}
        if not m.get("available"):
            continue
        rows.append({"phase": tag, "Born stable": m["born_stable"],
                     "B (GPa)": m["bulk_GPa"], "B_exp": lit["B"], "dB %": 100 * (m["bulk_GPa"] / lit["B"] - 1),
                     "G (GPa)": m["shear_GPa"], "G_exp": lit["G"], "dG %": 100 * (m["shear_GPa"] / lit["G"] - 1)})
    return pd.DataFrame(rows).set_index("phase").round(1)
