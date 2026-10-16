"""heatup.constants - physical constants and unit conversions (single source)."""

#: Boltzmann constant
KB_EV = 8.617333262e-5          # eV / K
KB_MEV = KB_EV * 1e3            # meV / K
#: eV/Å^3 -> GPa
EV_A3_TO_GPA = 160.21766208
GPA_TO_EV_A3 = 1.0 / EV_A3_TO_GPA
#: Planck constant in meV*fs:  E[meV] = H_MEV_FS * nu[cycles/fs]
H_MEV_FS = 4135.667696
#: 1 THz in meV
MEV_PER_THZ = 4.135667696
#: Gas constant etc. are never needed: everything is per atom in eV.

#: Vibrational modes per atom (3 for a crystal).
MODES_PER_ATOM = 3.0
#: Frequencies below this are acoustic/numerical zero modes and excluded
#: from thermodynamic integrals (meV).
OMEGA_MIN_MEV = 0.1
