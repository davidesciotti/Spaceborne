import numpy as np
import matplotlib.pyplot as plt

"""
plot the constraints from Table 1 in the paper. Very unrefined; I wanted to check that:
- the 3x2pt is NOT always an intermediate case between WL and GC
"""

param_names_label = ["$\Omega_{{\\rm m},0}$", "$\Omega_{{\\rm b},0}$", "$w_0$", "$w_a$", "$h$", "$n_s$",
                      "$\sigma_8$", "FoM"]


colors = ['tab:blue', 'tab:orange', 'tab:green']
# WL, Pessismistic WL, Optimistic
WL_Pes = (np.asarray((2.95, 1.01, 1.80, 1.34, 1.10, 1.04, 2.22, 0.48)) - 1)*100
WL_Opt = (np.asarray((2.81, 1.00, 1.61, 1.18, 1.10, 1.00, 2.17, 0.41)) - 1)*100

# GCph, Pessismistic GCph, Optimistic
GCph_Pes = (np.asarray((1.01, 1.01, 1.02, 1.03, 1.00, 1.01, 1.00, 0.96)) - 1)*100
GCph_Opt = (np.asarray((1.04, 1.03, 1.12, 1.15, 1.00, 1.03, 1.04, 0.87)) - 1)*100
# 3x2pt, Pessismistic 3x2pt, Optimistic
tx2pt_Pes = (np.asarray((1.98, 1.10, 1.87, 1.51, 1.10, 1.23, 1.63, 0.45)) - 1)*100
tx2pt_Opt = (np.asarray((1.60, 1.01, 1.53, 1.35, 1.01, 1.07, 1.42, 0.56)) - 1)*100

plt.plot(WL_Pes, '-o', ls='--', label='WL Pes', color=colors[0])
plt.plot(WL_Opt, '-o', label='WL Opt', color=colors[0])
plt.plot(GCph_Pes, '-o', ls='--', label='GCph Pes', color=colors[1])
plt.plot(GCph_Opt, '-o', label='GCph Opt', color=colors[1])
plt.plot(tx2pt_Pes, '-o', ls='--', label='3x2pt Pes', color=colors[2])
plt.plot(tx2pt_Opt, '-o', label='3x2pt Opt', color=colors[2])
plt.xticks(range(8), param_names_label)
plt.ylabel('$(\\sigma_{GS}/\\sigma_{GO} - 1) \\times 100$')

plt.grid()
plt.legend()
plt.tight_layout()

