import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys

sys.path.append('/')
import mpl_cfg



mpl.rcParams.update(mpl_cfg.mpl_rcParams_dict)

mpl.pyplot.set_cmap('rainbow')
ep_or_ed = 'EP'
zbins = 10
n_of_z_fs1_paper = np.genfromtxt(f'/home/davide/Documenti/Lavoro/Programmi/common_data/vincenzo/SPV3_07_2022/'
                                 f'Flagship_1_restored/InputNz/Lenses/Flagship/niTab-{ep_or_ed}{zbins}.dat')

plt.figure()
for zi in range(zbins):
    plt.plot(n_of_z_fs1_paper[:, 0], n_of_z_fs1_paper[:, zi + 1], label='$z_{%d}$' % zi)

plt.legend()
plt.show()
plt.xlim(0, 2.5)
plt.xlabel('$z$')
plt.ylabel('$n_i(z)$')
plt.tight_layout()

# savefig in .pdf
plt.savefig(
    f'/home/davide/Documenti/Lavoro/Programmi/SSC_restructured_v2/jobs/SSC_paper_final/output/plots/'
    f'n_of_z_{ep_or_ed}{zbins}.pdf',
    dpi=500, bbox_inches='tight', pad_inches=0.1, transparent=True, format='pdf')
