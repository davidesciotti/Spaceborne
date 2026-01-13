import numpy as np
import matplotlib.pyplot as plt

nz = np.load(
    '/Users/davidesciotti/Documents/Work/Code/common_data/Spaceborne_jobs/cov_validation_2026/input/nzs.npz'
)

# extract all arrays
z = nz['z']
nz_1 = nz['nz_1']
nz_2 = nz['nz_2']

# convert to SB format
nz_tab = np.column_stack((z, nz_1[0, :], nz_1[1, :]))

for zi in range(2):
    plt.plot(nz_tab[:, 0], nz_tab[:, zi + 1], label=f'n(z) bin {zi + 1} (2D)')

np.savetxt(
    '/Users/davidesciotti/Documents/Work/Code/common_data/Spaceborne_jobs/cov_validation_2026/input/nzs.txt',
    nz_tab,
)

# now the Cls
import euclidlib as el
cl_dict = el.photo.angular_power_spectra('/Users/davidesciotti/Documents/Work/Code/common_data/Spaceborne_jobs/cov_validation_2026/input/cls_theory_lmax_3000.fits')

nbl = cl_dict['PxP'].shape

cl_ll_3d = np.zeros()

for zi in range(zbins):
    for zj in range(zbins):
        cl_ij = cl_dict['PxP'][zi, zj, 1, 1].data
        plt.plot(cl_dict['PxP'][zi, zj, 1, 1].ell, cl_ij, label=f'C_ell bin {zi + 1} x bin {zj + 1}')

