import numpy as np
import matplotlib.pyplot as plt


ROOT = (
    '/Users/davidesciotti/Documents/Work/Code/DATA/Spaceborne_jobs/cov_validation_2026'
)
nz = np.load(f'{ROOT}/input/nzs.npz')

# extract all arrays
z = nz['z']
nz_1 = nz['nz_1']
nz_2 = nz['nz_2']

# convert to SB format
nz_all = np.column_stack((z, nz_1[0, :], nz_1[1, :]))

for zi in range(2):
    plt.plot(nz_all[:, 0], nz_all[:, zi + 1], label=f'n(z) bin {zi + 1} (2D)')

nz_lns = np.column_stack((z, nz_1[0, :]))
nz_src = np.column_stack((z, nz_1[1, :]))

plt.figure()
plt.plot(nz_src[:, 0], nz_src[:, 1], label='src')
plt.plot(nz_lns[:, 0], nz_lns[:, 1], label='lens')
plt.legend()
plt.show()

np.savetxt(f'{ROOT}/input/nz_src.txt', nz_src)
np.savetxt(f'{ROOT}/input/nz_lns.txt', nz_lns)

assert False, 'stop here'


# now the Cls
import euclidlib as el

cl_dict = el.photo.angular_power_spectra(
    '/Users/davidesciotti/Documents/Work/Code/DATA/Spaceborne_jobs/cov_validation_2026/input/cls_theory_lmax_3000.fits'
)

nbl = cl_dict['PxP'].shape

cl_ll_3d = np.zeros()

for zi in range(zbins):
    for zj in range(zbins):
        cl_ij = cl_dict['PxP'][zi, zj, 1, 1].data
        plt.plot(
            cl_dict['PxP'][zi, zj, 1, 1].ell,
            cl_ij,
            label=f'C_ell bin {zi + 1} x bin {zj + 1}',
        )
