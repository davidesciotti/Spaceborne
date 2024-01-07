""" formula in https://arxiv.org/pdf/2007.00675.pdf"""

import numpy as np
import sys
import matplotlib.pyplot as plt

ROOT = '/Users/davide/Documents/Lavoro/Programmi'
sys.path.append(f'{ROOT}/Spaceborne/bin')
import cosmo_lib as cosmo_lib
import wf_cl_lib as wf_cl_lib

Nz_bins = 10
z_grid = np.linspace(1e-5, 3, 1000)

chi = cosmo_lib.ccl_comoving_distance(z_grid, use_h_units=False)
n_i_list = [wf_cl_lib.niz_unnormalized_simps(z_grid, zbin_idx) for zbin_idx in range(Nz_bins)]
n_i_list = [wf_cl_lib.normalize_niz_simps(n_i_list[zbin_idx], z_grid) for zbin_idx in range(Nz_bins)]

for i in range(Nz_bins):
    plt.plot(z_grid, n_i_list[i])
plt.show()

A_list = np.zeros((Nz_bins))
B_list = np.zeros((Nz_bins))
for i in range(Nz_bins):
    nz = n_i_list[i]
    A_list[i] = np.trapz(nz, z_grid)
    B_list[i] = np.trapz(nz / chi, z_grid)

BNT_matrix = np.eye(Nz_bins)
BNT_matrix[1, 0] = -1.
for i in range(2, Nz_bins):
    mat = np.array([[A_list[i - 1], A_list[i - 2]],
                    [B_list[i - 1], B_list[i - 2]]])
    A = -1. * np.array([A_list[i], B_list[i]])
    soln = np.dot(np.linalg.inv(mat), A)
    BNT_matrix[i, i - 1] = soln[0]
    BNT_matrix[i, i - 2] = soln[1]

plt.matshow(BNT_matrix)
