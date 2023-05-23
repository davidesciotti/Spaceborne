import numpy as np
import sys

sys.path.append('/Users/davide/Documents/Lavoro/Programmi/common_data/common_lib')
import my_module as mm

z_bins = 10

z_pairs_auto = z_bins * (z_bins + 1) // 2  # = 55
z_pairs_cross = z_bins ** 2  # = 100

ind_auto = np.zeros((z_pairs_auto, 2))  # for LL and GG
ind_cross = np.zeros((z_pairs_cross, 2))  # for GL/LG

ij = 0
for i in range(z_bins):
    for j in range(i, z_bins):
        ind_auto[ij, :] = i, j
        ij += 1

ij = 0
for i in range(z_bins):
    for j in range(z_bins):
        ind_cross[ij, :] = i, j
        ij += 1

ind_auto = ind_auto.astype('int')
ind_cross = ind_cross.astype('int')

print('WL and GCph \t\t LG/GL')
print('____________________________________')
print('ij \t i,j \t\t ij \t i,j')
for ij in range(z_pairs_auto):
    print(ij, '\t', ind_auto[ij, 0], ind_auto[ij, 1], '\t\t', ij, '\t', ind_cross[ij, 0], ind_cross[ij, 1])
print('\t\t\t ... \t ...')

ell_bins = 20
cl_LL_2D = np.zeros((ell_bins, z_pairs_auto))
cl_GL_2D = np.zeros((ell_bins, z_pairs_cross))

cl_LL_3D = np.zeros((ell_bins, z_bins, z_bins))
for ell in range(ell_bins):
    for i in range(z_bins):
        for j in range(i, z_bins):
            cl_LL_3D[ell, i, j] = np.random.rand()
            cl_LL_3D[ell, j, i] = cl_LL_3D[ell, i, j]

cl_GL_3D = np.random.rand(ell_bins, z_bins, z_bins)

for ell in range(ell_bins):
    for ij in range(z_pairs_auto):
        i, j = ind_auto[ij, 0], ind_auto[ij, 1]
        cl_LL_2D[ell, ij] = cl_LL_3D[ell, i, j]

for ell in range(ell_bins):
    for ij in range(z_pairs_cross):
        i, j = ind_cross[ij, 0], ind_cross[ij, 1]
        cl_GL_2D[ell, ij] = cl_GL_3D[ell, i, j]

# or, in a more pythonic way:
cl_LL_2D = cl_LL_3D[:, ind_auto[:, 0], ind_auto[:, 1]]
cl_GL_2D = cl_GL_3D[:, ind_cross[:, 0], ind_cross[:, 1]]

# or, numpy.ndarray.flatten:
for ell in range(ell_bins):
    cl_GL_2D[ell, :] = cl_GL_3D[ell, :].flatten(order='C')

# 2D -> 3D cross
cl_GL_3D = np.zeros((ell_bins, z_bins, z_bins))
for ell in range(ell_bins):
    cl_GL_3D[ell, :, :] = cl_GL_2D[ell, :].reshape((z_bins, z_bins), order='C')

# 2D -> 3D auto
cl_LL_3D_v2 = np.zeros((ell_bins, z_bins, z_bins))
for ell in range(ell_bins):
    for ij in range(z_pairs_auto):
        i, j = ind_auto[ij, 0], ind_auto[ij, 1]
        cl_LL_3D_v2[ell, i, j] = cl_LL_2D[ell, ij]
        cl_LL_3D_v2[ell, j, i] = cl_LL_2D[ell, ij]

    # cl_LL_3D_v2[ell, :, :] = mm.symmetrize_2d_array(cl_LL_3D_v2[ell, :, :])

assert np.all(cl_LL_3D == cl_LL_3D_v2), 'cl_LL_3D_v2 is not correct'
mm.matshow(cl_LL_3D_v2[0, :, :], log=True, title='cl_LL_3D[0, :, :]')


# ! 2D <-> 1D
cl_LL_1D = cl_LL_2D.flatten(order='C')
cl_GL_1D = cl_GL_2D.flatten(order='C')

cl_LL_2D_v2 = cl_LL_1D.reshape((ell_bins, z_pairs_auto), order='C')
cl_GL_2D_v2 = cl_GL_1D.reshape((ell_bins, z_pairs_cross), order='C')

assert np.all(cl_LL_2D == cl_LL_2D_v2), 'cl_LL_2D_v2 is not correct'
assert np.all(cl_GL_2D == cl_GL_2D_v2), 'cl_GL_2D_v2 is not correct'

ind_bench = mm.build_full_ind('triu', 'row-major', z_bins)
assert np.all(ind_bench[:z_pairs_auto, 2:] == ind_auto), 'ind_auto is not correct'
assert np.all(ind_bench[z_pairs_auto:z_pairs_auto + z_pairs_cross, 2:] == ind_cross), 'ind_auto is not correct'


cov_test = np.load('/Users/davide/Documents/Lavoro/Programmi/cov_per_riccardo/output/Euclid/cov_3x2pt_GO_2D.npz')['arr_0']
mm.matshow(cov_test, log=True, title='3x2pt cov, ell_probe_zpair')
