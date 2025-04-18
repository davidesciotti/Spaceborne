from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np

from scipy.interpolate import CubicSpline, RegularGridInterpolator
from spaceborne import cosmo_lib
from spaceborne import sb_lib as sl




# SECOND TEST - 18/04/2025
# check CCL trispectrum
trisp_path = '/home/cosmo/davide.sciotti/data/Spaceborne/output/cache/trispectrum/cNG'
ka_str = 'amin0.25_amax0.98_asteps50_lnkmin-11.51_lnkmax4.61_ksteps200'
a_grid = np.load(f'{trisp_path}/a_arr_{ka_str}.npy')
lnk_grid = np.load(f'{trisp_path}/lnk1_arr_{ka_str}.npy')
trisp_LLLL = np.load(f'{trisp_path}/trisp_LLLL_{ka_str}.npy')
trisp_LLGL = np.load(f'{trisp_path}/trisp_LLGL_{ka_str}.npy')
trisp_LLGG = np.load(f'{trisp_path}/trisp_LLGG_{ka_str}.npy')
trisp_GLGL = np.load(f'{trisp_path}/trisp_GLGL_{ka_str}.npy')
trisp_GLGG = np.load(f'{trisp_path}/trisp_GLGG_{ka_str}.npy')
trisp_GGGG = np.load(f'{trisp_path}/trisp_GGGG_{ka_str}.npy')

a_ix = 0
plt.figure()
plt.semilogy(lnk_grid, np.diag(trisp_LLLL[a_ix]), label='LLLL')
plt.semilogy(lnk_grid, np.diag(trisp_LLGL[a_ix]), label='LLGL')
plt.semilogy(lnk_grid, np.diag(trisp_LLGG[a_ix]), label='LLGG')
plt.semilogy(lnk_grid, np.diag(trisp_GLGL[a_ix]), label='GLGL')
plt.semilogy(lnk_grid, np.diag(trisp_GLGG[a_ix]), label='GLGG')
plt.semilogy(lnk_grid, np.diag(trisp_GGGG[a_ix]), label='GGGG')
plt.legend()
plt.show()
# END SECOND TEST


tkka_path = (
    '/home/davide/Documenti/Lavoro/Programmi/Spaceborne/output/cache/trispectrum/cNG'
)
k_a_str_dense = 'amin0.25_amax0.98_asteps50_lnkmin-11.51_lnkmax4.61_ksteps200'
k_a_str_coarse = k_a_str_dense.replace('asteps50', 'asteps20')
k_a_str_coarse = k_a_str_coarse.replace('ksteps200', 'ksteps100')
probe_block = 'LLLL'

a_arr_dense = np.load(f'{tkka_path}/a_arr_{k_a_str_dense}.npy')
lk1_arr_dense = np.load(f'{tkka_path}/lnk1_arr_{k_a_str_dense}.npy')
lk2_arr_dense = np.load(f'{tkka_path}/lnk2_arr_{k_a_str_dense}.npy')
tkk_arr_dense = np.load(f'{tkka_path}/trisp_{probe_block}_{k_a_str_dense}.npy')

a_arr_coarse = np.load(f'{tkka_path}/a_arr_{k_a_str_coarse}.npy')
lk1_arr_coarse = np.load(f'{tkka_path}/lnk1_arr_{k_a_str_coarse}.npy')
lk2_arr_coarse = np.load(f'{tkka_path}/lnk2_arr_{k_a_str_coarse}.npy')
tkk_arr_coarse = np.load(f'{tkka_path}/trisp_{probe_block}_{k_a_str_coarse}.npy')

np.testing.assert_allclose(
    lk1_arr_dense,
    lk2_arr_dense,
    atol=0,
    rtol=1e-9,
    err_msg='lk1_arr and lk2_arr different',
)
np.testing.assert_allclose(
    lk1_arr_coarse,
    lk2_arr_coarse,
    atol=0,
    rtol=1e-9,
    err_msg='lk1_arr and lk2_arr different',
)


# interpolate coarse trisp on the dense grid
# tkk_spline = CubicSpline(a_arr_coarse, tkk_arr_coarse, axis=0, extrapolate=False)
interpolator = RegularGridInterpolator(
    (a_arr_coarse, lk1_arr_coarse, lk2_arr_coarse), tkk_arr_coarse, method='slinear'
)

a_dense_xx, lk1_dense_xx, lk2_dense_xx = np.meshgrid(
    a_arr_dense, lk1_arr_dense, lk2_arr_dense, indexing='ij'
)
tkk_arr_dense_interp = interpolator((a_dense_xx, lk1_dense_xx, lk2_dense_xx))


# interpolate dense trisp on the coarse grid
interpolator = RegularGridInterpolator(
    (a_arr_dense, lk1_arr_dense, lk2_arr_dense), tkk_arr_dense, method='slinear'
)

a_coarse_xx, lk1_coarse_xx, lk2_coarse_xx = np.meshgrid(
    a_arr_coarse, lk1_arr_coarse, lk2_arr_coarse, indexing='ij'
)
tkk_arr_coarse_interp = interpolator((a_coarse_xx, lk1_coarse_xx, lk2_coarse_xx))

a_ix = 10
plt.loglog(
    np.exp(lk1_arr_coarse),
    np.diag(tkk_arr_coarse_interp[a_ix, :, :]),
    label='dense interp',
)
plt.loglog(np.exp(lk1_arr_dense), np.diag(tkk_arr_dense[a_ix, :, :]), label='dense')
plt.loglog(np.exp(lk1_arr_coarse), np.diag(tkk_arr_coarse[a_ix, :, :]), label='coarse')
plt.legend()
plt.xlabel('ln(k)')
plt.ylabel(f'diag(tkk_arr[a_ix={a_ix}, :, :])')
plt.title(probe_block)


# check their difference
np.testing.assert_allclose(
    tkk_arr_dense_interp,
    tkk_arr_dense,
    atol=0,
    rtol=1e-9,
    err_msg='tkk_spline and tkk_arr_dense different',
)

sl.compare_arrays(tkk_arr_dense_interp[0, ...], tkk_arr_dense[0, ...])
sl.matshow(tkk_arr_dense_interp[0, ...] / tkk_arr_dense[0, ...], log=True)




sl.matshow(tkk_arr_dense[:, 0, :])


# plot diag
colors = plt.cm.viridis(np.linspace(0, 1, len(a_arr_dense)))
sm = cm.ScalarMappable(
    cmap=plt.cm.viridis, norm=Normalize(vmin=a_arr_dense.min(), vmax=a_arr_dense.max())
)
sm.set_array([])
fig, ax = plt.subplots()
for i, _ in enumerate(a_arr_dense):
    ax.semilogy(lk1_arr_dense, np.diag(tkk_arr_dense[i, :, :]), marker='.', c=colors[i])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('a')
ax.set_xlabel('ln(k)')
ax.set_ylabel('tkk_arr[i, :, :]')

# check simmetry
for i, a in enumerate(a_arr_dense):
    np.testing.assert_allclose(
        tkk_arr_dense[i, :, :],
        tkk_arr_dense[i, :, :].T,
        atol=0,
        rtol=1e-9,
        err_msg=f'Trispectrum not symmetric for a = {a}',
    )
