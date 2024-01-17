import pickle
import sys
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

sys.path.append('/home/cosmo/davide.sciotti/data/common_lib_and_cfg/common_lib')
import common_lib.my_module as mm
import common_lib.cosmo_lib as csmlib
import common_lib.wf_cl_lib as wf_cl_lib
import common_cfg.ISTF_fid_params as ISTFfid
import common_cfg.mpl_cfg as mpl_cfg


sys.path.append('/home/cosmo/davide.sciotti/data/SSC_restructured_v2/bin')
import ell_values

matplotlib.use('Qt5Agg')
plt.rcParams.update(mpl_cfg.mpl_rcParams_dict)

cfg = mm.read_yaml('../cfg/cfg_pyccl_pipeline.yml')
fiducial_pars_dict_nested = mm.read_yaml(
    '/home/cosmo/davide.sciotti/data/common_lib_and_cfg/common_cfg/ISTF_fiducial_params.yml')
fiducial_pars_dict = mm.flatten_dict(fiducial_pars_dict_nested)

zbins = cfg['zbins']
ell_min = cfg['ell_min']
ell_max = cfg['ell_max']
nbl = cfg['nbl']
zpairs_auto, zpairs_cross, zpairs_3x2pt = mm.get_zpairs(zbins)

# colormap
cmap = matplotlib.cm.get_cmap("rainbow")
colors = [cmap(i) for i in np.linspace(0, 1, zbins)]

cosmo_ccl = csmlib.instantiate_cosmo_ccl_obj(fiducial_pars_dict)
n_of_z = np.genfromtxt('/home/cosmo/davide.sciotti/data/CLOE_validation/data/n_of_z/nzTabISTF.dat')
z_grid_nz = n_of_z[:, 0]
n_of_z = n_of_z[:, 1:]
dndz = (z_grid_nz, n_of_z)

ell_LL, _ = ell_values.compute_ells(nbl, ell_min, ell_max, 'ISTF')
ell_GL, _ = ell_values.compute_ells(nbl, ell_min, ell_max, 'ISTF')
ell_GG, _ = ell_values.compute_ells(nbl, ell_min, ell_max, 'ISTF')

# ell_LL, ell_GL, ell_GG = ell_vinc, ell_vinc, ell_vinc  # for a better test


list_params_to_vary = list(fiducial_pars_dict_nested['FM_ordered_params'].keys())

dz_param_list = [f'dz{zi:02d}_photo' for zi in range(1, zbins + 1)]
mag_param_list = [f'm{zi:02d}_photo' for zi in range(1, zbins + 1)]
elements_to_remove = ['Om_Lambda0'] + dz_param_list + mag_param_list
list_params_to_vary = [param for param in list_params_to_vary if param not in elements_to_remove]

list_params_to_vary = ['w_a']  # as a test
# list_params_to_vary.remove('w_a')  # as a test

cl_LL, cl_GL, cl_GG, dcl_LL, dcl_GL, dcl_GG = wf_cl_lib.cls_and_derivatives(
    fiducial_pars_dict, list_params_to_vary, zbins, dndz, ell_LL, ell_GL, ell_GG,
    'step-wise', pk=None, use_only_flat_models=True)


fiducial_idx = cl_LL['Om_m0'].shape[0] // 2
for param in list_params_to_vary:
    try:
        np.testing.assert_allclose(cl_LL['Om_m0'][fiducial_idx+1, ...], cl_LL[param][fiducial_idx, ...], rtol=1e-5, atol=0)
    except AssertionError:
        print(f'It looks like the index corresponding to the fiducial cosmology in the cl derivatives is not the {dif}th {param}')
        raise

for param in list_params_to_vary:
    np.save(f'../output/cl_LL_nbl{nbl}_ellmax{ell_max}.pkl', cl_LL)
    np.save(f'../output/cl_GL_nbl{nbl}_ellmax{ell_max}.pkl', cl_GL)
    np.save(f'../output/cl_GG_nbl{nbl}_ellmax{ell_max}.pkl', cl_GG)
    np.save(f'../output/dcl_LL_nbl{nbl}_ellmax{ell_max}.pkl', dcl_LL)
    np.save(f'../output/dcl_GL_nbl{nbl}_ellmax{ell_max}.pkl', dcl_GL)
    np.save(f'../output/dcl_GG_nbl{nbl}_ellmax{ell_max}.pkl', dcl_GG)
np.savetxt('../output/ell_LL.txt', ell_LL)
np.savetxt('../output/ell_GL.txt', ell_GL)
np.savetxt('../output/ell_GG.txt', ell_GG)

# compare against vincenzo
param_to_plot = 'A_IA'
path_vinc = '/home/cosmo/davide.sciotti/data/common_data/vincenzo/14may/CijDers/EP10'
dcl_vinc_LL_2d = np.genfromtxt(f'{path_vinc}/dCijLLdAia-GR-Flat-eNLA-NA.dat')
dcl_vinc_GL_2d = np.genfromtxt(f'{path_vinc}/dCijGLdAia-GR-Flat-eNLA-NA.dat')
dcl_vinc_GG_2d = np.genfromtxt(f'{path_vinc}/dCijGGdAia-GR-Flat-eNLA-NA.dat')
ell_vinc = 10 ** dcl_vinc_LL_2d[:, 0]  # TODO use these to compute my derivatives for a better comparison
nbl_vinc = len(ell_vinc)

dcl_vinc_LL_2d = dcl_vinc_LL_2d[:, 1:]
dcl_vinc_GL_2d = dcl_vinc_GL_2d[:, 1:]
dcl_vinc_GG_2d = dcl_vinc_GG_2d[:, 1:]

zj = 8
ell_idx = 5

dcl_vinc_LL_3d = mm.cl_2D_to_3D_symmetric(dcl_vinc_LL_2d, nbl_vinc, zpairs_auto, zbins)
dcl_vinc_GL_3d = mm.cl_2D_to_3D_asymmetric(dcl_vinc_GL_2d, nbl_vinc, zbins, order='C')
dcl_vinc_GG_3d = mm.cl_2D_to_3D_symmetric(dcl_vinc_GG_2d, nbl_vinc, zpairs_auto, zbins)

# interpolate vincenzo on my grid
dcl_vinc_LL_3d_func = interp1d(ell_vinc, dcl_vinc_LL_3d, kind='linear', axis=0)
dcl_vinc_GL_3d_func = interp1d(ell_vinc, dcl_vinc_GL_3d, kind='linear', axis=0)
dcl_vinc_GG_3d_func = interp1d(ell_vinc, dcl_vinc_GG_3d, kind='linear', axis=0)
dcl_vinc_LL_3d_interp = dcl_vinc_LL_3d_func(ell_LL)
dcl_vinc_GL_3d_interp = dcl_vinc_GL_3d_func(ell_GL)
dcl_vinc_GG_3d_interp = dcl_vinc_GG_3d_func(ell_GG)

fig, axs = plt.subplots(1, 3, figsize=(18, 6))
for zi in range(zbins):
    axs[0].plot(ell_LL, np.abs(dcl_LL[param_to_plot][:, zi, zj]), c=colors[zi], alpha=0.6)
    axs[0].plot(ell_LL, np.abs(dcl_vinc_LL_3d_interp[:, zi, zj]), '--', c=colors[zi], alpha=0.6)
axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[0].set_title('LL')

# Plot for GL
for zi in range(zbins):
    axs[1].plot(ell_GL, np.abs(dcl_GL[param_to_plot][:, zi, zj]), c=colors[zi], alpha=0.6)
    axs[1].plot(ell_GL, np.abs(dcl_vinc_GL_3d_interp[:, zi, zj]), '--', c=colors[zi], alpha=0.6)
axs[1].set_xscale('log')
axs[1].set_yscale('log')
axs[1].set_title('GL')

# Plot for GG
for zi in range(zbins):
    axs[2].plot(ell_GG, np.abs(dcl_GG[param_to_plot][:, zi, zj]), c=colors[zi], alpha=0.6)
    axs[2].plot(ell_GG, np.abs(dcl_vinc_GG_3d_interp[:, zi, zj]), '--', c=colors[zi], alpha=0.6)
axs[2].set_xscale('log')
axs[2].set_yscale('log')
axs[2].set_title('GG')

plt.tight_layout()
plt.show()

# ells = np.logspace(np.log10(ell_min), np.log10(ell_max), nbl + 1)
# cl_LL_3d = wf_cl_lib.cl_PyCCL(wl_kernel, wl_kernel, ells, zbins, p_of_k_a=None, cosmo=cosmo_ccl)
# cl_GL_3d = wf_cl_lib.cl_PyCCL(gc_kernel, wl_kernel, ells, zbins, p_of_k_a=None, cosmo=cosmo_ccl)
# cl_GG_3d = wf_cl_lib.cl_PyCCL(gc_kernel, gc_kernel, ells, zbins, p_of_k_a=None, cosmo=cosmo_ccl)

# for zi in range(zbins):
#     plt.plot(ells, cl_GG_3d[:, zi, zi], label=f'zbin {zi}')
