import pdb
import pickle
import sys
import time
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyccl as ccl
import yaml
from joblib import Parallel, delayed
from scipy.special import erf
import ray
from tqdm import tqdm

ray.shutdown()
ray.init()

# get project directory adn import useful modules
project_path = Path.cwd().parent

sys.path.append(f'../../common_lib_and_cfg/common_lib')
import my_module as mm
import cosmo_lib

sys.path.append(f'../../common_lib_and_cfg/common_config')
import ISTF_fid_params as ISTF_fid
import mpl_cfg


sys.path.append(f'../../cl_v2/bin')
import wf_cl_lib

matplotlib.use('Qt5Agg')
start_time = time.perf_counter()
plt.rcParams.update(mpl_cfg.mpl_rcParams_dict)


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def initialize_trispectrum():
    # ! =============================================== halo model =========================================================
    # TODO we're not sure about the values of Delta and rho_type
    # mass_def = ccl.halos.massdef.MassDef(Delta='vir', rho_type='matter', c_m_relation=name)
    # from https://ccl.readthedocs.io/en/latest/api/pyccl.halos.massdef.html?highlight=.halos.massdef.MassDef#pyccl.halos.massdef.MassDef200c

    # HALO MODEL PRESCRIPTIONS:
    # KiDS1000 Methodology: https://www.pure.ed.ac.uk/ws/portalfiles/portal/188893969/2007.01844v2.pdf, after (E.10)
    # Krause2017: https://arxiv.org/pdf/1601.05779.pdf
    # about the mass definition, the paper says:
    # "Throughout this paper we define halo properties using the over density ∆ = 200 ¯ρ, with ¯ρ the mean matter density"
    halomod_start_time = time.perf_counter()
    # mass definition
    if hm_recipe == 'KiDS1000':  # arXiv:2007.01844
        c_m = 'Duffy08'  # ! NOT SURE ABOUT THIS
        mass_def = ccl.halos.MassDef200m(c_m=c_m)
        c_M_relation = ccl.halos.concentration.ConcentrationDuffy08(mdef=mass_def)
    elif hm_recipe == 'Krause2017':  # arXiv:1601.05779
        c_m = 'Bhattacharya13'  # see paper, after Eq. 1
        mass_def = ccl.halos.MassDef200m(c_m=c_m)
        c_M_relation = ccl.halos.concentration.ConcentrationBhattacharya13(mdef=mass_def)  # above Eq. 12
    else:
        raise ValueError('Wrong choice of hm_recipe: it must be either "KiDS1000" or "Krause2017".')

    # TODO pass mass_def object? plus, understand what exactly is mass_def_strict

    # mass function
    massfunc = ccl.halos.hmfunc.MassFuncTinker10(cosmo_ccl, mass_def=mass_def, mass_def_strict=True)

    # halo bias
    hbias = ccl.halos.hbias.HaloBiasTinker10(cosmo_ccl, mass_def=mass_def, mass_def_strict=True)

    # concentration-mass relation

    # TODO understand better this object. We're calling the abstract class, is this ok?
    # HMCalculator
    hmc = ccl.halos.halo_model.HMCalculator(cosmo_ccl, massfunc, hbias, mass_def=mass_def,
                                            log10M_min=8.0, log10M_max=16.0, nlog10M=128,
                                            integration_method_M='simpson', k_min=1e-05)

    # halo profile
    halo_profile = ccl.halos.profiles.HaloProfileNFW(c_M_relation=c_M_relation,
                                                     fourier_analytic=True, projected_analytic=False,
                                                     cumul2d_analytic=False, truncated=True)

    # it was p_of_k_a=Pk, but it should use the LINEAR power spectrum, so we leave it as None (see documentation:
    # https://ccl.readthedocs.io/en/latest/api/pyccl.halos.halo_model.html?highlight=halomod_Tk3D_SSC#pyccl.halos.halo_model.halomod_Tk3D_SSC)
    # 🐛 bug fixed: normprof shoud be True
    # 🐛 bug fixed?: p_of_k_a=None instead of Pk

    tkka = ccl.halos.halo_model.halomod_Tk3D_SSC(cosmo_ccl, hmc,
                                                 prof1=halo_profile, prof2=None, prof12_2pt=None,
                                                 prof3=None, prof4=None, prof34_2pt=None,
                                                 normprof1=True, normprof2=True, normprof3=True, normprof4=True,
                                                 p_of_k_a=None, lk_arr=None, a_arr=None, extrap_order_lok=1,
                                                 extrap_order_hik=1, use_log=False)
    # assert False, 'should I use HaloProfileHOD for number counts???'  # TODO
    print('trispectrum computed in {:.2f} seconds'.format(time.perf_counter() - halomod_start_time))
    return tkka


def compute_cov_SSC_ccl(cosmo, kernel_A, kernel_B, kernel_C, kernel_D, ell, tkka, f_sky,
                        ind_AB, ind_CD, integration_method='spline'):
    zpairs_AB = ind_AB.shape[0]
    zpairs_CD = ind_CD.shape[0]

    # TODO switch off the integration method and see if it crashes
    # parallel version:
    start_time = time.perf_counter()
    cov_ng = Parallel(
        n_jobs=-1, backend='threading')(delayed(ccl.covariances.angular_cl_cov_SSC)(cosmo,
                                                                                    cltracer1=kernel_A[ind_AB[ij, -2]],
                                                                                    cltracer2=kernel_B[ind_AB[ij, -1]],
                                                                                    ell=ell, tkka=tkka,
                                                                                    sigma2_B=None, fsky=f_sky,
                                                                                    cltracer3=kernel_C[ind_CD[kl, -2]],
                                                                                    cltracer4=kernel_D[ind_CD[kl, -1]],
                                                                                    ell2=None,
                                                                                    integration_method=integration_method)
                                        for kl in tqdm(range(zpairs_CD))
                                        for ij in range(zpairs_AB))
    print(f'parallel version took {time.perf_counter() - start_time} seconds')

    cov_ng = np.array(cov_ng).transpose(1, 2, 0).reshape(nbl, nbl, zpairs_AB, zpairs_CD)

    return cov_ng


def compute_cov_cNG_ccl(cosmo, kernel_A, kernel_B, kernel_C, kernel_D, ell, tkka, f_sky,
                        ind_AB, ind_CD, integration_method='spline'):
    # TODO unify this with compute_cov_SSC_ccl
    zpairs_AB = ind_AB.shape[0]
    zpairs_CD = ind_CD.shape[0]

    # parallel version:
    start_time = time.perf_counter()
    cov_ng = Parallel(
        n_jobs=-1, backend='threading')(delayed(ccl.covariances.angular_cl_cov_cNG)(cosmo,
                                                                                    cltracer1=kernel_A[ind_AB[ij, -2]],
                                                                                    cltracer2=kernel_B[ind_AB[ij, -1]],
                                                                                    ell=ell, tkka=tkka, fsky=f_sky,
                                                                                    cltracer3=kernel_C[ind_CD[kl, -2]],
                                                                                    cltracer4=kernel_D[ind_CD[kl, -1]],
                                                                                    ell2=None,
                                                                                    integration_method=integration_method)
                                        for kl in tqdm(range(zpairs_CD))
                                        for ij in range(zpairs_AB))
    print(f'parallel version took {time.perf_counter() - start_time} seconds')

    cov_ng = np.array(cov_ng).transpose(1, 2, 0).reshape(nbl, nbl, zpairs_AB, zpairs_CD)

    return cov_ng


def compute_3x2pt_PyCCL(ng_function, cosmo, kernel_dict, ell, tkka, f_sky, integration_method,
                        probe_ordering, ind_dict, output_4D_array=True):
    cov_ng_3x2pt_dict_8D = {}
    for A, B in probe_ordering:
        for C, D in probe_ordering:
            print('3x2pt: working on probe combination ', A, B, C, D)
            cov_ng_3x2pt_dict_8D[A, B, C, D] = ng_function(cosmo=cosmo,
                                                           kernel_A=kernel_dict[A],
                                                           kernel_B=kernel_dict[B],
                                                           kernel_C=kernel_dict[C],
                                                           kernel_D=kernel_dict[D],
                                                           ell=ell, tkka=tkka, f_sky=f_sky,
                                                           ind_AB=ind_dict[A + B],
                                                           ind_CD=ind_dict[C + D],
                                                           integration_method=integration_method)

    if output_4D_array:
        return mm.cov_3x2pt_8D_dict_to_4D(cov_ng_3x2pt_dict_8D, probe_ordering)

    return cov_ng_3x2pt_dict_8D


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


# ! POTENTIAL ISSUES:
# 1. input files (WF, ell, a, pk...)
# 2. halo model recipe
# 3. ordering of the resulting covariance matrix
# * fanstastic collection of notebooks: https://github.com/LSSTDESC/CCLX


def compute_cov_ng_with_pyccl(probe, general_cfg, covariance_cfg):
# ! settings
with open('../../exact_SSC/config/cfg_exactSSC_ISTF.yml') as f:
    cfg = yaml.safe_load(f)

probes = cfg['probes']
which_NGs = cfg['which_NGs']
save_covs = cfg['save_covs']
hm_recipe = 'Krause2017'
GL_or_LG = cfg['GL_or_LG']
ell_min = cfg['ell_min']
ell_max = cfg['ell_max']
nbl = cfg['nbl']
zbins = cfg['zbins']
triu_tril = cfg['triu_tril']
row_col_major = cfg['row_col_major']
# use_ray = cfg['use_ray']  # TODO finish this!
z_grid = np.linspace(cfg['z_min_sigma2'], cfg['z_max_sigma2'], cfg['z_steps_sigma2'])
f_sky = general_cfg['fsky']
n_samples_wf = cfg['n_samples_wf']
get_3xtpt_cov_in_4D = cfg['get_3xtpt_cov_in_4D']
bias_model = cfg['bias_model']
# ! settings

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

# TODO plot kernels and cls to check that they make sense

# get number of redshift pairs
zpairs_auto, zpairs_cross, zpairs_3x2pt = mm.get_zpairs(zbins)
ind = mm.build_full_ind(triu_tril, row_col_major, zbins)
ind_auto = ind[:zpairs_auto, :]
ind_cross = ind[zpairs_auto:zpairs_auto + zpairs_cross, :]

assert GL_or_LG == 'GL', 'you should update ind_cross (used in ind_dict) for GL, but we work with GL...'

# ! compute cls, just as a test
ell_grid, _ = ell_utils.compute_ells(nbl, ell_min, ell_max, ell_grid_recipe)
np.savetxt(f'{project_path}/output/ell_values/ell_values_nbl{nbl}.txt', ell_grid)

# Create new Cosmology object with a given set of parameters. This keeps track of previously-computed cosmological
# functions
cosmo_ccl = wf_cl_lib.instantiate_ISTFfid_PyCCL_cosmo_obj()

# source redshift distribution, default ISTF values for bin edges & analytical prescription for the moment
niz_unnormalized_arr = np.asarray(
    [wf_cl_lib.niz_unnormalized_analytical(z_grid, zbin_idx) for zbin_idx in range(zbins)])
niz_normalized_arr = wf_cl_lib.normalize_niz_simps(niz_unnormalized_arr, z_grid).T
n_of_z = niz_normalized_arr

# galaxy bias
galaxy_bias_2d_array = wf_cl_lib.build_galaxy_bias_2d_arr(bias_values=None, z_values=None, zbins=zbins,
                                                          z_grid=z_grid, bias_model=bias_model,
                                                          plot_bias=False)

# IA bias
ia_bias_1d_array = wf_cl_lib.build_IA_bias_1d_arr(z_grid, input_lumin_ratio=None, cosmo=cosmo_ccl,
                                                  A_IA=None, eta_IA=None, beta_IA=None, C_IA=None, growth_factor=None,
                                                  Omega_m=None)

# # ! compute tracer objects
wf_lensing = [ccl.tracers.WeakLensingTracer(cosmo_ccl, dndz=(z_grid, n_of_z[:, zbin_idx]),
                                            ia_bias=(z_grid, ia_bias_1d_array), use_A_ia=False, n_samples=n_samples_wf)
              for zbin_idx in range(zbins)]

wf_galaxy = [ccl.tracers.NumberCountsTracer(cosmo_ccl, has_rsd=False, dndz=(z_grid, n_of_z[:, zbin_idx]),
                                            bias=(z_grid, galaxy_bias_2d_array[:, zbin_idx]),
                                            mag_bias=None, n_samples=n_samples_wf)
             for zbin_idx in range(zbins)]

# the cls are not needed, but just in case:
# cl_LL_3D = wf_cl_lib.cl_PyCCL(wf_lensing, wf_lensing, ell_grid, zbins, p_of_k_a=None, cosmo=cosmo_ccl)
# cl_GL_3D = wf_cl_lib.cl_PyCCL(wf_galaxy, wf_lensing, ell_grid, zbins, p_of_k_a=None, cosmo=cosmo_ccl)
# cl_GG_3D = wf_cl_lib.cl_PyCCL(wf_galaxy, wf_galaxy, ell_grid, zbins, p_of_k_a=None, cosmo=cosmo_ccl)


# notebook for mass_relations: https://github.com/LSSTDESC/CCLX/blob/master/Halo-mass-function-example.ipynb
# Cl notebook: https://github.com/LSSTDESC/CCL/blob/v2.0.1/examples/3x2demo.ipynb

tkka = initialize_trispectrum()

# covariance ordering stuff
probe_ordering = (('L', 'L'), (GL_or_LG[0], GL_or_LG[1]), ('G', 'G'))
probe_ordering = (('G', 'L'),)

# convenience dictionaries
ind_dict = {
    'LL': ind_auto,
    'GG': ind_auto,
    'GL': ind_cross
}

probe_idx_dict = {
    'L': 0,
    'G': 1
}

kernel_dict = {
    'L': wf_lensing,
    'G': wf_galaxy
}

# integration_method_dict = {
#     'LL': {
#         'SSC': 'spline',
#         'cNG': 'spline',
#     },
#     'GG': {
#         'SSC': 'qag_quad',
#         'cNG': 'qag_quad',
#     },
#     '3x2pt': {
#         'SSC': 'qag_quad',
#         'cNG': 'spline',
#     }
# }

integration_method_dict = {
    'LL': {
        'SSC': 'qag_quad',
        'cNG': 'qag_quad',
    },
    'GG': {
        'SSC': 'qag_quad',
        'cNG': 'qag_quad',
    },
    '3x2pt': {
        'SSC': 'qag_quad',
        'cNG': 'qag_quad',
    }
}

for probe in probes:
    for which_NG in which_NGs:

        assert probe in ['LL', 'GG', '3x2pt'], 'probe must be either LL, GG, or 3x2pt'
        assert which_NG in ['SSC', 'cNG'], 'which_NG must be either SSC or cNG'
        assert ell_grid_recipe in ['ISTF', 'ISTNL'], 'ell_grid_recipe must be either ISTF or ISTNL'

        if ell_grid_recipe == 'ISTNL' and nbl != 20:
            print('Warning: ISTNL uses 20 ell bins')

        if probe == 'LL':
            kernel = wf_lensing
        elif probe == 'GG':
            kernel = wf_galaxy

        # just a check on the settings
        print(f'\n****************** settings ****************'
              f'\nprobe = {probe}\nwhich_NG = {which_NG}'
              f'\nintegration_method = {integration_method_dict[probe][which_NG]}'
              f'\nwhich_ells = {ell_grid_recipe}\nnbl = {nbl}\nhm_recipe = {hm_recipe}')

        # ! note that the ordering is such that out[i2, i1] = Cov(ell2[i2], ell[i1]). Transpose 1st 2 dimensions??
        # * ok: the check that the matrix symmetric in ell1, ell2 is below
        # print(f'check: is cov_SSC_{probe}[ell1, ell2, ...] == cov_SSC_{probe}[ell2, ell1, ...]?', np.allclose(cov_6D, np.transpose(cov_6D, (1, 0, 2, 3, 4, 5)), rtol=1e-7, atol=0))

        # ! =============================================== compute covs ===============================================

        if which_NG == 'SSC':
            ng_function = compute_cov_SSC_ccl
        elif which_NG == 'cNG':
            ng_function = compute_cov_cNG_ccl
        else:
            raise ValueError('which_NG must be either SSC or cNG')

        if probe in ['LL', 'GG']:
            assert probe[0] == probe[1], 'probe must be either LL or GG'

            kernel_A = kernel_dict[probe[0]]
            kernel_B = kernel_dict[probe[1]]
            kernel_C = kernel_dict[probe[0]]
            kernel_D = kernel_dict[probe[1]]
            ind_AB = ind_dict[probe[0] + probe[1]]
            ind_CD = ind_dict[probe[0] + probe[1]]

            cov_ng_4D = ng_function(cosmo_ccl,
                                    kernel_A=kernel_A,
                                    kernel_B=kernel_B,
                                    kernel_C=kernel_C,
                                    kernel_D=kernel_D,
                                    ell=ell_grid, tkka=tkka, f_sky=f_sky,
                                    ind_AB=ind_AB, ind_CD=ind_CD,
                                    integration_method=integration_method_dict[probe][which_NG])

        elif probe == '3x2pt':
            # TODO remove this if statement and use the same code for all probes
            cov_ng_4D = compute_3x2pt_PyCCL(ng_function=ng_function, cosmo=cosmo_ccl,
                                            kernel_dict=kernel_dict,
                                            ell=ell_grid, tkka=tkka, f_sky=f_sky,
                                            probe_ordering=probe_ordering,
                                            ind_dict=ind_dict,
                                            output_4D_array=True,
                                            integration_method=integration_method_dict[probe][which_NG])

            cov_ng_2D = mm.cov_4D_to_2D(cov_ng_4D)

        else:
            raise ValueError('probe must be either LL, GG, or 3x2pt')

        if save_covs:
            output_folder = f'{project_path}/output/covmat/after_script_update'
            filename = f'cov_PyCCL_{which_NG}_{probe}_nbl{nbl}_ellmax{ell_max}_HMrecipe{hm_recipe}'

            np.savez_compressed(f'{output_folder}/{filename}_4D.npz', cov_ng_4D)
            cov_6D = mm.cov_4D_to_6D(cov_ng_4D, nbl, zbins, 'LL', ind)

            # mm.test_folder_content(output_folder, output_folder + 'benchmarks', 'npy', verbose=False, rtol=1e-10)

print('done')
