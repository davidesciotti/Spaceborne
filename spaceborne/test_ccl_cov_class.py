import sys
import time
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import os
import numpy as np
import warnings
import matplotlib.lines as mlines
import matplotlib as mpl
import gc
import matplotlib.gridspec as gridspec
import yaml
from scipy.ndimage import gaussian_filter1d
import pprint
from copy import deepcopy
import numpy.testing as npt
import multiprocessing
pp = pprint.PrettyPrinter(indent=4)

ROOT = os.getenv('ROOT')
SB_ROOT = f'{ROOT}/Spaceborne'

# project modules
sys.path.append(SB_ROOT)
import bin.ell_values as ell_utils
import bin.cl_preprocessing as cl_utils
import bin.compute_Sijkl as Sijkl_utils
import bin.covariance as covmat_utils
import bin.fisher_matrix as FM_utils
import bin.my_module as mm
import bin.cosmo_lib as csmlib
import bin.wf_cl_lib as wf_cl_lib
import bin.plots_FM_running as plot_utils
import common_cfg.mpl_cfg as mpl_cfg


# job config
import jobs.SPV3.config.config_SPV3 as cfg

# mpl.use('Agg')
plt.rcParams.update(mpl_cfg.mpl_rcParams_dict)
script_start_time = time.perf_counter()

num_cores = multiprocessing.cpu_count()
os.environ['OMP_NUM_THREADS'] = str(num_cores)

import spaceborne.pyccl_interface as pyccl_interface


general_cfg = cfg.general_cfg
covariance_cfg = cfg.covariance_cfg
Sijkl_cfg = cfg.Sijkl_cfg
fm_cfg = cfg.FM_cfg
pyccl_cfg = covariance_cfg['PyCCL_cfg']

zbins = general_cfg['zbins']

with open(general_cfg['fid_yaml_filename'].format(zbins=zbins)) as f:
    fid_pars_dict = yaml.safe_load(f)
flat_fid_pars_dict = mm.flatten_dict(fid_pars_dict)
general_cfg['flat_fid_pars_dict'] = flat_fid_pars_dict

ell_max_WL = general_cfg['ell_max_WL']
ell_max_GC = general_cfg['ell_max_GC']
ell_max_3x2pt = general_cfg['ell_max_3x2pt']
magcut_source = general_cfg['magcut_source']
magcut_lens = general_cfg['magcut_lens']
zcut_source = general_cfg['zcut_source']
zcut_lens = general_cfg['zcut_lens']
flat_or_nonflat = general_cfg['flat_or_nonflat']
center_or_min = general_cfg['center_or_min']
zmax = int(general_cfg['zmax'] * 10)
triu_tril = covariance_cfg['triu_tril']
row_col_major = covariance_cfg['row_col_major']
GL_or_LG = covariance_cfg['GL_or_LG']
n_probes = general_cfg['n_probes']
which_pk = general_cfg['which_pk']
idIA = general_cfg['idIA']
idB = general_cfg['idB']
idM = general_cfg['idM']
idR = general_cfg['idR']
idBM = general_cfg['idBM']
which_ng_cov_suffix = 'G' + ''.join(covariance_cfg[covariance_cfg['ng_cov_code'] + '_cfg']['which_ng_cov'])
bnt_transform = general_cfg['BNT_transform']
shift_nz_interpolation_kind = covariance_cfg['shift_nz_interpolation_kind']
nz_gaussian_smoothing = covariance_cfg['nz_gaussian_smoothing']  # does not seem to have a large effect...
nz_gaussian_smoothing_sigma = covariance_cfg['nz_gaussian_smoothing_sigma']
shift_nz = covariance_cfg['shift_nz']  # ! are vincenzo's kernels shifted?? it looks like they are not
normalize_shifted_nz = covariance_cfg['normalize_shifted_nz']
compute_bnt_with_shifted_nz_for_zcuts = covariance_cfg['compute_bnt_with_shifted_nz_for_zcuts']  # ! let's test this
include_ia_in_bnt_kernel_for_zcuts = covariance_cfg['include_ia_in_bnt_kernel_for_zcuts']
nbl_WL_opt = general_cfg['nbl_WL_opt']
colors = cm.rainbow(np.linspace(0, 1, zbins))

with open(general_cfg['fid_yaml_filename'].format(zbins=zbins)) as f:
    fid_pars_dict = yaml.safe_load(f)
flat_fid_pars_dict = mm.flatten_dict(fid_pars_dict)
general_cfg['flat_fid_pars_dict'] = flat_fid_pars_dict


# this is just to make the .format() more compact
variable_specs = {'zbins': zbins, 'magcut_lens': magcut_lens,
                  'zcut_lens': zcut_lens,
                  'magcut_source': magcut_source, 'zcut_source': zcut_source, 'zmax': zmax,
                  'ell_max_WL': ell_max_WL, 'ell_max_GC': ell_max_GC, 'ell_max_3x2pt': ell_max_3x2pt,
                  'idIA': idIA, 'idB': idB, 'idM': idM, 'idR': idR, 'idBM': idBM,
                  'flat_or_nonflat': flat_or_nonflat,
                  'which_pk': which_pk, 'BNT_transform': bnt_transform,
                  'which_ng_cov': which_ng_cov_suffix,
                  'ng_cov_code': covariance_cfg['SSC_code'],
                  'EP_or_ED': general_cfg['EP_or_ED'],
                  }


ell_grid = np.geomspace(10, 5000, 32)
nofz_folder = covariance_cfg["nofz_folder"]
nofz_filename = covariance_cfg["nofz_filename"].format(**variable_specs)
n_of_z_load = np.genfromtxt(f'{nofz_folder}/{nofz_filename}')

# initialize class, ie initialize cosmo
ccl_obj = pyccl_interface.PycclClass(fid_pars_dict)

# set n_z
ccl_obj.set_nz(n_of_z_load)
ccl_obj.check_nz_tuple(zbins)

# set ia_bias
ccl_obj.set_ia_bias_tuple()

# TODO here I'm still setting some cfgs, which do not go in the Class init
ccl_obj.zbins = zbins  # TODO is this inelegant?

# SET GALAXY BIAS
if general_cfg['which_forecast'] == 'SPV3':
    ccl_obj.set_gal_bias_tuple_spv3(magcut_lens=general_cfg['magcut_lens'] / 10)


elif general_cfg['which_forecast'] == 'ISTF':
    bias_func_str = general_cfg['bias_function']
    bias_model = general_cfg['bias_model']
    ccl_obj.set_gal_bias_tuple_istf(bias_function_str=bias_func_str, bias_model=bias_model)


# set pk
# this is a test to use the actual P(k) from the input files, but the agreement gets much worse
if general_cfg['which_forecast'] == 'SPV3' and pyccl_cfg['which_pk_for_pyccl'] == 'CLOE':
    cloe_pk_filename = general_cfg['CLOE_pk_filename'].format(
        flat_or_nonflat=general_cfg['flat_or_nonflat'], which_pk=general_cfg['which_pk'])
    ccl_obj.p_of_k_a = ccl_obj.pk_obj_from_file(pk_filename=cloe_pk_filename)
    # TODO finish implementing this
    raise NotImplementedError('range needs to be extended to higher redshifts to match tkka grid (probably larger k range too), \
        some other small consistency checks needed')

elif general_cfg['which_forecast'] == 'SPV3' and pyccl_cfg['which_pk_for_pyccl'] == 'PyCCL':
    ccl_obj.p_of_k_a = 'delta_matter:delta_matter'

elif general_cfg['which_forecast'] == 'ISTF':
    ccl_obj.p_of_k_a = 'delta_matter:delta_matter'

# save gal bias for Robert - not needed at the moment
gal_bias_table_ascii_name = f'{covariance_cfg["nofz_folder"]}/gal_bias_table_{general_cfg["which_forecast"]}.ascii'
ccl_obj.save_gal_bias_table_ascii(filename=gal_bias_table_ascii_name)

# set mag bias
ccl_obj.set_mag_bias_tuple(has_magnification_bias=general_cfg['has_magnification_bias'], magcut_lens=general_cfg['magcut_lens'] / 10)


# set kernel arrays and objects
ccl_obj.set_kernel_obj(general_cfg['has_rsd'], covariance_cfg['PyCCL_cfg']['n_samples_wf'])
ccl_obj.set_kernel_arr(z_grid_wf=ccl_obj.zgrid_nz, has_magnification_bias=general_cfg['has_magnification_bias'])

ccl_obj.compute_cls(ell_grid, ccl_obj.p_of_k_a, 'spline')

zmin = 1e-3
zmax = 3
zsteps = 1000
ccl_obj.set_sigma2_b(zmin, zmax, zsteps, covariance_cfg['fsky'], pyccl_cfg)

# for zi in range(zbins):
plt.plot(ccl_obj.sigma2_b_tuple[0], ccl_obj.sigma2_b_tuple[1], label=f'lensing')
# plt.plot(ccl_obj.ell_grid, ccl_obj.cl_gl_3d[:, zi, zi], color=colors[zi], label=f'lensing', ls='--')
# plt.plot(ccl_obj.ell_grid, ccl_obj.cl_gg_3d[:, zi, zi], color=colors[zi], label=f'lensing', ls=':')
# plt.xscale('log')
plt.yscale('log')
