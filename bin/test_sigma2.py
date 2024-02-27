# ! remove from here
import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import sys

import os
from tqdm import tqdm

import yaml
ROOT = os.getenv('ROOT')
SB_ROOT = f'{ROOT}/Spaceborne'

# project modules
sys.path.append(SB_ROOT)
import bin.my_module as mm
import bin.cosmo_lib as cosmo_lib
import pyccl as ccl
import bin.sigma2_SSC as sigma2_SSC
import bin.mask_fits_to_cl as mask_fits_to_cl
import healpy as hp
import PySSC
# job config
import jobs.SPV3.config.config_SPV3 as cfg

covariance_cfg = cfg.covariance_cfg
pyccl_cfg = covariance_cfg['PyCCL_cfg']


area_deg2 = 15000  # 14700, 15000, 30000 or 5, for the various tests


fsky = cosmo_lib.deg2_to_fsky(area_deg2)

# load a yaml file
with open(f'{SB_ROOT}/common_cfg/SPV3_fiducial_params_magcut245_zbins13.yml', 'r') as f:
    fid_pars_dict = yaml.load(f, Loader=yaml.FullLoader)

flat_fid_pars_dict = mm.flatten_dict(fid_pars_dict)
flat_fid_pars_dict_ccl = cosmo_lib.map_keys(flat_fid_pars_dict, key_mapping=None)
cosmo_ccl = cosmo_lib.instantiate_cosmo_ccl_obj(flat_fid_pars_dict_ccl,
                                                fid_pars_dict['other_params']['camb_extra_parameters'])



k_grid_sigma2 = np.logspace(-4, 2, 5000)

# this is not linearly spaced, in a!!!
z_grid_tkka = np.linspace(pyccl_cfg['z_grid_tkka_min'],
                          pyccl_cfg['z_grid_tkka_max'],
                          pyccl_cfg['z_grid_tkka_steps'])

z_grid_sigma2_B = z_grid_tkka
a_grid_sigma2_B = cosmo_lib.z_to_a(z_grid_sigma2_B)[::-1]

# this is; better option
a_grid_sigma2_B = np.linspace(
    cosmo_lib.z_to_a(pyccl_cfg['z_grid_tkka_max']),
    cosmo_lib.z_to_a(pyccl_cfg['z_grid_tkka_min']),
    pyccl_cfg['z_grid_tkka_steps'])

# z_grid_sigma2_B = z_grid_tkka
z_grid_sigma2_B = cosmo_lib.a_to_z(a_grid_sigma2_B)[::-1]

area_deg2 = int(cosmo_lib.fsky_to_deg2(fsky))
nside = 2048
coord = ['C', 'E']


ell_mask = np.load(pyccl_cfg['ell_mask_filename'].format(area_deg2=area_deg2, nside=nside))
cl_mask = np.load(pyccl_cfg['cl_mask_filename'].format(area_deg2=area_deg2, nside=nside))

# mask = mask_fits_to_cl.generate_polar_cap(area_deg2, nside)
# hp.mollview(mask, coord=coord, title='mask', cmap='inferno_r')
# ell_mask, cl_mask, fsky_mask = mask_fits_to_cl.get_mask_quantities(clmask=None, mask=mask, mask2=None, verbose=True)
# assert mm.percent_diff(fsky_mask, fsky, abs_value=True) < 1, 'fsky is not correct'

cl_mask_norm = cl_mask * (2 * ell_mask + 1) / (4 * np.pi * fsky)**2

cosmo_ccl.compute_linear_power()
p_of_k_a = cosmo_ccl.get_linear_power()

# mask
sigma2_B = ccl.covariances.sigma2_B_from_mask(
    cosmo=cosmo_ccl, a_arr=a_grid_sigma2_B, mask_wl=cl_mask_norm, p_of_k_a=p_of_k_a)
sigma2_B_tuple_ccl_mask = (a_grid_sigma2_B, sigma2_B)

# disc
sigma2_B = ccl.covariances.sigma2_B_disc(
    cosmo=cosmo_ccl, a_arr=a_grid_sigma2_B, fsky=fsky, p_of_k_a=p_of_k_a)
sigma2_B_tuple_ccl_disc = (a_grid_sigma2_B, sigma2_B)

# spaceborne, curved full sky
sigma2_B = np.array([sigma2_SSC.sigma2_func(z1, z1, k_grid_sigma2, cosmo_ccl, 'full-curved-sky', ell_mask=None, cl_mask=None)
                     for z1 in tqdm(z_grid_sigma2_B)])
sigma2_B_tuple_sb_curved_full_sky = (a_grid_sigma2_B, sigma2_B[::-1])

# spaceborne, mask
sigma2_B = np.array([sigma2_SSC.sigma2_func(z1, z1, k_grid_sigma2, cosmo_ccl, 'mask', ell_mask=ell_mask, cl_mask=cl_mask)
                     for z1 in tqdm(z_grid_sigma2_B)])
sigma2_B_tuple_sb_mask = (a_grid_sigma2_B, sigma2_B[::-1])

# PySSC, curved full sky
class_cosmo_params = {'omega_b': 0.022, 'omega_cdm': 0.12, 'H0': 67.,
                      'n_s': 0.96, 'A_s': 2.035e-9, 'output': 'mPk', 'P_k_max_h/Mpc': 1000}
sigma2_B = np.diag(PySSC.sigma2_fullsky(z_grid_sigma2_B[2:], class_cosmo_params))
sigma2_B_tuple_pyssc_curved_full_sky = (a_grid_sigma2_B[:-2], sigma2_B[::-1])


# file (curved, full sky)
# z_grid_sigma2_B_import = np.load(pyccl_cfg['z_grid_sigma2_B_filename'])
# sigma2_B = np.load(pyccl_cfg['sigma2_B_filename'])
# sigma2_B = np.diag(sigma2_B) if sigma2_B.ndim == 2 else sigma2_B

# sigma2_B_interp_func = interp1d(z_grid_sigma2_B_import, sigma2_B, kind='linear')
# sigma2_B = sigma2_B_interp_func(z_grid_sigma2_B)[::-1]  # flip it, it's a function of a

# sigma2_B_tuple_file = (a_grid_sigma2_B, sigma2_B)

plt.figure()
# plt.plot(sigma2_B_tuple_ccl_mask[0], sigma2_B_tuple_ccl_mask[1], label='mask')
# plt.plot(sigma2_B_tuple_ccl_disc[0], sigma2_B_tuple_ccl_disc[1], label='disc', ls='--')
# plt.plot(sigma2_B_tuple_file[0], sigma2_B_tuple_file[1], label='dav')
# plt.plot(sigma2_B_tuple_file[0], sigma2_B_tuple_file[1] / fsky, label='dav/fsky')
# plt.plot(sigma2_B_tuple_file[0], sigma2_B_tuple_sb_mask[1], label='sb mask', ls='--')
plt.plot(sigma2_B_tuple_sb_curved_full_sky[0], sigma2_B_tuple_sb_curved_full_sky[1], label='sb curved full sky', ls='-', marker='o')
# plt.plot(sigma2_B_tuple_file[0], sigma2_B_tuple_sb_curved_full_sky[1] / fsky, label='sb curved full sky/fsky', ls='--')
plt.plot(sigma2_B_tuple_pyssc_curved_full_sky[0],
         sigma2_B_tuple_pyssc_curved_full_sky[1], label='pyssc curved full sky', ls='-', marker='o')
plt.xlabel('$a$')
plt.ylabel('$\sigma^2_B(a)$')
plt.yscale('log')
plt.legend()
plt.axvline(cosmo_lib.z_to_a(0.5), ls='--', color='k')
plt.show()

assert False, 'stop here'


# sigma2 from ccl (can be imported from file, as well...)
a_arr_sigma2_ccl_new_import_mask = np.load(f'{ROOT}/exact_SSC/output/sigma2/PyCCL/a_arrmask.npy')
sigma2_ccl_new_import_mask = np.load(f'{ROOT}/exact_SSC/output/sigma2/PyCCL/s2b_arrmask.npy')

a_arr_sigma2_ccl_new_import_None = np.load(f'{ROOT}/exact_SSC/output/sigma2/PyCCL/a_arrNone.npy')
sigma2_ccl_new_import_None = np.load(f'{ROOT}/exact_SSC/output/sigma2/PyCCL/s2b_arrNone.npy')

a_arr_sigma2_ccl_new_import_zsteps3000_ISTF = np.load(f'{ROOT}/exact_SSC/output/sigma2/PyCCL/a_arrzsteps3000_ISTF.npy')
sigma2_ccl_new_import_zsteps3000_ISTF = np.load(f'{ROOT}/exact_SSC/output/sigma2/PyCCL/s2b_arrzsteps3000_ISTF.npy')

plt.plot(a_arr_sigma2_ccl_new_import_mask, sigma2_ccl_new_import_mask, label='mask')
plt.plot(a_arr_sigma2_ccl_new_import_None, sigma2_ccl_new_import_None, label='None')
plt.plot(a_arr_sigma2_ccl_new_import_zsteps3000_ISTF, sigma2_ccl_new_import_zsteps3000_ISTF, label='zsteps3000_ISTF')
plt.legend()
plt.yscale('log')
plt.xscale('log')


z_grid_dav_ISTF = np.load(
    f'{ROOT}/exact_SSC/output/sigma2/z_grid_sigma2_zsteps3000_ISTF.npy')
sigma2_dav_ISTF = np.load(f'{ROOT}/exact_SSC/output/sigma2/sigma2_zsteps3000_ISTF.npy')

# TODO the path in the cfg dile seems to be wrong, it points to the folder above, but no interpolation
z_grid_dav_SPV3 = np.load(
    f'{ROOT}/exact_SSC/output/SPV3/separate_universe/jan_2024/d2ClAB_dVddeltab/z_grid_ssc_integrand_zsteps2899.npy')
sigma2_dav_SPV3 = np.load(
    f'{ROOT}/exact_SSC/output/SPV3/separate_universe/jan_2024/d2ClAB_dVddeltab/sigma2_zsteps2899_SPV3.npy')

z_grid_sigma2_SPV3_3000 = np.load(ROOT + '/exact_SSC/output/sigma2/z_grid_sigma2_zsteps3000_SPV3_serial.npy')
sigma2_SPV3_3000 /= np.load(ROOT + '/exact_SSC/output/sigma2/sigma2_zsteps3000_SPV3_serial.npy')


sigma2_dav_SPV3_diag = np.diag(sigma2_dav_SPV3)
sigma2_dav_ISTF_diag = np.diag(sigma2_dav_ISTF)

sigma2_dav_SPV3_diag_interp_func = scipy.interpolate.interp1d(
    z_grid_dav_SPV3, sigma2_dav_SPV3_diag, kind='linear', fill_value='extrapolate')
sigma2_dav_SPV3_diag_interp = sigma2_dav_SPV3_diag_interp_func(z_grid_tkka)


plt.figure()
plt.plot(z_grid_tkka, sigma2_B_ccl_SPV3, label='ccl SPV3')
plt.plot(z_grid_tkka, sigma2_B_ccl_ISTF, label='ccl ISTF')
plt.plot(z_grid_tkka, sigma2_B_ccl_SPV3_polar_cap, label='ccl polar cap')
plt.plot(z_grid_dav_SPV3, np.diag(sigma2_dav_SPV3), label='dav SPV3')
plt.plot(z_grid_dav_ISTF, np.diag(sigma2_dav_ISTF), label='dav ISTF', ls='--')
plt.plot(z_grid_sigma2_SPV3_3000, sigma2_SPV3_3000, label='dav SPV3 3000', ls='--')
plt.yscale('log')
plt.legend()
plt.xlabel('z')
plt.ylabel(r'$\sigma^2$')

plt.figure()
diff = mm.percent_diff(sigma2_B_ccl_SPV3, sigma2_dav_SPV3_diag_interp)
plt.plot(z_grid_tkka, diff, label='perc diff')
plt.yscale('log')
plt.xlabel('z')
plt.ylabel('ratio ccl/dav')


# test all the different sigma2 i have: the goal is to delete useless filsigma2_zsteps2899_SPV3_pyssces, ISTF and SPV3 are practically identical...add()
z_grid_sigma2_ksteps10000 = np.load(f'{ROOT}/exact_SSC/output/sigma2/z_grid_sigma2_ksteps10000.npy.npy')
sigma2_ksteps10000 = np.load(f'{ROOT}/exact_SSC/output/sigma2/sigma2_ksteps10000.npy')
sigma2_zsteps2899_SPV3_pyssc = np.load(f'{ROOT}/exact_SSC/output/sigma2/sigma2_zsteps2899_SPV3_pyssc.npy')
z_grid_sigma2_zsteps2900_SPV3_pyssc = np.load(f'{ROOT}/exact_SSC/output/sigma2/z_grid_sigma2_zsteps2900_SPV3_pyssc.npy')
sigma2_zsteps2900_SPV3_pyssc = np.load(f'{ROOT}/exact_SSC/output/sigma2/sigma2_zsteps2900_SPV3_pyssc.npy')
z_grid_sigma2_zsteps3000_SPV3_serial = np.load(
    f'{ROOT}/exact_SSC/output/sigma2/z_grid_sigma2_zsteps3000_SPV3_serial.npy')
sigma2_zsteps3000_SPV3_serial = np.load(f'{ROOT}/exact_SSC/output/sigma2/sigma2_zsteps3000_SPV3_serial.npy')
z_grid_sigma2_zsteps2990_SPV3_night = np.load(f'{ROOT}/exact_SSC/output/sigma2/z_grid_sigma2_zsteps2990_SPV3_night.npy')
sigma2_zsteps2990_SPV3_night = np.load(f'{ROOT}/exact_SSC/output/sigma2/sigma2_zsteps2990_SPV3_night.npy')
z_grid_sigma2_zsteps3000_ISTF = np.load(f'{ROOT}/exact_SSC/output/sigma2/z_grid_sigma2_zsteps3000_ISTF.npy')
sigma2_zsteps3000_ISTF = np.load(f'{ROOT}/exact_SSC/output/sigma2/sigma2_zsteps3000_ISTF.npy')


plt.plot(z_grid_sigma2_ksteps10000, np.diag(sigma2_ksteps10000), label='sigma2_ksteps10000')
plt.plot(z_grid_sigma2_zsteps2900_SPV3_pyssc, np.diag(sigma2_zsteps2900_SPV3_pyssc))
plt.plot(z_grid_sigma2_zsteps3000_SPV3_serial, np.diag(sigma2_zsteps3000_SPV3_serial))
plt.plot(z_grid_sigma2_zsteps2990_SPV3_night, np.diag(sigma2_zsteps2990_SPV3_night))
plt.plot(z_grid_sigma2_zsteps3000_ISTF, np.diag(sigma2_zsteps3000_ISTF))
plt.yscale('log')

plt.plot(z_grid_sigma2_ksteps10000, sigma2_ksteps10000[100, :], label='sigma2_ksteps10000')
plt.plot(z_grid_sigma2_zsteps2900_SPV3_pyssc,
         sigma2_zsteps2900_SPV3_pyssc[100, :], label='sigma2_zsteps2900_SPV3_pyssc')
plt.plot(z_grid_sigma2_zsteps3000_SPV3_serial,
         sigma2_zsteps3000_SPV3_serial[100, :], label='sigma2_zsteps3000_SPV3_serial')
plt.plot(z_grid_sigma2_zsteps2990_SPV3_night,
         sigma2_zsteps2990_SPV3_night[100, :], label='sigma2_zsteps2990_SPV3_night')
plt.plot(z_grid_sigma2_zsteps3000_ISTF, sigma2_zsteps3000_ISTF[100, :], label='sigma2_zsteps3000_ISTF')
# plt.yscale('log')
plt.legend()

common_zgrid = np.linspace(0.1, 3, 250)
sigma2_ksteps10000_interp_func = interp1d(z_grid_sigma2_ksteps10000, sigma2_ksteps10000, kind='linear')
sigma2_ksteps10000_interp_func = interp1d(z_grid_sigma2_ksteps10000, sigma2_ksteps10000, kind='linear')
sigma2_ksteps10000_interp_func = interp1d(z_grid_sigma2_ksteps10000, sigma2_ksteps10000, kind='linear')
sigma2_ksteps10000_interp_func = interp1d(z_grid_sigma2_ksteps10000, sigma2_ksteps10000, kind='linear')
sigma2_ksteps10000_interp_func = interp1d(z_grid_sigma2_ksteps10000, sigma2_ksteps10000, kind='linear')

sigma2_ksteps10000_interp = sigma2_ksteps10000_interp_func(common_zgrid)
