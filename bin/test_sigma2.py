# ! remove from here
import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import sys

import os
ROOT = os.getenv('ROOT')
SB_ROOT = f'{ROOT}/Spaceborne'

# project modules
sys.path.append(SB_ROOT)
import bin.my_module as mm

# job config
import jobs.SPV3_magcut_zcut_thesis.config.config_SPV3_magcut_zcut_thesis as cfg

covariance_cfg = cfg.covariance_cfg


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

assert False

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
