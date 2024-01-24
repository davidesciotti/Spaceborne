# ! remove from here
import scipy
import numpy as np
import matplotlib.pyplot as plt

import sys

ROOT = '/home/davide/Documenti/Lavoro/Programmi'
SB_ROOT = f'{ROOT}/Spaceborne'

# project modules
sys.path.append(SB_ROOT)
import bin.my_module as mm

# job config
import jobs.SPV3_magcut_zcut_thesis.config.config_SPV3_magcut_zcut_thesis as cfg

covariance_cfg = cfg.covariance_cfg


z_grid_tkka = np.load(f'{covariance_cfg["PyCCL_cfg"]["cov_path"]}/z_grid_tkka.npy')
sigma2_B_ccl = np.load(f'{covariance_cfg["PyCCL_cfg"]["cov_path"]}/sigma2_B_ccl.npy')
sigma2_B_ccl_polar_cap = np.load(f'{covariance_cfg["PyCCL_cfg"]["cov_path"]}/sigma2_B_ccl_polar_cap.npy')


z_grid_dav_ISTF = np.load(
    '/home/davide/Documenti/Lavoro/Programmi/exact_SSC/output/sigma2/z_grid_sigma2_zsteps3000_ISTF.npy')
sigma2_dav_ISTF = np.load('/home/davide/Documenti/Lavoro/Programmi/exact_SSC/output/sigma2/sigma2_zsteps3000_ISTF.npy')

# TODO the path in the cfg dile seems to be wrong, it points to the folder above, but no interpolation
z_grid_dav_SPV3 = np.load(
    '/home/davide/Documenti/Lavoro/Programmi/exact_SSC/output/SPV3/separate_universe/jan_2024/d2ClAB_dVddeltab/z_grid_ssc_integrand_zsteps2899.npy')
sigma2_dav_SPV3 = np.load(
    '/home/davide/Documenti/Lavoro/Programmi/exact_SSC/output/SPV3/separate_universe/jan_2024/d2ClAB_dVddeltab/sigma2_zsteps2899_SPV3.npy')

sigma2_dav_SPV3_diag = np.diag(sigma2_dav_SPV3)
sigma2_dav_ISTF_diag = np.diag(sigma2_dav_ISTF)

sigma2_dav_SPV3_diag_interp_func = scipy.interpolate.interp1d(
    z_grid_dav_SPV3, sigma2_dav_SPV3_diag, kind='linear', fill_value='extrapolate')
sigma2_dav_SPV3_diag_interp = sigma2_dav_SPV3_diag_interp_func(z_grid_tkka)

plt.figure()
plt.plot(z_grid_tkka, sigma2_B_ccl, label='ccl')
plt.plot(z_grid_tkka, sigma2_B_ccl_polar_cap, label='ccl polar cap')
plt.plot(z_grid_dav_SPV3, np.diag(sigma2_dav_SPV3), label='dav SPV3')
plt.plot(z_grid_dav_ISTF, np.diag(sigma2_dav_ISTF), label='dav ISTF', ls='--')
plt.yscale('log')
plt.legend()

plt.figure()
diff = mm.percent_diff(sigma2_B_ccl, sigma2_dav_SPV3_diag_interp)
plt.plot(z_grid_tkka, diff, label='perc diff')
plt.yscale('log')

# save 


# ! remove until here
