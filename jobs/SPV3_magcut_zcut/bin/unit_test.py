import sys
import time
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

project_path = Path.cwd().parent.parent.parent
job_path = Path.cwd().parent

sys.path.append(f'{project_path}/lib')
import my_module as mm

sys.path.append(f'{project_path}/bin')
import ell_values_running as ell_utils

sys.path.append(f'{project_path}/config')
import ISTF_fid_params
import mpl_cfg

sys.path.append(f'{job_path}/config')
import config_SPV3 as cfg

matplotlib.use('Qt5Agg')
plt.rcParams.update(mpl_cfg.mpl_rcParams_dict)
start_time = time.perf_counter()


def test_cov(probe_vinc, nbl_WL, zbins, plot_cl, plot_cov, plot_hist, check_dat, specs, EP_or_ED, rtol=1.):
    """this test compares the GO covmat for SPV3 against vincenzo's files on Google Drive
    check_dat specifies whether one wants to check the covmats saved in dat format (quite unnecessary..., but still)"""

    if probe_vinc == 'WLO':
        probe_dav = 'WL'
        nbl = nbl_WL
        ell_max = 5000
        probe_dav_2 = probe_dav

    elif probe_vinc == 'GCO':
        probe_dav = 'GC'
        nbl = nbl_WL - 3
        ell_max = 3000
        probe_dav_2 = probe_dav

    elif probe_vinc == '3x2pt':
        probe_dav = probe_vinc
        nbl = nbl_WL - 3
        ell_max = 3000
        probe_dav_2 = 'XC'  # just some naming convention

    else:
        raise ValueError(f'Unknown probe_vinc: {probe_vinc}')

    print(f'\n********** probe: {probe_vinc}, zbins: {zbins} **********')

    start = time.perf_counter()
    cov_vin = np.genfromtxt(
        f'{job_path}/input/CovMats/{probe_vinc}/cm-{probe_vinc}-{nbl_WL}-{specs}-{EP_or_ED}{zbins:02}.dat')

    if check_dat:
        cov_dav = np.genfromtxt(
            f'{job_path}/output/covmat/vincenzos_format/GaussOnly/{probe_vinc}/cm-{probe_vinc}-{nbl_WL}-{specs}-{EP_or_ED}{zbins:02}.dat')
    else:
        cov_dav = np.load(
            f'{job_path}/output/covmat/zbins{zbins:02}/covmat_GO_{probe_dav}_lmax{probe_dav_2}{ell_max}_nbl{nbl}_zbins{zbins:02}_2D.npy')
    # cov_dav_old = np.load(
    #     f'{job_path.parent}/SSC_comparison/output/covmat/covmat_GO_{probe_dav}_lmax{probe_dav}{ell_max}_nbl30_2D.npy')

    print(f'cov loaded in {time.perf_counter() - start:.2f} s')

    # CHECKS
    if cov_vin.shape == cov_dav.shape:
        result_emoji = '✅'
    else:
        result_emoji = '❌'
    print(f'are the shapes of cov_vin and cov_dav equal? {result_emoji}')

    diff = mm.percent_diff_nan(cov_vin, cov_dav, eraseNaN=True)

    if np.all(np.abs(diff) < rtol):
        result_emoji = '✅'
        additional_info = ''
    else:
        result_emoji = '❌'
        no_outliers = np.where(np.abs(diff) > rtol)[0].shape[0]
        outliers_fract = no_outliers / diff.shape[0] ** 2 * 100
        additional_info = f'\nmax discrepancy: {np.max(np.abs(diff)):.2f}%;' \
                          f'\nfraction of elements with discrepancy > {rtol} %: {outliers_fract:.2f} %' \
                          f'\nnumber of elements with discrepancy > {rtol} %: {no_outliers}'

    print(f'are cov_vin and cov_dav different by less than {rtol}% ? {result_emoji} {additional_info}')

    zpairs_auto, zpairs_cross, zpairs_3x2pt = mm.get_zpairs(zbins)

    if plot_hist:
        plt.hist(diff[np.where(np.abs(diff) > rtol)], bins = 50)

    # PLOTS
    if plot_cov:

        if probe_dav != '3x2pt':
            zpairs = zpairs_auto
        else:
            zpairs = zpairs_3x2pt

        mm.matshow(cov_vin[:zpairs, :zpairs], log=True, abs_val=True, title=f'{probe_dav}, cov_vin')
        mm.matshow(cov_dav[:zpairs, :zpairs], log=True, abs_val=True, title=f'{probe_dav}, cov_dav')
        mm.matshow(diff[:zpairs, :zpairs], log=True, abs_val=True, title=f'{probe_dav}, percent_diff')

    if plot_cl:

        cl_dav_new = np.load(f'{job_path}/output/cl_3d/cl_ll_3d_zbins{zbins}_ellmax5000.npy')
        cl_dav_old = np.load(f'{job_path.parent}/SSC_comparison/output/cl_3d/C_LL_WLonly_3D.npy')
        ell_LL_new, _ = ell_utils.compute_ells(nbl=32, ell_min=10, ell_max=5000, recipe='ISTF')
        ell_LL_old, _ = ell_utils.compute_ells(nbl=30, ell_min=10, ell_max=5000, recipe='ISTF')

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                  '#17becf']

        plt.figure()
        for i in range(zbins):
            j = i
            plt.plot(ell_LL_new, cl_dav_new[:, i, j], c=colors[i])
            plt.plot(ell_LL_old, cl_dav_old[:, i, j], '--', c=colors[i])
        plt.grid()
        plt.show()
