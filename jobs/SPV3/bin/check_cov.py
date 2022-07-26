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
import cosmo_lib as csmlib

sys.path.append(f'{project_path}/bin')
import ell_values_running as ell_utils

sys.path.append(f'{project_path}/common_config')
import ISTF_fid_params
import mpl_cfg

matplotlib.use('Qt5Agg')
plt.rcParams.update(mpl_cfg.mpl_rcParams_dict)
start_time = time.perf_counter()

# ! options
probe_vinc = '3x2pt'
nbl_WL = 32
zbins = 10
plot_cl = False
plot_cov = True
# ! end options


# for probe_vinc in ['WLO', 'GCO', '3x2pt']:

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

start = time.perf_counter()
cov_vin = np.genfromtxt(
    f'{job_path}/input/covmat/cm-{probe_vinc}-{nbl_WL}-wzwaCDM-Flat-GR-TB-idMag0-idRSD0-idFS0-idSysWL3-idSysGC4-EP{zbins}.dat')
cov_dav = np.load(
    f'{job_path}/output/covmat/zbins{zbins}/covmat_GO_{probe_dav}_lmax{probe_dav_2}{ell_max}_nbl{nbl}_zbins{zbins}_2D.npy')
# cov_dav_old = np.load(
#     f'{job_path.parent}/SSC_comparison/output/covmat/covmat_GO_{probe_dav}_lmax{probe_dav}{ell_max}_nbl30_2D.npy')
print(f'cov loaded in {time.perf_counter() - start:.2f} s')


# CHECKS
if cov_vin.shape == cov_dav.shape:
    print(f'{probe_vinc}: the shapes of cov_vin and cov_dav are equal ✅')
else:
    print(f'{probe_vinc}: the shapes of cov_vin and cov_dav are different ❌')

diff = mm.percent_diff_nan(cov_vin, cov_dav, eraseNaN=True)

rtol = 1e0  # in "percent" units
if np.all(np.abs(diff) < rtol):
    print(f'{probe_vinc}: cov_vin and cov_dav are different by less than {rtol}% ✅')
else:
    print(f'{probe_vinc}: cov_vin and cov_dav are different by less than {rtol}% ❌')
    print(f'max discrepancy: {np.max(np.abs(diff)):.2f}%')

zpairs_auto, zpairs_cross, zpairs_3x2pt = mm.get_pairs(zbins)

# PLOTS
if plot_cov:

    if probe_dav != '3x2pt':
        zpairs = zpairs_auto
    else:
        zpairs = zpairs_3x2pt

    mm.matshow(cov_vin[:zpairs, :zpairs], log=True, title='cov_vin')
    mm.matshow(cov_dav[:zpairs, :zpairs], log=True, title='cov_dav')
    mm.matshow(diff[:zpairs, :zpairs], log=True, title='percent_diff')

if plot_cl:

    cl_dav_new = np.load(f'{job_path}/output/cl_3d/cl_ll_3d_zbins{zbins}_ellmax5000.npy')
    cl_dav_old = np.load(f'{job_path.parent}/SSC_comparison/output/cl_3d/C_LL_WLonly_3D.npy')
    ell_LL_new, _ = ell_utils.compute_ells(nbl=32, ell_min=10, ell_max=5000, recipe='ISTF')
    ell_LL_old, _ = ell_utils.compute_ells(nbl=30, ell_min=10, ell_max=5000, recipe='ISTF')

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
              '#17becf']

    plt.figure()
    for i in range(10):
        j = i
        plt.plot(ell_LL_new, cl_dav_new[:, i, j], c=colors[i])
        plt.plot(ell_LL_old, cl_dav_old[:, i, j], '--', c=colors[i])
    plt.show()
    plt.grid()

print('done')
