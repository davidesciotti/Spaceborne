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

probe_vinc = 'WLO'
case = 'opt'
nbl_WL = 32
zbins = 10

if probe_vinc == 'WLO':
    probe_dav = 'WL'
elif probe_vinc == 'GCO':
    probe_dav = 'GC'

cov_vin = np.genfromtxt(
    f'{job_path}/input/covmat/cm-{probe_vinc}-{nbl_WL}-wzwaCDM-Flat-GR-TB-idMag0-idRSD0-idFS0-idSysWL3-idSysGC4-EP{zbins}.dat')
cov_dav = np.load(
    f'{job_path}/output/covmat/zbins{zbins}/covmat_GO_{probe_dav}_lmaxWL5000_nbl{nbl_WL}_zbins{zbins}_2D.npy')
cov_dav_old = np.load(
    f'{job_path.parent}/SSC_comparison/output/covmat/covmat_GO_{probe_dav}_lmaxWL5000_nbl30_2D.npy')

print(f'cov_vin.shape = {cov_vin.shape}')
print(f'cov_dav.shape = {cov_dav.shape}')

mm.matshow(cov_vin[:55, :55], log=True, title='cov_vin')
mm.matshow(cov_dav[:55, :55], log=True, title='cov_dav')
mm.matshow(cov_vin[:55, :55]/cov_dav[:55, :55], log=False, title='ratio')

cl_dav_new = np.load(f'{job_path}/output/cl_3d/cl_ll_3d_zbins10_ellmax5000.npy')
cl_dav_old = np.load(f'{job_path.parent}/SSC_comparison/output/cl_3d/C_LL_WLonly_3D.npy')
ell_LL_new, _ = ell_utils.compute_ells(nbl=32, ell_min=10, ell_max=5000, recipe='ISTF')
ell_LL_old, _ = ell_utils.compute_ells(nbl=30, ell_min=10, ell_max=5000, recipe='ISTF')

print(f'cl_dav_new.shape = {cl_dav_new.shape}')
print(f'cl_dav_old.shape = {cl_dav_old.shape}')

plt.figure()
for i in range(10):
    j = i
    plt.plot(ell_LL_new, cl_dav_new[:, i, j], label='cl_dav_new')

plt.plot(ell_LL_old, cl_dav_old[:, 0, 0], label='cl_dav_old')
plt.legend()
plt.show()
plt.grid()