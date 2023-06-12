import sys
import time
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

project_path = Path.cwd().parent.parent.parent
job_path = Path.cwd().parent

sys.path.append(f'{project_path.parent}/common_lib_and_cfg/common_lib')
import my_module as mm
import cosmo_lib as csmlib

sys.path.append(f'{project_path.parent}/common_lib_and_cfg/common_config')
import ISTF_fid_params as ISTFfid
import mpl_cfg

matplotlib.use('Qt5Agg')
plt.rcParams.update(mpl_cfg.mpl_rcParams_dict)
start_time = time.perf_counter()

probe = '3x2pt'
GO_or_GS = 'GO'
ML = 245
MS = 245
ZL = 2
ZS = 2
which_pk = 'HMCode2020'
EP_or_ED = 'ED'
zbins = 13
idIA = 2
idB = 3
idM = 3
idR = 1
test_cov = False
test_fm = True

probe_dict = {
    'WL': 'WLO',
    'GC': 'GCO',
    '3x2pt': '3x2pt'
}

cov_dav_path = '/Users/davide/Documents/Lavoro/Programmi/SSC_restructured_v2/jobs/SPV3_magcut_zcut/output/' \
               'Flagship_2/covmat/BNT_False/cov_ell_cuts_False'
cov_vinc_path = '/Users/davide/Documents/Lavoro/Programmi/common_data/vincenzo/SPV3_07_2022/LiFEforSPV3/' \
                'OutputFiles/CovMats/GaussOnly'
fm_dav_path = '/Users/davide/Documents/Lavoro/Programmi/SSC_restructured_v2/jobs/SPV3_magcut_zcut/output/' \
              'Flagship_2/FM/BNT_False/ell_cuts_False'
fm_vinc_path = cov_vinc_path.replace('CovMats', 'FishMat') + '/Flat'

for probe in ('WL', 'GC', '3x2pt'):

    if test_cov:
        cov_dav = np.load(f'{cov_dav_path}/covmat_{GO_or_GS}_{probe}_zbins{EP_or_ED}{zbins}'
                          f'_ML{ML}_ZL{ZL:02d}_MS{MS}_ZS{ZS:02d}_idIA{idIA}_idB{idB}_idM{idM}_idR{idR}'
                          f'_kmaxhoverMpc2.239_2D.npz')['arr_0']
        cov_vinc = np.genfromtxt(f'{cov_vinc_path}/{probe_dict[probe]}/{which_pk}/'
                                 f'cm-{probe_dict[probe]}-{EP_or_ED}{zbins}'
                                 f'-ML{ML}-MS{MS}-idIA{idIA}-idB{idB}-idM{idM}-idR{idR}.dat')

        np.testing.assert_allclose(cov_dav, cov_vinc, rtol=1e-5, atol=0)

        print(f'cov {probe}, test passed')

    elif test_fm:
        fm_dav = np.genfromtxt(
            f'{fm_dav_path}/FM_{GO_or_GS}_{probe}_zbins{EP_or_ED}{zbins}_ML{ML}_ZL{ZL:02d}_MS{MS}_ZS{ZS:02d}_'
            f'idIA{idIA}_idB{idB}_idM{idM}_idR{idR}_kmaxhoverMpc2.239.txt')
        fm_vinc = np.genfromtxt(f'{fm_vinc_path}/{probe_dict[probe]}/{which_pk}/'
                                f'fm-{probe_dict[probe]}-{EP_or_ED}{zbins}'
                                f'-ML{ML}-MS{MS}-idIA{idIA}-idB{idB}-idM{idM}-idR{idR}.dat')

        np.testing.assert_allclose(fm_dav, fm_vinc, rtol=1e-3, atol=0)

        # mm.compare_arrays(fm_dav, fm_vinc, 'dav', 'vinc', plot_array=True, log_array=True,
        #                   plot_diff=True, log_diff=True)

        print(f'FM {probe}, test passed ✅')
