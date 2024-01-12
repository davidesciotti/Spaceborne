import sys
import time
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

ROOT = '/home/davide/Documenti/Lavoro/Programmi'
SB_ROOT = f'{ROOT}/Spaceborne'

sys.path.append(SB_ROOT)
import bin.plots_FM_running as plot_utils
import bin.my_module as mm
import common_cfg.mpl_cfg as mpl_cfg

matplotlib.use('Qt5Agg')
plt.rcParams.update(mpl_cfg.mpl_rcParams_dict)
start_time = time.perf_counter()

probe = '3x2pt'
GO_or_GS = 'G'
ssc_code = 'exactSSC'
ML = 245
MS = 245
ZL = 2
ZS = 2
which_pk = 'HMCodeBar'
EP_or_ED = 'EP'
zbins = 13
idIA = 2
idB = 3
idM = 3
idR = 1
test_cov = False
test_fm = True
flat_or_nonflat = 'Flat'
# flat_or_nonflat = ''

probe_dict = {
    'WL': 'WLO',
    'GC': 'GCO',
    '3x2pt': '3x2pt'
}

path_dav = '/home/davide/Documenti/Lavoro/Programmi/Spaceborne/jobs/SPV3_magcut_zcut_thesis/output/Flagship_2'
cov_dav_path = f'{path_dav}/covmat/BNT_False/ell_cuts_False'
cov_vinc_path = '/home/davide/Documenti/Lavoro/Programmi/common_data/vincenzo/SPV3_07_2022/LiFEforSPV3/' \
                f'OutputFiles/CovMats/GaussOnly/Full'

fm_dav_path = cov_dav_path.replace('covmat', 'FM')
fm_vinc_path = cov_vinc_path.replace('CovMats', 'FishMat').replace('/Full', '') + f'/{flat_or_nonflat}'

# for probe in ('WL', 'GC', '3x2pt'):
for probe in ('3x2pt',):

    suffix = f'zbins{EP_or_ED}{zbins}' \
        f'_ML{ML}_ZL{ZL:02d}_MS{MS}_ZS{ZS:02d}_idIA{idIA}_idB{idB}_idM{idM}_idR{idR}_pk{which_pk}'

    cov_dav_filename = f'covmat_{GO_or_GS}_{probe}_{suffix}_2D.npz'
    fm_dav_txt_filename = cov_dav_filename.replace('covmat', 'FM').replace('npz', 'txt').replace('_2D', '')
    fm_pickle_filename = f'FM_GSSC_{ssc_code}_{suffix}.pickle'

    if test_cov:
        cov_dav = np.load(f'{cov_dav_path}/{cov_dav_filename}')['arr_0']
        cov_vinc = np.genfromtxt(f'{cov_vinc_path}/{probe_dict[probe]}/{which_pk}/'
                                 f'cm-{probe_dict[probe]}-{EP_or_ED}{zbins}'
                                 f'-ML{ML}-MS{MS}-idIA{idIA}-idB{idB}-idM{idM}-idR{idR}.dat')

        np.testing.assert_allclose(cov_dav, cov_vinc, rtol=1e-5, atol=0)

        print(f'cov {probe}, test passed ✅')

    if test_fm:
        fm_dav = mm.load_pickle(f'{fm_dav_path}/{fm_pickle_filename}')[f'FM_{probe}_G']
        fm_vinc = np.genfromtxt(f'{fm_vinc_path}/{probe_dict[probe]}/{which_pk}/'
                                f'fm-{probe_dict[probe]}-{EP_or_ED}{zbins}'
                                f'-ML{ML}-MS{MS}-idIA{idIA}-idB{idB}-idM{idM}-idR{idR}.dat')

        mm.compare_arrays(fm_dav, fm_vinc, 'dav', 'vinc', plot_array=True, log_array=True,
                          plot_diff=False, log_diff=False)

        np.testing.assert_allclose(fm_dav, fm_vinc, rtol=1e-3, atol=0)

        print(f'FM {probe}, test passed ✅')
