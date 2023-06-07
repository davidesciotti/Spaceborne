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

probe = 'WL'
GO_or_GS = 'GO'
ML = 245
MS = 245
which_pk = 'HMCode2020'
EP_or_ED = 'EP'
zbins = 13

probe_dict = {
    'WL': 'WLO',
    'GC': 'GCO',
    '3x2pt': '3x2pt'
}

for probe in ('WL', 'GC', '3x2pt'):

    dav_path = '/Users/davide/Documents/Lavoro/Programmi/SSC_restructured_v2/jobs/SPV3_magcut_zcut/output/' \
               'Flagship_2/covmat/BNT_False/cov_ell_cuts_False'
    vinc_path = '/Users/davide/Documents/Lavoro/Programmi/common_data/vincenzo/SPV3_07_2022/LiFEforSPV3/OutputFiles/CovMats/GaussOnly'
    cov_dav = np.load(f'{dav_path}/covmat_{GO_or_GS}_{probe}_lmax5000_nbl32_zbins{EP_or_ED}{zbins}'
                      f'_ML{ML}_ZL02_MS{MS}_ZS02_kmax_h_over_Mpc2.238806_2D.npz')['arr_0']
    cov_vinc = np.genfromtxt(
        f'{vinc_path}/{probe_dict[probe]}/{which_pk}/cm-WLO-{EP_or_ED}{zbins}-ML{ML}-MS{MS}-idIA2-idB3-idM3-idR1.dat')

    np.testing.assert_allclose(cov_dav, cov_vinc, rtol=1e-5, atol=0)

    print(f'{probe}, test passed')