import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import unit_test as ut
import sys
from pathlib import Path
# get project directory
project_path = Path.cwd().parent.parent.parent.parent
# import configuration and functions modules
sys.path.append(str(project_path / 'config'))
import config_SPV3 as cfg

matplotlib.use('Qt5Agg')

# ! options
plot_cl = False
check_cov_pesopt = False
compare_covmats = True
zbins = 10
EP_or_ED = 'EP'
zbins_list = (3, 5, 7, 9, 10)
# ! end options


if plot_cl:
    probe = 'WLO'
    path = f'/Users/davide/Documents/Lavoro/Programmi/common_data/vincenzo/SPV3_07_2022/DataVecTabs/3D_reshaped/{probe}'
    nbl_WL_opt = 32
    nbl_WL_pes = 26
    ellmaxWL_opt = 5000
    ellmaxWL_pes = 1500

    if probe in ['WLO', 'GCO']:
        probe_dav = probe[:2]
    elif probe == 'WLA':
        probe_dav = 'WA'
    else:
        raise ValueError('Probe not recognized')

    cl_ll_3d_opt = np.load(
        f'{path}/dv-{probe}-{nbl_WL_opt}-wzwaCDM-Flat-GR-TB-idMag0-idRSD0-idFS0-idSysWL3-idSysGC4-EP10.npy')
    cl_ll_3d_pes = np.load(
        f'{path}/dv-{probe}-{nbl_WL_pes}-wzwaCDM-Flat-GR-TB-idMag0-idRSD0-idFS0-idSysWL3-idSysGC4-EP10.npy')
    ell_opt = np.loadtxt(f'{path}/ell_{probe_dav}_ellmaxWL{ellmaxWL_opt}.txt')
    ell_pes = np.loadtxt(f'{path}/ell_{probe_dav}_ellmaxWL{ellmaxWL_pes}.txt')

    plt.figure()
    plt.plot(ell_opt, cl_ll_3d_opt[:, 0, 0], '.-', label='opt')
    plt.plot(ell_pes, cl_ll_3d_pes[:, 0, 0], '--', marker='o', alpha=0.5, label='pes')
    plt.legend()
    plt.show()
    plt.yscale('log')

probe_v_lst = ['WLO', 'WLA', 'GCO', '3x2pt']
path = '/Users/davide/Documents/Lavoro/Programmi/SSC_restructured_v2/jobs/SPV3/output/covmat/vincenzos_format/GaussOnly'
if check_cov_pesopt:
    # this checks that the pes covmat can be obtained from the opt one, simply by cutting it
    for probe_v in probe_v_lst:
        for zbins in zbins_list:
            nbl_opt = 32
            nbl_pes = 26
            cov_opt = np.genfromtxt(
                f'{path}/{probe_v}/cm-{probe_v}-{nbl_opt}-wzwaCDM-Flat-GR-TB-idMag0-idRSD0-idFS0-idSysWL3-idSysGC4-{EP_or_ED}{zbins:02}.dat')
            cov_pes = np.genfromtxt(
                f'{path}/{probe_v}/cm-{probe_v}-{nbl_pes}-wzwaCDM-Flat-GR-TB-idMag0-idRSD0-idFS0-idSysWL3-idSysGC4-{EP_or_ED}{zbins:02}.dat')

            cov_pes_elem = cov_pes.shape[0]

            print(probe_v, zbins, 'is the pes covmat just a part of the opt one?',
                  np.array_equal(cov_opt[:cov_pes_elem, :cov_pes_elem], cov_pes))

if compare_covmats:
    probe_vinc = 'WLO'
    zbins = 10
    # for probe_vinc in ['WLO', 'GCO', '3x2pt']:
        # for zbins in zbins_list:
    ut.test_cov(probe_vinc, 32, zbins, plot_cl=False, plot_cov=True, check_dat=False,
                specs=cfg.general_config['specs'], EP_or_ED=EP_or_ED)
    print(f'probe_vinc {probe_vinc}, zbins {zbins} done')
