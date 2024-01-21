import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import unit_test as ut
import sys
from pathlib import Path

# get project directory
project_path = Path.cwd().parent.parent.parent.parent
job_path = Path.cwd().parent
# import configuration and functions modules
sys.path.append(f'{project_path}/config')
import config_SPV3 as cfg

sys.path.append(f'{project_path.parent}/common_data/common_lib')
import my_module as mm

matplotlib.use('Qt5Agg')

# ! options
plot_cl = False
check_cov_pesopt = False
compare_covmats = False
new_vs_old_input_files = False
plot_nz = False
check_EP_vs_ED = False
EP_or_ED = 'ED'
zbins_list = (3, 5, 7, 9, 10, 11, 13, 15, 17)
probe_list = ('WLO', 'GCO', '3x2pt')
# ! end options

nbl_opt = 32
nbl_pes = 26

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

if plot_cl:
    # check that the pes cls are just a subset of the opt ones
    probe = 'WLO'
    path = f'{job_path}/DataVecTabs/3D_reshaped/{probe}'
    nbl_WL_opt = 32
    nbl_WL_pes = 26
    ellmaxWL_opt = 5000
    ellmaxWL_pes = 1500
    zbins = 10

    if probe in ['WLO', 'GCO']:
        probe_dav = probe[:2]
    elif probe == 'WLA':
        probe_dav = 'WA'
    else:
        raise ValueError('Probe not recognized')

    cl_ll_3d_opt = np.load(
        f'{path}/dv-{probe}-{nbl_WL_opt}-{cfg.general_config["specs"]}-{EP_or_ED}{zbins}.npy')
    cl_ll_3d_pes = np.load(
        f'{path}/dv-{probe}-{nbl_WL_pes}-{cfg.general_config["specs"]}-{EP_or_ED}{zbins}.npy')
    ell_opt = np.loadtxt(f'{path}/ell_{probe_dav}_ellmaxWL{ellmaxWL_opt}.txt')
    ell_pes = np.loadtxt(f'{path}/ell_{probe_dav}_ellmaxWL{ellmaxWL_pes}.txt')

    plt.figure()
    plt.plot(ell_opt, cl_ll_3d_opt[:, 0, 0], '.-', label='opt')
    plt.plot(ell_pes, cl_ll_3d_pes[:, 0, 0], '--', marker='o', alpha=0.5, label='pes')
    plt.legend()
    plt.show()
    plt.yscale('log')

probe_v_lst = ['WLO', 'WLA', 'GCO', '3x2pt']
path = '/home/davide/Documenti/Lavoro/Programmi/Spaceborne/jobs/SPV3/output/covmat/vincenzos_format/GaussOnly'
if check_cov_pesopt:
    # this checks that the pes covmat can be obtained from the opt one, simply by cutting it
    for probe_v in probe_v_lst:
        for zbins in zbins_list:
            cov_opt = np.genfromtxt(
                f'{path}/{probe_v}/cm-{probe_v}-{nbl_opt}-{cfg.general_config["specs"]}-{EP_or_ED}{zbins:02}.dat')
            cov_pes = np.genfromtxt(
                f'{path}/{probe_v}/cm-{probe_v}-{nbl_pes}-{cfg.general_config["specs"]}-{EP_or_ED}{zbins:02}.dat')

            cov_pes_elem = cov_pes.shape[0]

            print(f'probe: {probe_v}, zbins: {zbins}; is the pes covmat just a part of the opt one?',
                  np.array_equal(cov_opt[:cov_pes_elem, :cov_pes_elem], cov_pes))

if compare_covmats:
    # probe_list = ('WLO',)
    # zbins_list = (10,)
    for probe_vinc in probe_list:
        for zbins in zbins_list:
            ut.test_cov(probe_vinc, 32, zbins, plot_hist=False, plot_cl=False, plot_cov=False, check_dat=True,
                        specs=cfg.general_config['specs'], EP_or_ED=EP_or_ED, rtol=5.)
        print(f'probe_vinc {probe_vinc}, zbins {zbins} done')

if new_vs_old_input_files:
    """ Check the new input files"""

    path_new = job_path
    path_old = f'{path_new}/old_probably_discard'

    folder_id_dict = {
        'DataVectors': 'dv',
        'CovMats': 'cm',
        'ResFunTabs': 'rf',
    }

    for probe_vinc in ('GCO', 'WLO', '3x2pt'):
        # for zbins in zbins_list:
        for folder, id in folder_id_dict.items():
            print(f'probe_vinc: {probe_vinc}, zbins: {zbins}, folder: {folder}, id: {id}')
            vinc_new = np.genfromtxt(
                f'{path_new}/{folder}/{probe_vinc}/{id}-{probe_vinc}-{nbl_opt}-{cfg.general_config["specs"]}-{EP_or_ED}{zbins:02}.dat')
            vinc_old = np.genfromtxt(
                f'{path_old}/{folder}/{probe_vinc}/{id}-{probe_vinc}-{nbl_opt}-{cfg.general_config["specs"]}-{EP_or_ED}{zbins:02}.dat')
            mm.compare_2D_arrays(vinc_new, vinc_old, plot=False, log_arr=True, log_diff=False, abs_val=True, rtol=1.)

if plot_nz:
    lens_or_source = 'Lenses'
    flagship_or_redbook = 'RedBook'
    for zbins in zbins_list:
        niz_flagship = np.genfromtxt(
            f'{job_path}/input/InputNz/Lenses/{flagship_or_redbook}/niTab-{EP_or_ED}{zbins:02}.dat')
        niz_redbook = np.genfromtxt(
            f'{job_path}/input/InputNz/Sources/{flagship_or_redbook}/niTab-{EP_or_ED}{zbins:02}.dat')

        # print(f'niz_flagship shape: {niz_flagship.shape}')
        # print(f'niz_redbook shape: {niz_redbook.shape}')

        print(f'zbins {zbins}; is niz_A == niz_B?', np.array_equal(niz_flagship, niz_redbook))

    for zbin in range(zbins):
        plt.plot(niz_flagship[:, 0], niz_flagship[:, zbin + 1], '--', label='niz_flagship', c=colors[zbin])
        plt.plot(niz_redbook[:, 0], niz_redbook[:, zbin + 1], label='niz_redbook', c=colors[zbin])
    plt.legend()
    plt.grid()

if check_EP_vs_ED:
    probe_list = ('WLO',)
    zbins_list = (10,)
    for probe in probe_list:
        for zbins in zbins_list:
            cov_EP = np.genfromtxt(
                f'{path}/{probe}/cm-{probe}-{nbl_opt}-{cfg.general_config["specs"]}-EP{zbins:02}.dat')
            cov_ED = np.genfromtxt(
                f'{path}/{probe}/cm-{probe}-{nbl_opt}-{cfg.general_config["specs"]}-ED{zbins:02}.dat')
            mm.compare_2D_arrays(cov_EP, cov_ED, 'EP', 'ED')


print('done')
