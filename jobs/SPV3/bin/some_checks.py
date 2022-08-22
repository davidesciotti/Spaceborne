import matplotlib.pyplot as plt
import numpy as np
import matplotlib

import my_module
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
compare_covmats = True
new_vs_old_input_files = False
plot_nz = False
EP_or_ED = cfg.general_config['EP_or_ED']
zbins_list = (10,)
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

    if probe in ['WLO', 'GCO']:
        probe_dav = probe[:2]
    elif probe == 'WLA':
        probe_dav = 'WA'
    else:
        raise ValueError('Probe not recognized')

    cl_ll_3d_opt = np.load(
        f'{path}/dv-{probe}-{nbl_WL_opt}-{cfg.general_config["specs"]}-EP10.npy')
    cl_ll_3d_pes = np.load(
        f'{path}/dv-{probe}-{nbl_WL_pes}-{cfg.general_config["specs"]}-EP10.npy')
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
            cov_opt = np.genfromtxt(
                f'{path}/{probe_v}/cm-{probe_v}-{nbl_opt}-{cfg.general_config["specs"]}-{EP_or_ED}{zbins:02}.dat')
            cov_pes = np.genfromtxt(
                f'{path}/{probe_v}/cm-{probe_v}-{nbl_pes}-{cfg.general_config["specs"]}-{EP_or_ED}{zbins:02}.dat')

            cov_pes_elem = cov_pes.shape[0]

            print(f'probe: {probe_v}, zbins: {zbins}; is the pes covmat just a part of the opt one?',
                  np.array_equal(cov_opt[:cov_pes_elem, :cov_pes_elem], cov_pes))

if compare_covmats:
    # probe_vinc = 'WLO'
    zbins = 10
    for probe_vinc in ['WLO', 'GCO']:
        for zbins in zbins_list:
            ut.test_cov(probe_vinc, 32, zbins, plot_cl=False, plot_cov=True, check_dat=False,
                        specs=cfg.general_config['specs'], EP_or_ED=EP_or_ED)
        print(f'probe_vinc {probe_vinc}, zbins {zbins} done')

if new_vs_old_input_files:

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
    niz_flagship = np.genfromtxt(f'{job_path}/input/InputNz/{lens_or_source}/Flagship/niTab-{EP_or_ED}{zbins:02}.dat')
    niz_redbook = np.genfromtxt(f'{job_path}/input/InputNz/{lens_or_source}/RedBook/niTab-{EP_or_ED}{zbins:02}.dat')

    print(f'niz_flagship shape: {niz_flagship.shape}')
    print(f'niz_redbook shape: {niz_redbook.shape}')

    for zbin in range(zbins):
        plt.plot(niz_flagship[:, 0], niz_flagship[:, zbin + 1], '--', label='niz_flagship', c=colors[zbin])
        plt.plot(niz_redbook[:, 0], niz_redbook[:, zbin + 1], label='niz_redbook', c=colors[zbin])
    plt.legend()
    plt.grid()


    for zbin in range(zbins):
        ng = 30 # gal/arcmin2
        conversion_factor = 11818102.860035626  # deg to arcmin^2
        n_bar = ng / zbins * conversion_factor

        print('nbar = ngal/degsq for the bin:')
        print(np.trapz(y=niz_redbook[:, zbin + 1], x=niz_redbook[:, 0]))

print('done')



