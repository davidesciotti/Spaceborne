import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('Qt5Agg')

project_path = Path.cwd().parent.parent.parent
job_path = Path.cwd().parent
home_path = Path.home()

sys.path.append(str(project_path))
sys.path.append(str(job_path))

# job-specific modules and configurations
# sys.path.append(str(job_path / 'configs'))
# sys.path.append(str(job_path / 'bin/utils'))


# job configuration
import configs.config_IST_forecast as config
# import bin.utils.utils_IST_forecast

import lib.my_module as mm
import bin.ell_values_running as ell_utils
import bin.Cl_preprocessing_running as Cl_utils
import bin.covariance_running as covmat_utils
import bin.FM_running as FM_utils
import bin.plots_FM_running as plot_utils
import bin.utils_running as utils

start_time = time.perf_counter()

# assert 1 > 2

###############################################################################
#################### PARAMETERS AND SETTINGS DEFINITION #######################
###############################################################################

# import the configuration dictionaries from common_config.py
general_config = config.general_config
covariance_config = config.covariance_config
FM_config = config.FM_config
plot_config = config.plot_config

# plot settings:
params = plot_config['params']
markersize = plot_config['markersize']
dpi = plot_config['dpi']
pic_format = plot_config['pic_format']
plt.rcParams.update(params)

# consistency checks:
utils.consistency_checks(general_config, covariance_config)

# for the time being, I/O is manual and from the main
# load inputs (job-specific)
ind_ordering = covariance_config['ind_ordering']
ind = np.genfromtxt(project_path / f"config/common_data/ind/indici_{ind_ordering}_like.dat").astype(int) - 1
covariance_config['ind'] = ind

Sijkl_dav = np.load(project_path / "config/common_data/Sijkl/Sijkl_WFdavide_nz10000_IA_3may.npy")  # davide, eNLA
# Sijkl_marco = np.load(project_path / "config/common_data/Sijkl/Sijkl_WFmarco_nz10000_zNLA_gen22.npy")  # marco, zNLA
# Sijkl_sylv = np.load(project_path / "config/common_data/Sijkl/sylvain_cosmologyFixe_IAFixe_Sijkl_GCph_WL_XC_3may.npy")  # sylvain, eNLA

Sijkl = Sijkl_dav

assert np.all(Sijkl == Sijkl_dav), 'Sijkl should be Sijkl_dav'

###############################################################################
######################### FORECAST COMPUTATION ################################
###############################################################################

# some variables used for I/O naming and to compute Sylvain's deltas
ell_max_WL = general_config['ell_max_WL']
ell_max_GC = general_config['ell_max_GC']
ell_max_XC = ell_max_GC
ell_max_WL = general_config['ell_max_WL']
nbl = general_config['nbl']

# MAIN BODY OF THE SCRIPT
# compute ell and delta ell values
ell_dict, delta_dict = ell_utils.generate_ell_and_deltas(general_config)
# import and interpolate the cls
cl_dict_2D = utils.import_and_interpolate_cls(general_config, covariance_config, ell_dict)
# reshape them to 3D
cl_dict_3D = Cl_utils.reshape_cls_2D_to_3D(general_config, ell_dict, cl_dict_2D)
# compute covariance matrix
cov_dict = covmat_utils.compute_cov(general_config, covariance_config, ell_dict, delta_dict, cl_dict_3D, Sijkl)
# compute Fisher Matrix
FM_dict = FM_utils.compute_FM(general_config, covariance_config, FM_config, ell_dict, cov_dict)
# plot forecasts
# plt.figure()
# for plot_config['GO_or_GS'] in ['GO', 'GS']:
#     for plot_config['probe'] in ['WL', 'GC', '3x2pt']:
plot_utils.plot_FM(general_config, covariance_config, plot_config, FM_dict)

# save:
if covariance_config['save_covariance']:
    np.save(job_path / f'output/covmat/covmat_GO_WL_lmaxWL{ell_max_WL}_nbl{nbl}_2D.npy', cov_dict['cov_WL_GO_2D'])
    np.save(job_path / f'output/covmat/covmat_GO_GC_lmaxGC{ell_max_GC}_nbl{nbl}_2D.npy', cov_dict['cov_GC_GO_2D'])
    np.save(job_path / f'output/covmat/covmat_GO_3x2pt_lmaxXC{ell_max_XC}_nbl{nbl}_2D.npy', cov_dict['cov_3x2pt_GO_2D'])
    np.save(job_path / f'output/covmat/covmat_GO_WA_lmaxWL{ell_max_WL}_nbl{nbl}_2D.npy', cov_dict['cov_WA_GO_2D'])

    np.save(job_path / f'output/covmat/covmat_GS_WL_lmaxWL{ell_max_WL}_nbl{nbl}_2D.npy', cov_dict['cov_WL_GS_2D'])
    np.save(job_path / f'output/covmat/covmat_GS_GC_lmaxGC{ell_max_GC}_nbl{nbl}_2D.npy', cov_dict['cov_GC_GS_2D'])
    np.save(job_path / f'output/covmat/covmat_GS_3x2pt_lmaxXC{ell_max_XC}_nbl{nbl}_2D.npy', cov_dict['cov_3x2pt_GS_2D'])
    np.save(job_path / f'output/covmat/covmat_GS_WA_lmaxWL{ell_max_WL}_nbl{nbl}_2D.npy', cov_dict['cov_WA_GS_2D'])

if FM_config['save_FM']:
    np.savetxt(job_path / f"/output/FM/FM_WL_GO_lmaxWL{ell_max_WL}_nbl{nbl}.txt", FM_dict['FM_WL_GO'])  # WLonly
    np.savetxt(job_path / f"/output/FM/FM_GC_GO_lmaxGC{ell_max_GC}_nbl{nbl}.txt", FM_dict['FM_GC_GO'])  # GConly
    np.savetxt(job_path / f"/output/FM/FM_3x2pt_GO_lmaxXC{ell_max_XC}_nbl{nbl}.txt", FM_dict['FM_3x2pt_GO'])  # ALL

    np.savetxt(job_path / f"/output/FM/FM_WL_GS_lmaxWL{ell_max_WL}_nbl{nbl}.txt", FM_dict['FM_WL_GS'])
    np.savetxt(job_path / f"/output/FM/FM_GC_GS_lmaxGC{ell_max_GC}_nbl{nbl}.txt", FM_dict['FM_GC_GS'])
    np.savetxt(job_path / f"/output/FM/FM_3x2pt_GS_lmaxXC{ell_max_XC}_nbl{nbl}.txt", FM_dict['FM_3x2pt_GS'])

########### TESTS #############################################
# check FMS
if general_config['which_forecast'] == 'sylvain':
    path_import = '/Users/davide/Documents/Lavoro/Programmi/SSCcomp_prove/output/FM/common_ell_and_deltas/Cij_14may'
elif general_config['which_forecast'] == 'IST':
    path_import = '/Users/davide/Documents/Lavoro/Programmi/SSCcomp_prove/output/FM/ISTspecs_indVincenzo/Cij_14may'
elif general_config['which_forecast'] == 'CLOE':
    path_import = '/Users/davide/Documents/Lavoro/Programmi/SSCcomp_prove/output/FM/ISTspecs_indVincenzo/Cij_14may'

FM_d_old = dict(mm.get_kv_pairs(path_import, filetype="txt"))
tolerance = 0.0001

# plot forecasts to get an idea of the difference: do not just compare the FMs



print('\ncovariance_config=', covariance_config['block_index'], 'ind_ordering=', ind_ordering, '\n')

for probe in ['WL', 'GC', '3x2pt']:
    for GO_or_GS in ['GO', 'GS']:

        if GO_or_GS == 'GS':
            GO_or_GS_old = 'G+SSC'
        else:
            GO_or_GS_old = 'G'

        if probe == '3x2pt':
            probe_lmax = 'XC'
            probe_lmax2 = 'GC'
        else:
            probe_lmax = probe
            probe_lmax2 = probe

        nbl = general_config['nbl']
        ell_max = general_config[f'ell_max_{probe_lmax2}']

        FM_old = FM_d_old[f'FM_{probe}_{GO_or_GS_old}_lmax{probe_lmax}{ell_max}_nbl{nbl}']
        FM_new = FM_dict[f'FM_{probe}_{GO_or_GS}']

        diff = mm.percent_diff(FM_old, FM_new)
        # mm.matshow(diff, title=f'{probe}, {GO_or_GS}')

        # perform 2 separate tests
        for test_result_bool in [np.all(np.abs(diff) < tolerance), np.allclose(FM_old, FM_new)]:

            # transform into nicer and easier to understand string (symbol)
            if test_result_bool:
                test_result_str = '✅'
            else:
                test_result_str = '❌'

            print(f'is the percent difference between the FM < {tolerance}?, {probe}, {GO_or_GS}, {test_result_str}')
