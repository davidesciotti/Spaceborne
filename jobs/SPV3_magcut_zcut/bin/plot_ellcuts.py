import sys
import time
from operator import itemgetter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from getdist import MCSamples, plots
from matplotlib import ticker
from matplotlib.cm import get_cmap
from getdist.gaussian_mixtures import GaussianND
import pandas as pd

project_path = Path.cwd().parent.parent.parent

sys.path.append(str(project_path.parent / 'common_data'))
import common_lib.my_module as mm
import common_config.mpl_cfg as mpl_cfg
import common_config.ISTF_fid_params as ISTF_fid

sys.path.append(str(project_path / 'bin'))
import plots_FM_running as plot_utils

sys.path.append(str(project_path / 'jobs/SPV3_magcut_zcut/config'))
import config_SPV3_magcut_zcut as cfg

# plot config
matplotlib.rcParams.update(mpl_cfg.mpl_rcParams_dict)
matplotlib.use('Qt5Agg')
markersize = 10

########################################################################################################################

# ! options
zbins = 13
zbins_list = np.array((zbins,), dtype=int)
probes = ('3x2pt', 'WL', 'GC')
model = 'flat'
which_diff = 'normal'
flagship_version = 2
check_old_FM = False
pes_opt = 'opt'
which_uncertainty = 'marginal'
fix_IA = False
fix_gal_bias = False  # whether to remove the rows/cols for the shear bias nuisance parameters (ie whether to fix them)
fix_shear_bias = False  # whether to remove the rows/cols for the shear bias nuisance parameters (ie whether to fix them)
fix_dzWL = True  # whether to remove the rows/cols for the dz nuisance parameters (ie whether to fix them)
fix_dzGC = True  # whether to remove the rows/cols for the dz nuisance parameters (ie whether to fix them)
bar_plot_cosmo = True
triangle_plot = False
bar_plot_nuisance = False
dpi = 500
zmax = 25
EP_or_ED = 'ED'
n_cosmo_params = 8
pic_format = 'pdf'
plot_fom = True
# ! end options

# compute percent diff of the cases chosen - careful of the indices!
if which_diff == 'normal':
    diff_funct = mm.percent_diff
else:
    diff_funct = mm.percent_diff_mean

# ML_list = [230, 230, 245, 245]
# ZL_list = [0, 2, 0, 2]
# MS_list = [245, 245, 245, 245]
# ZS_list = [0, 0, 0, 2]

ML_list = [245, 245]
ZL_list = [0, 2]
MS_list = [245, 245]
ZS_list = [0, 2]

ML_list = [245]
ZL_list = [0]
MS_list = [245]
ZS_list = [0]

uncert_ratio_dict = {}
uncert_G_dict = {}
uncert_GS_dict = {}

# for probe in probes:
#     uncert_ratio_dict[probe] = {}
#     uncert_G_dict[probe] = {}
#     uncert_GS_dict[probe] = {}
#     for ML in ML_list:
#         uncert_ratio_dict[probe][ML] = {}
#         uncert_G_dict[probe][ML] = {}
#         uncert_GS_dict[probe][ML] = {}
#         for ZL in ZL_list:
#             uncert_ratio_dict[probe][ML][ZL] = {}
#             uncert_G_dict[probe][ML][ZL] = {}
#             uncert_GS_dict[probe][ML][ZL] = {}
#             for MS in MS_list:
#                 uncert_ratio_dict[probe][ML][ZL][MS] = {}
#                 uncert_G_dict[probe][ML][ZL][MS] = {}
#                 uncert_GS_dict[probe][ML][ZL][MS] = {}
#                 for ZS in ZS_list:
#                     uncert_ratio_dict[probe][ML][ZL][MS][ZS] = []
#                     uncert_G_dict[probe][ML][ZL][MS][ZS] = []
#                     uncert_GS_dict[probe][ML][ZL][MS][ZS] = []

for probe in probes:
    for ML, ZL, MS, ZS in zip(ML_list, ZL_list, MS_list, ZS_list):

        nparams_toplot = n_cosmo_params

        lmax = 3000
        nbl = 29
        if probe == 'WL':
            lmax = 5000
            nbl = 32

        FM_Ellcuts_path = f'/Users/davide/Documents/Lavoro/Programmi/SSC_restructured_v2/jobs/SPV3_magcut_zcut/output' \
                          f'/Flagship_{flagship_version}/FM/BNT_True/ell_cuts_True'
        FM_noEllcuts_path = FM_Ellcuts_path.replace('ell_cuts_True', 'ell_cuts_False')
        FM_filename = f'FM_zbins{EP_or_ED}{zbins:02d}-ML{ML:03d}-ZL{ZL:02d}-MS{MS:03d}-ZS{ZS:02d}.pickle'

        FM_Ellcuts_dict = mm.load_pickle(f'{FM_Ellcuts_path}/{FM_filename}')
        FM_noEllcuts_dict = mm.load_pickle(f'{FM_noEllcuts_path}/{FM_filename}')

        # these should not change when passed the second time to the function
        _params = FM_noEllcuts_dict['parameters']
        _fid = FM_noEllcuts_dict['fiducial_values']

        FM_GO_Ellcuts = FM_Ellcuts_dict[f'FM_{probe}_GO']
        FM_GS_Ellcuts = FM_Ellcuts_dict[f'FM_{probe}_GS']
        FM_GO_noEllcuts = FM_noEllcuts_dict[f'FM_{probe}_GO']
        FM_GS_noEllcuts = FM_noEllcuts_dict[f'FM_{probe}_GS']

        # fix the desired parameters and remove null rows/columns
        FM_GO_noEllcuts, param_names, fid = mm.mask_FM(FM_GO_noEllcuts, _params, _fid, n_cosmo_params, fix_IA,
                                                       fix_gal_bias)
        FM_GS_noEllcuts, _, _ = mm.mask_FM(FM_GS_noEllcuts, _params, _fid, n_cosmo_params, fix_IA, fix_gal_bias)
        FM_GO_Ellcuts, _, _ = mm.mask_FM(FM_GO_Ellcuts, _params, _fid, n_cosmo_params, fix_IA, fix_gal_bias)
        FM_GS_Ellcuts, _, _ = mm.mask_FM(FM_GS_Ellcuts, _params, _fid, n_cosmo_params, fix_IA, fix_gal_bias)
        wzwa_idx = [param_names.index('wz'), param_names.index('wa')]
        assert len(fid) == len(param_names), 'the fiducial values list and parameter names should have the same length'

        FMs = [FM_GO_noEllcuts, FM_GS_noEllcuts, FM_GO_Ellcuts, FM_GS_Ellcuts]

        # cases = ['FM_GO_noEllcuts', 'FM_GS_noEllcuts', 'FM_GO_Ellcuts', 'FM_GS_Ellcuts', 'abs(percent_diff)']
        cases = ['FM_GO_noEllcuts', 'FM_GO_Ellcuts', 'abs(percent_diff)']
        # cases = ['FM_GO_noEllcuts', 'FM_GS_noEllcuts', 'abs(percent_diff)']
        key_to_compare_A, key_to_compare_B = cases[1], cases[0]  # which cases to take the percent diff and ratio of

        data = []
        fom = {}
        uncert = {}
        for FM, case in zip(FMs, cases):
            uncert[case] = np.asarray(mm.uncertainties_FM(FM, nparams=nparams_toplot, fiducials=fid[:nparams_toplot],
                                                          which_uncertainty=which_uncertainty, normalize=True))
            fom[case] = mm.compute_FoM(FM, w0wa_idxs=wzwa_idx) / 10
            print(f'FoM({probe}, {case}): {fom[case]}')

        uncert['abs(percent_diff)'] = np.abs(diff_funct(uncert[key_to_compare_A], uncert[key_to_compare_B]))
        uncert['ratio'] = uncert[key_to_compare_A] / uncert[key_to_compare_B]

        for case in cases:
            data.append(uncert[case])

        # # store uncertainties in dictionaries to easily retrieve them in the different cases
        # uncert_G_dict[probe][ML][ZL][MS][ZS] = uncert['G']
        # uncert_GS_dict[probe][ML][ZL][MS][ZS] = uncert['GS']
        # uncert_ratio_dict[probe][ML][ZL][MS][ZS] = uncert['ratio']
        # # append the FoM values at the end of the array
        # uncert_ratio_dict[probe][ML][ZL][MS][ZS] = np.append(
        #     uncert_ratio_dict[probe][ML][ZL][MS][ZS], fom['GS'] / fom['G'])

        data = np.asarray(data)
        param_names_label = param_names[:nparams_toplot]

        if plot_fom:
            fom_array = np.array([fom[key_to_compare_A], fom[key_to_compare_B],
                                  np.abs(mm.percent_diff(fom[key_to_compare_A], fom[key_to_compare_B]))])
            param_names_label += ['FoM/10']
            nparams_toplot += 1
            data = np.column_stack((data, fom_array))

        title = '%s, $\\ell_{\\rm max} = %i$, zbins %s%i' % (probe, lmax, EP_or_ED, zbins)
        title += f'\nML = {ML / 10}, MS = {MS / 10}, ZL = {ZL / 10}, ZS = {ZS / 10}, zmax = {zmax / 10}'

        plot_utils.bar_plot(data[:, :nparams_toplot], title, cases, nparams=nparams_toplot,
                            param_names_label=param_names_label, bar_width=0.15)

        # plt.savefig(job_path / f'output/Flagship_{flagship_version}/plots/'
        #                        f'bar_plot_{probe}_ellmax{lmax}_zbins{EP_or_ED}{zbins:02}'
        #                        f'_ZL{zcut_lens:02d}_MS{magcut_source:03d}_ZS{zcut_source:02d}.png')
