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
probes = ('WL',)
pes_opt_list = ('opt',)
EP_or_ED_list = ('ED',)
which_comparison = 'GO_vs_GS'  # this is just to set the title of the plot
which_Rl = 'var'
nparams_chosen = 7
which_job = 'SPV3'
model = 'flat'
which_diff = 'normal'
flagship_version = 2
check_old_FM = False
pes_opt = 'opt'
which_uncertainty = 'marginal'
fix_gal_bias = True  # whether to remove the rows/cols for the shear bias nuisance parameters (ie whether to fix them)
fix_shear_bias = True  # whether to remove the rows/cols for the shear bias nuisance parameters (ie whether to fix them)
fix_dzWL = True  # whether to remove the rows/cols for the dz nuisance parameters (ie whether to fix them)
fix_dzGC = True  # whether to remove the rows/cols for the dz nuisance parameters (ie whether to fix them)
w0wa_rows = [2, 3]
bar_plot_cosmo = True
triangle_plot = False
plot_prior_contours = False
bar_plot_nuisance = False
pic_format = 'pdf'
BNT_transform = False
dpi = 500
magcut_lens = 230
magcut_source = 245
zcut_lens = 0
zcut_source = 0
zmax = 25

nparams_toplot = 8
EP_or_ED = 'ED'
# ! end options

ML_list = [230, 230, 245, 245]
ZL_list = [0, 2, 0, 2]
MS_list = [245, 245, 245, 245]
ZS_list = [0, 0, 2, 2]

job_path = project_path / f'jobs/{which_job}'
uncert_ratio_dict = {}
uncert_G_dict = {}
uncert_GS_dict = {}

# TODO fix this
if which_job == 'SPV3':
    nbl = 32
else:
    raise ValueError

# compute percent diff of the cases chosen - careful of the indices!
if which_diff == 'normal':
    diff_funct = mm.percent_diff
else:
    diff_funct = mm.percent_diff_mean

for probe in probes:
    uncert_ratio_dict[probe] = {}
    for ML in ML_list:
        uncert_ratio_dict[probe][ML] = {}
        for ZL in ZL_list:
            uncert_ratio_dict[probe][ML][ZL] = {}
            for MS in MS_list:
                uncert_ratio_dict[probe][ML][ZL][MS] = {}
                for ZS in ZS_list:
                    uncert_ratio_dict[probe][ML][ZL][MS][ZS] = []

for probe in probes:
    for ML, ZL, MS, ZS in zip(ML_list, ZL_list, MS_list, ZS_list):

        lmax = 3000
        nbl = 29
        if probe == 'WL':
            lmax = 5000
            nbl = 32

        FM_path = f'/Users/davide/Documents/Lavoro/Programmi/SSC_restructured_v2/jobs/SPV3_magcut_zcut/output/Flagship_{flagship_version}/BNT_{BNT_transform}/FM'
        FM_GO_filename = f'FM_{probe}_GO_lmax{lmax}_nbl{nbl}_zbins{EP_or_ED:s}{zbins:02d}' \
                         f'-ML{magcut_lens:d}-ZL{zcut_lens:02d}-MS{magcut_source:d}-ZS{zcut_source:02d}.txt'
        FM_GS_filename = f'FM_{probe}_GS_lmax{lmax}_nbl{nbl}_zbins{EP_or_ED:s}{zbins:02d}' \
                         f'-ML{magcut_lens:d}-ZL{zcut_lens:02d}-MS{magcut_source:d}-ZS{zcut_source:02d}.txt'

        FM_GO = np.genfromtxt(f'{FM_path}/{FM_GO_filename}')
        FM_GS = np.genfromtxt(f'{FM_path}/{FM_GS_filename}')

        # param names
        paramnames_cosmo = ["Om", "Ox", "Ob", "wz", "wa", "h", "ns", "s8"]
        paramnames_IA = ["Aia", "eIA", "bIA"]
        paramnames_galbias = [f'bG{zbin_idx:02d}' for zbin_idx in range(1, zbins + 1)]
        paramnames_shearbias = [f'm{zbin_idx:02d}' for zbin_idx in range(1, zbins + 1)]
        paramnames_dzWL = [f'dzWL{zbin_idx:02d}' for zbin_idx in range(1, zbins + 1)]
        paramnames_dzGC = [f'dzGC{zbin_idx:02d}' for zbin_idx in range(1, zbins + 1)]
        paramnames_3x2pt = paramnames_cosmo + paramnames_IA + paramnames_galbias + paramnames_shearbias + \
                           paramnames_dzWL + paramnames_dzGC

        # TODO decide which parameters to fix
        # if fix_shear_bias:
        #     paramnames_3x2pt = [param for param in paramnames_3x2pt if "m" not in param]

        # fiducial values
        ng_folder = f'{project_path.parent}/common_data/vincenzo/SPV3_07_2022/Flagship_{flagship_version}/InputNz/magcut_zcut'
        ng_filename = f'ngbsTab-{EP_or_ED:s}{zbins:02d}-zedMin{zcut_source:02d}-zedMax{zmax:02d}-mag{magcut_source:03d}.dat'

        fid_cosmo = [0.32, 0.68, 0.05, -1.0, 0.0, 0.67, 0.96, 0.816]  # ! Added Ox fiducial value
        # fid_cosmo = np.asarray([ISTF_fid.primary[key] for key in ISTF_fid.primary.keys()])[:7]
        fid_IA = np.asarray([ISTF_fid.IA_free[key] for key in ISTF_fid.IA_free.keys()])
        fid_galaxy_bias = np.genfromtxt(f'{ng_folder}/{ng_filename}')[:, 2]
        fid_shear_bias = np.zeros((zbins,))
        fid_dzWL = np.zeros((zbins,))
        fid_dzGC = np.zeros((zbins,))
        fid = np.concatenate((fid_cosmo, fid_IA, fid_galaxy_bias, fid_shear_bias, fid_dzWL, fid_dzGC))
        assert len(fid) == len(paramnames_3x2pt), 'the fiducial values list and parameter names should have the same length'

        # cut the fid, paramname and FM arrays/lists according to the parameters we want to fix
        paramnames_galbias_indices = [paramnames_3x2pt.index(param) for param in paramnames_galbias]
        paramnames_shearbias_indices = [paramnames_3x2pt.index(param) for param in paramnames_shearbias]
        paramnames_dzWL_indices = [paramnames_3x2pt.index(param) for param in paramnames_dzWL]
        paramnames_dzGC_indices = [paramnames_3x2pt.index(param) for param in paramnames_dzGC]

        idx_todelete = []
        if fix_gal_bias:
            idx_todelete.append(paramnames_galbias_indices)
        if fix_shear_bias:
            idx_todelete.append(paramnames_shearbias_indices)
        if fix_dzWL:
            idx_todelete.append(paramnames_dzWL_indices)
        if fix_dzGC:
            idx_todelete.append(paramnames_dzGC_indices)

        # delete these from everything
        fid = np.delete(fid, idx_todelete)
        paramnames_3x2pt = np.delete(paramnames_3x2pt, idx_todelete)
        FM_GO = np.delete(FM_GO, idx_todelete, axis=0)
        FM_GO = np.delete(FM_GO, idx_todelete, axis=1)
        FM_GS = np.delete(FM_GS, idx_todelete, axis=0)
        FM_GS = np.delete(FM_GS, idx_todelete, axis=1)

        # some checks
        assert which_diff in ['normal', 'mean'], 'which_diff should be "normal" or "mean"'
        assert which_uncertainty in ['marginal',
                                     'conditional'], 'which_uncertainty should be "marginal" or "conditional"'
        assert which_Rl in ['const', 'var'], 'which_Rl should be "const" or "var"'
        assert model in ['flat', 'nonflat'], 'model should be "flat" or "nonflat"'
        assert probe in ['WL', 'GC', '3x2pt'], 'probe should be "WL" or "GC" or "3x2pt"'
        assert which_job == 'SPV3', 'which_job should be "SPV3"'

        if pes_opt == 'opt':
            ell_max_WL = 5000
            ell_max_GC = 3000
        else:
            ell_max_WL = 1500
            ell_max_GC = 750

        ell_max = ell_max_WL

        # remove null rows and columns from FM, and corresponding entries from the fiducial values list and parameter names
        null_idx_GO = mm.find_null_rows_cols_2D(FM_GO)
        null_idx_GS = mm.find_null_rows_cols_2D(FM_GS)
        assert np.array_equal(null_idx_GO, null_idx_GS), 'the null rows/cols indices should be equal for GO and GS'

        if null_idx_GO is not None:
            FM_GO = mm.remove_null_rows_cols_array2D(FM_GO, null_idx_GO)
            FM_GS = mm.remove_null_rows_cols_array2D(FM_GS, null_idx_GO)
            paramnames_3x2pt = np.delete(paramnames_3x2pt, obj=null_idx_GO, axis=0)
            fid = np.delete(fid, obj=null_idx_GO, axis=0)
            assert len(fid) == len(
                paramnames_3x2pt), 'the fiducial values list and parameter names should have the same length'

        nparams = len(fid)

        title = '%s, $\\ell_{\\rm max} = %i$, zbins %s%i' % (probe, ell_max, EP_or_ED, zbins)
        title += f'\nML = {magcut_lens / 10}, MS = {magcut_source / 10}, ZL = {zcut_lens / 10}, ZS = {zcut_source / 10:}, zmax = 2.5'

        # TODO try with pandas dataframes

        # print('3', FM_GO.shape)
        # if model == 'flat':
        #     FM_GO = np.delete(FM_GO, obj=1, axis=0)
        #     FM_GO = np.delete(FM_GO, obj=1, axis=1)
        #     FM_GS = np.delete(FM_GS, obj=1, axis=0)
        #     FM_GS = np.delete(FM_GS, obj=1, axis=1)
        #     cosmo_params = 7
        # elif model == 'nonflat':
        #     w0wa_rows = [3, 4]  # Omega_DE is in position 1, so w0, wa are shifted by 1 position
        #     nparams += 1
        #     cosmo_params = 8
        #     fid = np.insert(arr=fid, obj=1, values=ISTF_fid.extensions['Om_Lambda0'], axis=0)
        # pars_labels_TeX = np.insert(arr=pars_labels_TeX, obj=1, values='$\\Omega_{\\rm DE, 0}$', axis=0)

        # fid = fid[:nparams]
        # pars_labels_TeX = pars_labels_TeX[:nparams]

        ####################################################################################################################

        cases = ('G', 'GS')
        FMs = (FM_GO, FM_GS)

        data = []
        fom = {}
        uncert = {}
        uncert_ratio_dict[probe][ML][ZL][MS][ZS] = []
        uncert_G_dict[probe][ML][ZL][MS][ZS] = []
        uncert_GS_dict[probe][ML][ZL][MS][ZS] = []
        for FM, case in zip(FMs, cases):
            uncert[case] = np.asarray(mm.uncertainties_FM(FM, nparams=nparams, fiducials=fid,
                                                          which_uncertainty=which_uncertainty, normalize=True))
            fom[case] = mm.compute_FoM(FM, w0wa_idxs=w0wa_rows)
            print(f'FoM({probe}, {case}): {fom[case]}')

        uncert['percent_diff'] = diff_funct(uncert['GS'], uncert['G'])
        uncert['ratio'] = uncert['GS'] / uncert['G']
        cases = ['G', 'GS', 'percent_diff']

        for case in cases:
            data.append(uncert[case])

        # store uncertainties in dictionaries to easily retrieve them in the different cases
        uncert_G_dict[probe][ML][ZL][MS][ZS] = uncert['G']
        uncert_GS_dict[probe][ML][ZL][MS][ZS] = uncert['GS']
        uncert_ratio_dict[probe][ML][ZL][MS][ZS] = uncert['ratio']
        # append the FoM values at the end of the array
        uncert_ratio_dict[probe][ML][ZL][MS][ZS] = np.append(
            uncert_ratio_dict[probe][ML][ZL][MS][ZS], fom['GS'] / fom['G'])

        for probe in probes:
            for zbins in zbins_list:
                for pes_opt in ('opt', 'pes'):
                    data = np.asarray(data)
                    plot_utils.bar_plot(data[:, :nparams_toplot], title, cases, nparams=nparams_toplot,
                                        param_names_label=paramnames_3x2pt[:nparams_toplot],
                                        bar_width=0.12,
                                        second_axis=False, no_second_axis_bars=1)

                # plt.savefig(job_path / f'output/plots/{which_comparison}/'
                #                        f'bar_plot_{probe}_ellmax{ell_max}_zbins{EP_or_ED}{zbins:02}_Rl{which_Rl}_{which_uncertainty}.png')
