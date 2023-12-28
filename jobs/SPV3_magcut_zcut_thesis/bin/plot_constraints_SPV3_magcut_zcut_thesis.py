import sys
import warnings
import matplotlib.cm as cm

import numpy as np
import pandas as pd
import yaml
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from chainconsumer import ChainConsumer
from tqdm import tqdm
%matplotlib qt

ROOT = '/Users/davide/Documents/Lavoro/Programmi'
SB_ROOT = f'{ROOT}/Spaceborne'


sys.path.append(SB_ROOT)
import bin.plots_FM_running as plot_utils
import bin.my_module as mm
import common_cfg.mpl_cfg as mpl_cfg

sys.path.append(f'{SB_ROOT}/jobs/config')
import jobs.SPV3_magcut_zcut_thesis.config.config_SPV3_magcut_zcut_thesis as cfg

mpl.rcParams.update(mpl_cfg.mpl_rcParams_dict)
mpl.use('Qt5Agg')
# Display all columns
pd.set_option('display.max_columns', None)
# Disable text wrapping within cells
pd.set_option('display.expand_frame_repr', False)


general_cfg = cfg.general_cfg
h_over_mpc_tex = mpl_cfg.h_over_mpc_tex
kmax_tex = mpl_cfg.kmax_tex
kmax_star_tex = mpl_cfg.kmax_star_tex
cosmo_params_tex = mpl_cfg.general_dict['cosmo_labels_TeX']


# ! options
specs_str = 'idIA2_idB3_idM3_idR1'
fm_root_path = ('/Users/davide/Documents/Lavoro/Programmi/Spaceborne/'
                'jobs/SPV3_magcut_zcut_thesis/output/Flagship_2/FM')
fm_path_raw = fm_root_path + '/BNT_{BNT_transform!s}/ell_cuts_{ell_cuts!s}'
fm_pickle_name_raw = 'FM_{which_ng_cov:s}_{ng_cov_code:s}_zbins{EP_or_ED:s}{zbins:02d}_' \
                    'ML{ML:03d}_ZL{ZL:02d}_MS{MS:03d}_ZS{ZS:02d}_{specs_str:s}_pk{which_pk:s}{which_grids:s}.pickle'
EP_or_ED = 'EP'
zbins = 13
num_params_tokeep = 7
fix_curvature = True
fix_gal_bias = False
fix_dz = True
fix_shear_bias = False  # this has to be an outer loop if you also want to vary the shear bias prior itself
include_fom = True
fid_shear_bias_prior = 5e-4
shear_bias_prior = fid_shear_bias_prior
gal_bias_perc_prior = None  # ! not quite sure this works properly...
string_columns = ['probe', 'which_cov_term', 'whose_FM', 'which_pk', 'BNT_transform', 'ell_cuts', 'which_cuts',
                  'center_or_min', 'fix_dz', 'fix_shear_bias', 'foc', 'kmax_h_over_Mpc']
triangle_plot = False
use_Wadd = False  # the difference is extremely small
pk_ref = 'HMCodebar'
fom_redbook = 400
target_perc_dispersion = 10  # percent
w0_uncert_redbook = 2  # percent
wa_uncert_redbook = 10  # percent
ML = 245
MS = 245
ZL = 2
ZS = 2
probes = ('WL', 'GC', '3x2pt')
which_cuts = 'Vincenzo'
whose_FM_list = ('davide',)
kmax_h_over_Mpc_plt = general_cfg['kmax_h_over_Mpc_list'][0]  # some cases are indep of kamx, just take the fist one

which_cov_term_list = ['G', 'GSSC']
ng_cov_code = 'PyCCL'  # exactSSC or PyCCL
which_grids = '_densegrids'  # '_defaultgrids' or '_CSSTgrids' or '_densegrids' or grids used for k and a arrays in pyccl
which_ng_cov = which_cov_term_list[1]
BNT_transform_list = [False, ]
center_or_min_list = ['center']
kmax_h_over_Mpc_list = (general_cfg['kmax_h_over_Mpc_list'][0],)
kmax_1_over_Mpc_vinc_str_list = ['025', '050', '075', '100', '125', '150', '175', '200', '300',
                                 '500', '1000', '1500', '2000']
# kmax_1_over_Mpc_vinc_list = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 3.00, 5.00, 10.00, 15.00, 20.00]

ell_cuts_list = [False, ]
fix_dz_list = [True, False]
fix_shear_bias_list = [True, False]
which_pk_list = (general_cfg['which_pk_list'][0], )
center_or_min_plt = 'center'
which_cuts_plt = 'Vincenzo'
save_plots = False
plor_corr_matrix = True
# ! options

probe_vinc_dict = {
    'WL': 'WLO',
    'GC': 'GCO',
    '3x2pt': '3x2pt',
}
num_string_columns = len(string_columns)
fm_uncert_df = pd.DataFrame()
correlation_dict = {}

if ng_cov_code == 'exactSSC':
    which_grids = ''

# quick check: between PyCCL SSC FMs computed with different grids
fm_dict_a = mm.load_pickle('/Users/davide/Documents/Lavoro/Programmi/Spaceborne/jobs/SPV3_magcut_zcut_thesis/output/Flagship_2/FM/BNT_False/ell_cuts_False/FM_GSSC_PyCCL_zbinsEP13_ML245_ZL02_MS245_ZS02_idIA2_idB3_idM3_idR1_pkHMCodebar_CSSTgrids.pickle')
fm_dict_b = mm.load_pickle('/Users/davide/Documents/Lavoro/Programmi/Spaceborne/jobs/SPV3_magcut_zcut_thesis/output/Flagship_2/FM/BNT_False/ell_cuts_False/FM_GSSC_PyCCL_zbinsEP13_ML245_ZL02_MS245_ZS02_idIA2_idB3_idM3_idR1_pkHMCodebar_densegrids.pickle')

for key in fm_dict_a.keys():
    if key not in ['param_names_dict', 'fiducial_values_dict', 'fiducials_dict_flattened']:
        print('CSSTgrids == densegrids?', key, np.allclose(fm_dict_a[key], fm_dict_b[key], rtol=1e-3, atol=0))


# TODO understand nan instead of None in the fm_uncert_df
# TODO maybe there is a bettewr way to handle the prior values in relation to the fix flag
# TODO superimpose bars

assert fix_curvature, 'I am studying only flat models'
assert 'Flagship_2' in fm_root_path, 'The input files used in this job for flagship version 2!'
assert which_cuts == 'Vincenzo', ('to begin with, use only Vincenzo/standard cuts. '
                                  'For the thesis, probably use just these')
assert not use_Wadd, 'import of Wadd not implemented yet'


for probe in probes:
    for BNT_transform in BNT_transform_list:
        for which_cov_term in which_cov_term_list:
            for which_pk in which_pk_list:
                for ell_cuts in ell_cuts_list:
                    for kmax_counter, kmax_h_over_Mpc in enumerate(kmax_h_over_Mpc_list):
                        for whose_FM in whose_FM_list:
                            for center_or_min in center_or_min_list:
                                for fix_dz in fix_dz_list:
                                    for fix_shear_bias in fix_shear_bias_list:

                                        shear_bias_prior = fid_shear_bias_prior

                                        if BNT_transform is False:
                                            ell_cuts = False
                                        # ! this needs to be switched off if you want to test whether BNT matches std implementation
                                        elif BNT_transform is True:
                                            ell_cuts = True

                                        if which_cov_term == 'GSSC':
                                            which_pk = 'HMCodebar'  # GSSC is only availane in this case
                                        

                                        names_params_to_fix = []

                                        if whose_FM == 'davide':
                                            fm_path = fm_path_raw.format(BNT_transform=BNT_transform, ell_cuts=ell_cuts)
                                            fm_pickle_name = fm_pickle_name_raw.format(which_ng_cov=which_ng_cov,
                                                                                        EP_or_ED=EP_or_ED,
                                                                                        zbins=zbins,
                                                                                        ML=ML, ZL=ZL, MS=MS, ZS=ZS,
                                                                                        specs_str=specs_str,
                                                                                        which_pk=which_pk,
                                                                                        ng_cov_code=ng_cov_code,
                                                                                        which_grids=which_grids)
                                            if ell_cuts:
                                                fm_path += f'/{which_cuts}/ell_{center_or_min}'
                                                fm_pickle_name = fm_pickle_name.replace(f'.pickle',
                                                                                        f'_kmaxhoverMpc{kmax_h_over_Mpc:.03f}.pickle')

                                            fm_dict = mm.load_pickle(f'{fm_path}/{fm_pickle_name}')

                                            fm = fm_dict[f'FM_{probe}_{which_cov_term}']

                                        elif whose_FM == 'vincenzo':

                                            # for vincenzo's file names - now I'm using a different grid
                                            # kmax_1_over_Mpc = int(np.round(kmax_h_over_Mpc * h * 100))

                                            kmax_1_over_Mpc = kmax_1_over_Mpc_vinc_str_list[kmax_counter]

                                            fm_path = '/Users/davide/Documents/Lavoro/Programmi/common_data/vincenzo/SPV3_07_2022/' \
                                                    f'LiFEforSPV3/OutputFiles/FishMat/GaussOnly/Flat/{which_pk}/TestKappaMax'
                                            fm_name = f'fm-{probe_vinc_dict[probe]}-{EP_or_ED}{zbins}-ML{ML}-MS{MS}-{specs_str.replace("_", "-")}' \
                                                    f'-kM{kmax_1_over_Mpc}.dat'
                                            fm = np.genfromtxt(f'{fm_path}/{fm_name}')

                                        if probe == '3x2pt' and use_Wadd:
                                            assert False, 'import of Wadd not implemented for Vincenzos FM yet'
                                            fm_wa = np.genfromtxt(
                                                f'{fm_path.replace("3x2pt", "WA")}/{fm_name.replace("3x2pt", "WA")}')
                                            fm += fm_wa

                                        # TODO probably better a yaml file, like below
                                        # with open('/Users/davide/Documents/Lavoro/Programmi/common_lib_and_cfg/common_config/'
                                        #           'fiducial_params_dict_for_FM.yml') as f:
                                        #     fiducials_dict = yaml.safe_load(f)
                                        fiducials_dict = fm_dict['fiducial_values_dict']
                                        h = fiducials_dict['h']

                                        assert fm.shape[0] == fm.shape[1], 'FM matrix is not square!'
                                        assert len(fiducials_dict) == fm.shape[0], 'FM matrix and fiducial parameters ' \
                                                                                'length do not match!'

                                        # fix some of the parameters (i.e., which columns to remove)
                                        # if fix_curvature:
                                        # print('fixing curvature')
                                        # names_params_to_fix += ['ODE']
                                        # else:
                                        #     num_params_tokeep += 1

                                        if fix_shear_bias:
                                            # print('fixing shear bias parameters')
                                            names_params_to_fix += [f'm{(zi + 1):02d}' for zi in range(zbins)]
                                            # in this way, 👇there is no need for a 'add_shear_bias_prior' (or similar) boolean flag
                                            shear_bias_prior = None

                                        if fix_gal_bias:
                                            # print('fixing galaxy bias parameters')
                                            names_params_to_fix += [f'bG{(zi + 1):02d}' for zi in range(zbins)]
                                            gal_bias_perc_prior = None

                                        if fix_dz:
                                            # print('fixing dz parameters')
                                            names_params_to_fix += [f'dzWL{(zi + 1):02d}' for zi in range(zbins)]

                                        fm, fiducials_dict = mm.mask_fm_v2(fm, fiducials_dict, names_params_to_fix,
                                                                        remove_null_rows_cols=True)

                                        param_names = list(fiducials_dict.keys())
                                        cosmo_param_names = list(fiducials_dict.keys())[:num_params_tokeep]

                                        # ! add prior on shear and/or gal bias
                                        if shear_bias_prior != None and probe in ['WL', '3x2pt']:
                                            shear_bias_param_names = [f'm{(zi + 1):02d}' for zi in range(zbins)]
                                            shear_bias_prior_values = np.array([shear_bias_prior] * zbins)
                                            fm = mm.add_prior_to_fm(fm, fiducials_dict, shear_bias_param_names,
                                                                    shear_bias_prior_values)

                                        if gal_bias_perc_prior != None and probe in ['GC', '3x2pt']:
                                            gal_bias_param_names = [f'bG{(zi + 1):02d}' for zi in range(zbins)]

                                            # go from sigma_b / b_fid to sigma_b
                                            gal_bias_idxs = [param_names.index(gal_bias_param_name)
                                                            for gal_bias_param_name in gal_bias_param_names]

                                            gal_bias_fid_values = np.array(list(fiducials_dict.values()))[gal_bias_idxs]
                                            gal_bias_prior_values = gal_bias_perc_prior * gal_bias_fid_values / 100
                                            fm = mm.add_prior_to_fm(fm, fiducials_dict, gal_bias_param_names,
                                                                    gal_bias_prior_values)

                                        # ! triangle plot
                                        if triangle_plot:
                                            if probe == '3x2pt' and which_cov_term == 'GSSC' and fix_shear_bias == False:
                                                # decide params to show in the triangle plot
                                                shear_bias_param_names = [f'm{(zi + 1):02d}_photo' for zi in
                                                                        range(zbins)]
                                                params_tot_list = cosmo_param_names + shear_bias_param_names

                                                trimmed_fid_dict = {param: fiducials_dict[param] for param in
                                                                    params_tot_list}

                                                # get the covariance matrix (careful on how you cut the FM!!)
                                                fm_idxs_tokeep = [list(fiducials_dict.keys()).index(param) for param in
                                                                params_tot_list]
                                                cov = np.linalg.inv(fm)[fm_idxs_tokeep, :][:, fm_idxs_tokeep]

                                                plot_utils.contour_plot_chainconsumer(cov, trimmed_fid_dict)

                                        # ! compute uncertainties from fm
                                        uncert_fm = mm.uncertainties_fm_v2(fm, fiducials_dict,
                                                                        which_uncertainty='marginal',
                                                                        normalize=True,
                                                                        percent_units=True)[:num_params_tokeep]

                                        # compute the FoM
                                        w0wa_idxs = param_names.index('wz'), param_names.index('wa')
                                        fom = mm.compute_FoM(fm, w0wa_idxs)

                                        # ! this piece of code is for the foc of the different cases
                                        corr_mat = mm.correlation_from_covariance(np.linalg.inv(fm))[:num_params_tokeep, :num_params_tokeep]
                                        foc = mm.figure_of_correlation(corr_mat)
                                        if plor_corr_matrix and which_cov_term == 'G' and BNT_transform is False and \
                                                ell_cuts is False and fix_dz is True and fix_shear_bias is False and \
                                                kmax_h_over_Mpc == kmax_h_over_Mpc_list[-1] and which_pk:
                                            correlation_dict[which_pk] = corr_mat

                                        df_columns_names = string_columns + [param_name for param_name in
                                                                            fiducials_dict.keys()][
                                                                            :num_params_tokeep] + ['FoM']

                                        # this is a list of lists just to have a 'row list' instead of a 'column list',
                                        # I still haven't figured out the problem, but in this way it works
                                        df_columns_values = [
                                            [probe, which_cov_term, whose_FM, which_pk, BNT_transform, ell_cuts, which_cuts,
                                            center_or_min, fix_dz, fix_shear_bias, foc, kmax_h_over_Mpc] +
                                            uncert_fm.tolist() + [fom]]

                                        assert len(df_columns_names) == len(
                                            df_columns_values[0]), 'Wrong number of columns!'

                                        fm_uncert_df_to_concat = pd.DataFrame(df_columns_values,
                                                                            columns=df_columns_names)
                                        fm_uncert_df = pd.concat([fm_uncert_df, fm_uncert_df_to_concat],
                                                                ignore_index=True)
                                        fm_uncert_df = fm_uncert_df.drop_duplicates()  # ! drop duplicates from df!!

# ! ============================================================ PLOTS ============================================================


# # ! bar plot
probe_toplot = '3x2pt'
include_fom = False

fm_uncert_df_toplot = fm_uncert_df[
    (fm_uncert_df['probe'] == probe_toplot) &
    (fm_uncert_df['whose_FM'] == 'davide') &
    (fm_uncert_df['which_pk'] == pk_ref) &
    (fm_uncert_df['fix_dz'] == False) &
    (fm_uncert_df['fix_shear_bias'] == False) &
    (fm_uncert_df['BNT_transform'] == False) &
    (fm_uncert_df['ell_cuts'] == False) &
    (fm_uncert_df['kmax_h_over_Mpc'] == 0.1) &
    (fm_uncert_df['which_cuts'] == which_cuts_plt) &
    (fm_uncert_df['center_or_min'] == center_or_min_plt)
    ]

fm_uncert_df_toplot = mm.compare_df_keys(fm_uncert_df_toplot, 'which_cov_term', which_cov_term_list[0],
                                      which_cov_term_list[1], num_string_columns)

data = fm_uncert_df_toplot.iloc[:, num_string_columns:].values
label_list = list(fm_uncert_df_toplot['which_cov_term'].values)
label_list = ['None' if value is None else value for value in label_list]

if include_fom:
    num_params_tokeep += 1
data = data[:, :num_params_tokeep]

ylabel = f'relative uncertainty [%]'
plot_utils.bar_plot(data, f'{probe_toplot}, {which_cov_term_list[1]}, {ng_cov_code}', label_list, bar_width=0.2, nparams=num_params_tokeep,
                    param_names_label=None,
                    second_axis=False, no_second_axis_bars=0, superimpose_bars=False, show_markers=False, ylabel=ylabel,
                    include_fom=include_fom, figsize=(10, 8))
# plt.savefig('../output/plots/WL_vs_GC_vs_3x2pt_GGSSC_perc_uncert_increase.pdf', bbox_inches='tight', dpi=600)



assert False, 'checking SSC btw pyccl and exactSSC'
# mm.plot_correlation_matrix(correlation_dict['HMCode2020'] / correlation_dict['TakaBird'], cosmo_params_tex,
                        #    title='HMCodebar/TakaBird')
if save_plots:
    plt.savefig('/Users/davide/Documents/Lavoro/Programmi/phd_thesis_plots/plots/correlation_matrix.pdf',
                bbox_inches='tight', dpi=500)

# ! check difference between ell_cuts True and False
df_true = fm_uncert_df[(fm_uncert_df['ell_cuts'] == True) &
                       (fm_uncert_df['kmax_h_over_Mpc'] == kmax_h_over_Mpc_list[-1])].iloc[:,
          num_string_columns:].values
df_false = fm_uncert_df[(fm_uncert_df['ell_cuts'] == False) &
                        (fm_uncert_df['kmax_h_over_Mpc'] == kmax_h_over_Mpc_list[-1])].iloc[:,
           num_string_columns:].values
diff = (df_true / df_false - 1) * 100
mm.matshow(diff, log=True, title=f'difference between ell_cuts True, kmax = {kmax_h_over_Mpc_list[-1]:.2f} and False')

# ! plot FoM pk_ref vs kmax
probe_toplot = '3x2pt'
reduced_df = fm_uncert_df[
    (fm_uncert_df['probe'] == probe_toplot) &
    (fm_uncert_df['which_cov_term'] == 'G') &
    (fm_uncert_df['whose_FM'] == 'davide') &
    (fm_uncert_df['which_pk'] == pk_ref) &
    (fm_uncert_df['BNT_transform'] == True) &
    (fm_uncert_df['ell_cuts'] == True) &
    (fm_uncert_df['which_cuts'] == which_cuts_plt) &
    (fm_uncert_df['center_or_min'] == center_or_min_plt)
    ]
fom_dz_false_sb_false = reduced_df[(reduced_df['fix_dz'] == False) &
                                   (reduced_df['fix_shear_bias'] == False)
                                   ]['FoM'].values
fom_dz_true_sb_false = reduced_df[(reduced_df['fix_dz'] == True) &
                                  (reduced_df['fix_shear_bias'] == False)
                                  ]['FoM'].values
fom_dz_false_sb_true = reduced_df[(reduced_df['fix_dz'] == False) &
                                  (reduced_df['fix_shear_bias'] == True)
                                  ]['FoM'].values
fom_dz_true_sb_true = reduced_df[(reduced_df['fix_dz'] == True) &
                                 (reduced_df['fix_shear_bias'] == True)
                                 ]['FoM'].values

# add FoM for no ell cuts case
fom_noellcuts = fm_uncert_df[
    (fm_uncert_df['probe'] == probe_toplot) &
    (fm_uncert_df['which_cov_term'] == 'G') &
    (fm_uncert_df['whose_FM'] == 'davide') &
    (fm_uncert_df['which_pk'] == pk_ref) &
    (fm_uncert_df['BNT_transform'] == False) &
    (fm_uncert_df['ell_cuts'] == False) &
    (fm_uncert_df['fix_dz'] == True) &
    (fm_uncert_df['fix_shear_bias'] == False) &
    (fm_uncert_df['which_cuts'] == which_cuts_plt) &
    (fm_uncert_df['center_or_min'] == center_or_min_plt) &
    (fm_uncert_df['kmax_h_over_Mpc'] == kmax_h_over_Mpc_plt)
    ]['FoM'].values[0]

# find kmax for a given FoM (400)
kmax_fom400_dz_false_sb_false = mm.find_inverse_from_array(kmax_h_over_Mpc_list, fom_dz_false_sb_false, fom_redbook)
kmax_fom400_dz_true_sb_false = mm.find_inverse_from_array(kmax_h_over_Mpc_list, fom_dz_true_sb_false, fom_redbook)
kmax_fom400_dz_false_sb_true = mm.find_inverse_from_array(kmax_h_over_Mpc_list, fom_dz_false_sb_true, fom_redbook)
kmax_fom400_dz_true_sb_true = mm.find_inverse_from_array(kmax_h_over_Mpc_list, fom_dz_true_sb_true, fom_redbook)

fom_ref = fom_dz_true_sb_false
kmax_fom400_ref = kmax_fom400_dz_true_sb_false

dz_tex = '$\\Delta z_i$'
sb_tex = '$m_i$'
title_plot = '3$\\times$2pt' if probe_toplot == '3x2pt' else None
plt.figure()
plt.plot(kmax_h_over_Mpc_list, fom_dz_false_sb_false, label=f'{dz_tex} free, {sb_tex} free', marker='o')
plt.plot(kmax_h_over_Mpc_list, fom_dz_true_sb_false, label=f'{dz_tex} fixed, {sb_tex} free (ref)', marker='o')
plt.plot(kmax_h_over_Mpc_list, fom_dz_false_sb_true, label=f'{dz_tex} free, {sb_tex} fixed', marker='o')
plt.plot(kmax_h_over_Mpc_list, fom_dz_true_sb_true, label=f'{dz_tex} fixed, {sb_tex} fixed', marker='o')
plt.axvline(kmax_fom400_dz_false_sb_false,
            label=f'{kmax_star_tex} = {kmax_fom400_dz_false_sb_false:.02f} {h_over_mpc_tex}', c='tab:blue', ls='--')
plt.axvline(kmax_fom400_dz_true_sb_false,
            label=f'{kmax_star_tex} = {kmax_fom400_dz_true_sb_false:.02f} {h_over_mpc_tex}', c='tab:orange', ls='--')
plt.axvline(kmax_fom400_dz_false_sb_true,
            label=f'{kmax_star_tex} = {kmax_fom400_dz_false_sb_true:.02f} {h_over_mpc_tex}', c='tab:green', ls='--')
plt.axvline(kmax_fom400_dz_true_sb_true, label=f'{kmax_star_tex} = {kmax_fom400_dz_true_sb_true:.02f} {h_over_mpc_tex}',
            c='tab:red', ls='--')
plt.axhline(fom_noellcuts, label='$\\ell_{\\rm max, opt}^{\\rm EC20} = 3000$', c='k', ls=':')
plt.axhline(fom_redbook, label=f'FoM = {fom_redbook}', c='k', ls='-', alpha=0.3)
plt.xlabel(f'{kmax_tex} [{h_over_mpc_tex}]')
plt.ylabel('3$\\times$2pt FoM')
plt.legend()

if save_plots:
    plt.savefig('/Users/davide/Documents/Lavoro/Programmi/phd_thesis_plots/plots/fom_hmcodebar_vs_kmax.pdf',
                bbox_inches='tight', dpi=500)

# ! plot cosmo pars vs kmax
center_or_min = 'center'
# choose what to plot
cosmo_params_df_dav = fm_uncert_df[
    (fm_uncert_df['probe'] == probe_toplot) &
    (fm_uncert_df['which_cov_term'] == 'G') &
    (fm_uncert_df['whose_FM'] == 'davide') &
    (fm_uncert_df['BNT_transform'] == BNT_transform) &
    (fm_uncert_df['ell_cuts'] == ell_cuts) &
    (fm_uncert_df['fix_dz'] == True) &
    (fm_uncert_df['fix_shear_bias'] == False) &
    (fm_uncert_df['center_or_min'] == center_or_min) &
    (fm_uncert_df['which_pk'] == pk_ref)
    ]

plt.figure()
for i, cosmo_param in enumerate(cosmo_param_names):
    plt.plot(kmax_h_over_Mpc_list, cosmo_params_df_dav[cosmo_param].values, label=f'{cosmo_params_tex[i]}', marker='o')
plt.axvline(kmax_fom400_ref, label=f'{kmax_star_tex} = {kmax_fom400_ref:.02f} {h_over_mpc_tex}', c='k', ls='--')

plt.ylabel('relative uncertainty [%]')
plt.xlabel(f'{kmax_tex} [{h_over_mpc_tex}]')
plt.legend()
plt.show()

if save_plots:
    plt.savefig('/Users/davide/Documents/Lavoro/Programmi/phd_thesis_plots/plots/cosmo_params_vs_kmax.pdf',
                bbox_inches='tight',
                dpi=500)

# ! plot different pks
center_or_min = 'center'
param_toplot = 'FoM'
# choose what to plot
cosmo_params_df_dav = fm_uncert_df[
    (fm_uncert_df['probe'] == '3x2pt') &
    (fm_uncert_df['which_cov_term'] == 'G') &
    (fm_uncert_df['whose_FM'] == 'davide') &
    (fm_uncert_df['fix_dz'] == True) &
    (fm_uncert_df['fix_shear_bias'] == False) &
    (fm_uncert_df['BNT_transform'] == BNT_transform) &
    (fm_uncert_df['ell_cuts'] == ell_cuts) &
    (fm_uncert_df['center_or_min'] == center_or_min)
    ]

# FoM dispersion between different pks, for each kmax
reduced_unc_df = cosmo_params_df_dav.groupby('kmax_h_over_Mpc')[param_toplot]
mean_fom_vs_kmax = reduced_unc_df.mean().values
stdev_fom_vs_kmax = reduced_unc_df.std().values
perc_deviation_vs_kmax = stdev_fom_vs_kmax / mean_fom_vs_kmax * 100

reduced_unc_df_notb = cosmo_params_df_dav.loc[
    cosmo_params_df_dav['which_pk'] != 'TakaBird'].groupby('kmax_h_over_Mpc')[param_toplot]

mean_fom_vs_kmax_notb = reduced_unc_df_notb.mean().values
stdev_fom_vs_kmax_notb = reduced_unc_df_notb.std().values
perc_deviation_vs_kmax_notb = stdev_fom_vs_kmax_notb / mean_fom_vs_kmax_notb * 100

# ! cutting the values above which the trend is no longer monothonic
kmax_perc_deviation = mm.find_inverse_from_array(kmax_h_over_Mpc_list[:6], perc_deviation_vs_kmax[:6],
                                                 target_perc_dispersion)
kmax_perc_deviation_notb = mm.find_inverse_from_array(kmax_h_over_Mpc_list[:6], perc_deviation_vs_kmax_notb[:6],
                                                      target_perc_dispersion)

kmax_fom400_tb = mm.find_inverse_from_array(kmax_h_over_Mpc_list,
                                            cosmo_params_df_dav[cosmo_params_df_dav['which_pk'] == 'TakaBird'][
                                                'FoM'].values, fom_redbook)
fom_kmax082_notb = mm.find_inverse_from_array(mean_fom_vs_kmax_notb, kmax_h_over_Mpc_list, kmax_perc_deviation_notb)

print(f'kmax_fom400_tb = {kmax_fom400_tb:.2f} {h_over_mpc_tex} for lmax = 3000. P. Taylor finds 0.7!!; make sure youre '
      f'fixing shear bias and dz for a fair comparison')
print(f'FoM = {fom_kmax082_notb:.2f} for kmax = {kmax_perc_deviation_notb:.2f} {h_over_mpc_tex} and no TakaBird.')

colors = cm.rainbow(np.linspace(0, 1, len(which_pk_list)))
plt.figure()
for i, which_pk in enumerate(which_pk_list):
    fom_vs_kmax_dav = cosmo_params_df_dav[cosmo_params_df_dav['which_pk'] == which_pk][param_toplot].values
    plt.plot(kmax_h_over_Mpc_list, fom_vs_kmax_dav, label=f'{which_pk}', marker='o', c=colors[i])
plt.errorbar(kmax_h_over_Mpc_list, mean_fom_vs_kmax, yerr=stdev_fom_vs_kmax, label='$\\mu \\pm \\sigma$',
             marker='.', c='k', ls=':', lw=2)
plt.axhline(fom_redbook, label=f'FoM = {fom_redbook}', c='k', ls='-', alpha=0.3)
plt.axvline(kmax_perc_deviation, label=f'{kmax_star_tex} = {kmax_perc_deviation:.2f} {h_over_mpc_tex}', c=colors[1],
            ls='--', alpha=0.5)
plt.axvline(kmax_perc_deviation_notb, label=f'{kmax_star_tex} = {kmax_perc_deviation_notb:.2f} {h_over_mpc_tex}',
            c='tab:green',
            ls='--', alpha=0.5)

plt.ylabel('3$\\times$2pt FoM')
plt.xlabel(f'{kmax_tex} [{h_over_mpc_tex}]')
plt.yscale('log')
plt.legend()
plt.show()
if save_plots:
    plt.savefig('/Users/davide/Documents/Lavoro/Programmi/phd_thesis_plots/plots/fom_vs_kmax_vs_pk.pdf',
                bbox_inches='tight',
                dpi=500)

# ! plot FoC
plt.figure()
for i, which_pk in enumerate(which_pk_list):
    foc = cosmo_params_df_dav[cosmo_params_df_dav['which_pk'] == which_pk]['foc'].values
    plt.plot(kmax_h_over_Mpc_list, foc, label=f'{which_pk}', marker='o', c=colors[i])

plt.ylabel('3$\\times$2pt FoC')
plt.xlabel(f'{kmax_tex} [{h_over_mpc_tex}]')
plt.yscale('log')
plt.legend()
plt.show()
if save_plots:
    plt.savefig('/Users/davide/Documents/Lavoro/Programmi/phd_thesis_plots/plots/foc_vs_kmax_vs_pk.pdf',
                bbox_inches='tight',
                dpi=500)

# ! plot FoM pk_ref go gs vs kmax
probe_toplot = '3x2pt'
params_toplot = [*cosmo_param_names, 'FoM']

if probe_toplot != '3x2pt':
    assert False, ('the ssc from spaceborne is only available for 3x2pt, not for WL or GC. Cut the 3x2pt '
                   'appropriately if you '
                   'want the other probes...')
ell_cuts = False
BNT_transform = False
go_gs_df = fm_uncert_df[
    (fm_uncert_df['probe'] == probe_toplot) &
    (fm_uncert_df['whose_FM'] == 'davide') &
    (fm_uncert_df['which_pk'] == pk_ref) &
    (fm_uncert_df['fix_dz'] == True) &
    (fm_uncert_df['fix_shear_bias'] == True) &
    (fm_uncert_df['BNT_transform'] == BNT_transform) &
    (fm_uncert_df['ell_cuts'] == ell_cuts) &
    (fm_uncert_df['which_cuts'] == which_cuts_plt) &
    (fm_uncert_df['center_or_min'] == center_or_min_plt)
    ]

go_gs_df = mm.compare_df_keys(go_gs_df, 'which_cov_term', which_cov_term_list[0], which_cov_term_list[1], num_string_columns)

cosmo_params_tex_plusfom = cosmo_params_tex + ['FoM']
plt.figure()
for i, param in enumerate(params_toplot):
    plt.plot(kmax_h_over_Mpc_list, go_gs_df[go_gs_df['which_cov_term'] == 'perc_diff'][param].values,
             label=f'{cosmo_params_tex_plusfom[i]}', marker='o')
plt.ylabel(f'perc_diff {probe_toplot}')
plt.xlabel(f'{kmax_tex} [{h_over_mpc_tex}]')
plt.legend()
plt.show()
#
# # find kmax for a given FoM (400)
# kmax_fom_400 = mm.find_inverse_from_array(kmax_h_over_Mpc_list, fom_values, fom_redbook)
#
# title_plot = '3$\\times$2pt' if probe_toplot == '3x2pt' else None
# plt.figure()
# plt.plot(kmax_h_over_Mpc_list, fom_values, label=f'FoM', marker='o')
# plt.axvline(kmax_fom_400, label=f'{kmax_star_tex} = {kmax_fom_400:.02f} {h_over_mpc_tex}', c='tab:blue', ls='--')
# plt.axhline(fom_noellcuts, label='$\\ell_{\\rm max, opt}^{\\rm EC20} = 3000$', c='k', ls=':')
# plt.axhline(fom_redbook, label=f'FoM = {fom_redbook}', c='k', ls='-', alpha=0.3)
# plt.xlabel(f'{kmax_tex} [{h_over_mpc_tex}]')
# plt.ylabel('3$\\times$2pt FoM')
# plt.legend()


print('done')
