"""
this code snippet can be appended at the end of main.py for a quick and dirty FM
estimation. It is not intended to be used for a serious analysis.
"""

FM_ordered_params = {
    'Om': 0.32,
    'Ob': 0.05,
    'wz': -1.0,
    'wa': 0.0,
    'h': 0.6737,
    'ns': 0.966,
    's8': 0.816,
    'logT': 7.75,
    'Aia': 0.16,
    'eIA': 1.66,
    'm01': 0.0,
    'm02': 0.0,
    'm03': 0.0,
    'm04': 0.0,
    'm05': 0.0,
    'm06': 0.0,
    'm07': 0.0,
    'm08': 0.0,
    'm09': 0.0,
    'm10': 0.0,
    'm11': 0.0,
    'm12': 0.0,
    'm13': 0.0,
    'dzWL01': -0.025749,
    'dzWL02': 0.022716,
    'dzWL03': -0.026032,
    'dzWL04': 0.012594,
    'dzWL05': 0.019285,
    'dzWL06': 0.008326,
    'dzWL07': 0.038207,
    'dzWL08': 0.002732,
    'dzWL09': 0.034066,
    'dzWL10': 0.049479,
    'dzWL11': 0.06649,
    'dzWL12': 0.000815,
    'dzWL13': 0.04907,
    # coefficients for the polynomial magnification and galaxy bias fits
    'bG01': 1.33291,
    'bG02': -0.72414,
    'bG03': 1.0183,
    'bG04': -0.14913,
    'bM01': -1.50685,
    'bM02': 1.35034,
    'bM03': 0.08321,
    'bM04': 0.04279,
}


fm_cfg = {
    'GL_or_LG': 'GL',
    'compute_FM': True,
    'save_FM_txt': False,
    'save_FM_dict': True,
    'load_preprocess_derivatives': False,
    'which_derivatives': 'Vincenzo',  # Vincenzo or Spaceborne,
    'derivatives_folder': '{ROOT:s}/common_data/vincenzo/SPV3_07_2022/LiFEforSPV3_may24/OutputFiles/DataVecDers/{flat_or_nonflat:s}/{which_pk:s}/{EP_or_ED:s}{zbins:02d}',
    'derivatives_filename': 'dDVd{param_name:s}-{probe:s}-ML{magcut_lens:03d}-MS{magcut_source:03d}-{EP_or_ED:s}{zbins:02d}.dat',
    'derivatives_prefix': 'dDVd',
    'derivatives_BNT_transform': False,
    'deriv_ell_cuts': False,
    'fm_folder': '{ROOT:s}/common_data/Spaceborne/jobs/SPV3/output/Flagship_{flagship_version}/FM/BNT_{BNT_transform:s}/ell_cuts_{ell_cuts:s}',
    'fm_txt_filename': 'fm_txt_filename',
    'fm_dict_filename': f'FM_dict_sigma2b_simpsdav.pickle',
    'test_against_vincenzo': False,
    'test_against_benchmarks': False,
    'FM_ordered_params': FM_ordered_params,
    'ind': ind,
    'block_index': 'ell',
    'zbins': zbins,
    'compute_SSC': True,
}


from spaceborne import fisher_matrix as fm_utils

flat_or_nonflat = 'Flat'
magcut_lens = 245  # valid for GCph
magcut_source = 245  # valid for WL
zmin_nz_lens = 2  # = 0.2
zmin_nz_source = 2  # = 0.2
zmax_nz = 25  # = 2.5
idIA = 2
idB = 3
idM = 3
idR = 1
idBM = 3  # for the SU responses
ep_or_ed = 'EP'
ROOT = '/home/davide/Documenti/Lavoro/Programmi'

variable_specs = {
    'flat_or_nonflat': flat_or_nonflat,
    'which_pk': 'HMCodeBar',
    'EP_or_ED': ep_or_ed,
    'zbins': zbins,
}

# list_params_to_vary = list(FM_ordered_params.keys())
list_params_to_vary = [param for param in FM_ordered_params.keys() if param != 'ODE']
# list_params_to_vary = ['h', 'wa', 'dzWL01', 'm06', 'bG02', 'bM02']
# list_params_to_vary = ['bM02', ]


# Vincenzo's derivatives
der_prefix = fm_cfg['derivatives_prefix']
derivatives_folder = fm_cfg['derivatives_folder'].format(**variable_specs, ROOT=ROOT)
fm_dict_filename = fm_cfg['fm_dict_filename'].format(**variable_specs, ROOT=ROOT)
# ! get vincenzo's derivatives' parameters, to check that they match with the yaml file
# check the parameter names in the derivatives folder, to see whether I'm setting the correct ones in the config file
vinc_filenames = sl.get_filenames_in_folder(derivatives_folder)
vinc_filenames = [
    vinc_filename
    for vinc_filename in vinc_filenames
    if vinc_filename.startswith(der_prefix)
]

# keep only the files corresponding to the correct magcut_lens, magcut_source and zbins
vinc_filenames = [
    filename
    for filename in vinc_filenames
    if all(
        x in filename
        for x in [f'ML{magcut_lens}', f'MS{magcut_source}', f'{ep_or_ed}{zbins:02d}']
    )
]
vinc_filenames = [filename.replace('.dat', '') for filename in vinc_filenames]

vinc_trimmed_filenames = [
    vinc_filename.split('-', 1)[0].strip() for vinc_filename in vinc_filenames
]
vinc_trimmed_filenames = [
    vinc_trimmed_filename[len(der_prefix) :]
    if vinc_trimmed_filename.startswith(der_prefix)
    else vinc_trimmed_filename
    for vinc_trimmed_filename in vinc_trimmed_filenames
]
vinc_param_names = list(set(vinc_trimmed_filenames))
vinc_param_names.sort()

# ! get fiducials names and values from the yaml file
# remove ODE if I'm studying only flat models
if flat_or_nonflat == 'Flat' and 'ODE' in FM_ordered_params:
    FM_ordered_params.pop('ODE')
fm_fid_dict = FM_ordered_params
param_names_3x2pt = list(fm_fid_dict.keys())
fm_cfg['param_names_3x2pt'] = param_names_3x2pt
fm_cfg['nparams_tot'] = len(param_names_3x2pt)

# sort them to compare with vincenzo's param names
my_sorted_param_names = param_names_3x2pt.copy()
my_sorted_param_names.sort()

for dzgc_param_name in [f'dzGC{zi:02d}' for zi in range(1, zbins + 1)]:
    if (
        dzgc_param_name in vinc_param_names
    ):  # ! added this if statement, not very elegant
        vinc_param_names.remove(dzgc_param_name)

# check whether the 2 lists match and print the elements that are in one list but not in the other
param_names_not_in_my_list = [
    vinc_param_name
    for vinc_param_name in vinc_param_names
    if vinc_param_name not in my_sorted_param_names
]
param_names_not_in_vinc_list = [
    my_sorted_param_name
    for my_sorted_param_name in my_sorted_param_names
    if my_sorted_param_name not in vinc_param_names
]

# Check if the parameter names match
if not np.all(vinc_param_names == my_sorted_param_names):
    # Print the mismatching parameters
    print(
        f'Params present in input folder but not in the cfg file: {param_names_not_in_my_list}'
    )
    print(
        f'Params present in cfg file but not in the input folder: {param_names_not_in_vinc_list}'
    )

# ! preprocess derivatives (or load the alreay preprocessed ones)
if fm_cfg['load_preprocess_derivatives']:
    warnings.warn(
        'loading preprocessed derivatives is faster but a bit more dangerous, make sure all the specs are taken into account'
    )
    dC_LL_4D = np.load(f'{derivatives_folder}/reshaped_into_4d_arrays/dC_LL_4D.npy')
    dC_GG_4D = np.load(f'{derivatives_folder}/reshaped_into_4d_arrays/dC_GG_4D.npy')
    dC_3x2pt_6D = np.load(
        f'{derivatives_folder}/reshaped_into_4d_arrays/dC_3x2pt_6D.npy'
    )

elif not fm_cfg['load_preprocess_derivatives']:
    der_prefix = fm_cfg['derivatives_prefix']
    dC_dict_1D = dict(sl.get_kv_pairs_v2(derivatives_folder, 'dat'))
    # check if dictionary is empty
    if not dC_dict_1D:
        raise ValueError(f'No derivatives found in folder {derivatives_folder}')

    # separate in 4 different dictionaries and reshape them (no interpolation needed in this case)
    dC_dict_LL_3D = {}
    dC_dict_GG_3D = {}
    dC_dict_3x2pt_5D = {}

    for key in vinc_filenames:  # loop over these, I already selected ML, MS and so on
        if not key.startswith('dDVddzGC'):
            if 'WLO' in key:
                dC_dict_LL_3D[key] = cl_utils.cl_SPV3_1D_to_3D(
                    dC_dict_1D[key], 'WL', nbl_WL_opt, zbins
                )[:nbl_WL, :, :]
            elif 'GCO' in key:
                dC_dict_GG_3D[key] = cl_utils.cl_SPV3_1D_to_3D(
                    dC_dict_1D[key], 'GC', nbl_GC, zbins
                )
            elif '3x2pt' in key:
                dC_dict_3x2pt_5D[key] = cl_utils.cl_SPV3_1D_to_3D(
                    dC_dict_1D[key], '3x2pt', nbl_3x2pt, zbins
                )

    # turn the dictionaries of derivatives into npy array of shape (nbl, zbins, zbins, nparams)
    dC_LL_4D_vin = fm_utils.dC_dict_to_4D_array(
        dC_dict_LL_3D, param_names_3x2pt, nbl_WL, zbins, der_prefix
    )
    dC_GG_4D_vin = fm_utils.dC_dict_to_4D_array(
        dC_dict_GG_3D, param_names_3x2pt, nbl_GC, zbins, der_prefix
    )
    dC_3x2pt_6D_vin = fm_utils.dC_dict_to_4D_array(
        dC_dict_3x2pt_5D, param_names_3x2pt, nbl_3x2pt, zbins, der_prefix, is_3x2pt=True
    )

    # free up memory
    del dC_dict_1D, dC_dict_LL_3D, dC_dict_GG_3D, dC_dict_3x2pt_5D

    # save these so they can simply be imported!
    if not os.path.exists(f'{derivatives_folder}/reshaped_into_np_arrays'):
        os.makedirs(f'{derivatives_folder}/reshaped_into_np_arrays')
    np.save(f'{derivatives_folder}/reshaped_into_np_arrays/dC_LL_4D.npy', dC_LL_4D_vin)
    np.save(f'{derivatives_folder}/reshaped_into_np_arrays/dC_GG_4D.npy', dC_GG_4D_vin)
    np.save(
        f'{derivatives_folder}/reshaped_into_np_arrays/dC_3x2pt_6D.npy', dC_3x2pt_6D_vin
    )

deriv_dict_vin = {
    'dC_LL_4D': dC_LL_4D_vin,
    'dC_GG_4D': dC_GG_4D_vin,
    'dC_3x2pt_6D': dC_3x2pt_6D_vin,
}

# ! ==================================== compute and save fisher matrix ================================================
fm_dict_vin = fm_utils.compute_FM(
    cfg['covariance'], fm_cfg, ell_dict, cov_dict, deriv_dict_vin, bnt_matrix
)

# TODO finish testing derivatives
# fm_dict_dav = fm_utils.compute_FM(cfg, ell_dict, cov_dict, deriv_dict_dav, bnt_matrix)
# fm_dict_vin_modified = {key + '_vin': value for key, value in fm_dict_vin.items()}
# del fm_dict_vin_modified['fiducial_values_dict_vin']
# fm_dict = {**fm_dict_dav, **fm_dict_vin_modified}

fm_dict = fm_dict_vin

# ordered fiducial parameters entering the FM
fm_dict['fiducial_values_dict'] = fm_cfg['FM_ordered_params']

fm_folder = '/home/davide/Documenti/Lavoro/Programmi/common_data/Spaceborne/jobs/SPV3/output/Flagship_2/FM/BNT_False/ell_cuts_False'
from spaceborne import plot_lib

fm_dict_filename = fm_cfg['fm_dict_filename']
if fm_cfg['save_FM_dict']:
    sl.save_pickle(f'{fm_folder}/{fm_dict_filename}', fm_dict)

# ! plot the results directly, as a quick check
nparams_toplot = 7
names_params_to_fix = []
divide_fom_by_10 = True
include_fom = True
which_uncertainty = 'marginal'

fix_dz = True
fix_shear_bias = True
fix_gal_bias = False
fix_mag_bias = False
shear_bias_prior = 5e-4
# dz_prior = np.array(2 * 1e-3 * (1 + np.array(cfg['covariance_cfg']['zbin_centers'])))

probes = ['WL', 'GC', 'XC', '3x2pt']
dz_param_names = [f'dzWL{(zi + 1):02d}' for zi in range(zbins)]
shear_bias_param_names = [f'm{(zi + 1):02d}' for zi in range(zbins)]
gal_bias_param_names = [f'bG{(zi + 1):02d}' for zi in range(4)]
mag_bias_param_names = [f'bM{(zi + 1):02d}' for zi in range(4)]
param_names_list = list(FM_ordered_params.keys())

if fix_dz:
    names_params_to_fix += dz_param_names

if fix_shear_bias:
    names_params_to_fix += shear_bias_param_names

if fix_gal_bias:
    names_params_to_fix += gal_bias_param_names

if fix_mag_bias:
    names_params_to_fix += mag_bias_param_names

fom_dict = {}
uncert_dict = {}
masked_fm_dict = {}
masked_fid_pars_dict = {}
perc_diff_probe = {}
fm_dict_toplot = deepcopy(fm_dict)
del fm_dict_toplot['fiducial_values_dict']
for key in list(fm_dict_toplot.keys()):
    if key != 'fiducial_values_dict' and '_WA_' not in key and '_2x2pt_' not in key:
        print(key)

        fm = deepcopy(fm_dict_toplot[key])

        masked_fm_dict[key], masked_fid_pars_dict[key] = sl.mask_fm_v2(
            fm,
            FM_ordered_params,
            names_params_to_fix=names_params_to_fix,
            remove_null_rows_cols=True,
        )

        if not fix_shear_bias and any(item in key for item in ['WL', 'XC', '3x2pt']):
            print(f'adding shear bias Gaussian prior to {key}')
            shear_bias_prior_values = np.array([shear_bias_prior] * zbins)
            masked_fm_dict[key] = sl.add_prior_to_fm(
                masked_fm_dict[key],
                masked_fid_pars_dict[key],
                shear_bias_param_names,
                shear_bias_prior_values,
            )

        if not fix_dz:
            print(f'adding dz Gaussian prior to {key}')
            masked_fm_dict[key] = sl.add_prior_to_fm(
                masked_fm_dict[key], masked_fid_pars_dict[key], dz_param_names, dz_prior
            )

        uncert_dict[key] = sl.uncertainties_fm_v2(
            masked_fm_dict[key],
            masked_fid_pars_dict[key],
            which_uncertainty=which_uncertainty,
            normalize=True,
            percent_units=True,
        )[:nparams_toplot]

        param_names = list(masked_fid_pars_dict[key].keys())
        cosmo_param_names = list(masked_fid_pars_dict[key].keys())[:nparams_toplot]

        w0wa_idxs = param_names.index('wz'), param_names.index('wa')
        fom_dict[key] = sl.compute_FoM(masked_fm_dict[key], w0wa_idxs=w0wa_idxs)

# compute percent diff btw Gauss and G+SSC, using the respective Gaussian covariance
for probe in probes:
    key_a = f'FM_{probe}_G'
    key_b = f'FM_{probe}_Gtot'

    uncert_dict[f'perc_diff_{probe}_G'] = sl.percent_diff(
        uncert_dict[key_b], uncert_dict[key_a]
    )
    fom_dict[f'perc_diff_{probe}_G'] = np.abs(
        sl.percent_diff(fom_dict[key_b], fom_dict[key_a])
    )

    nparams_toplot = 7
    divide_fom_by_10_plt = False if probe in ('WLXC') else divide_fom_by_10

    cases_to_plot = [
        f'FM_{probe}_G',
        f'FM_{probe}_Gtot',
        # f'FM_{probe}_GSSCcNG',
        f'perc_diff_{probe}_G',
        #  f'FM_{probe}_{which_ng_cov_suffix}',
        #  f'perc_diff_{probe}_{which_ng_cov_suffix}',
    ]

    # # transform dict. into an array and add the fom
    uncert_array, fom_array = [], []

    for case in cases_to_plot:
        uncert_array.append(uncert_dict[case])
        if divide_fom_by_10 and 'FM' in case and 'WL' not in case:
            fom_dict[case] /= 10
        fom_array.append(fom_dict[case])

    uncert_array = np.asarray(uncert_array)
    fom_array = np.asarray(fom_array)

    uncert_array = np.hstack((uncert_array, fom_array.reshape(-1, 1)))

    perc_diff_probe[probe] = np.append(
        uncert_dict[f'perc_diff_{probe}_G'], fom_dict[f'perc_diff_{probe}_G']
    )

    # label and title stuff
    fom_label = 'FoM/10\nperc_diff' if divide_fom_by_10 else 'FoM'
    param_names_label = (
        param_names_list[:nparams_toplot] + [fom_label]
        if include_fom
        else param_names_list[:nparams_toplot]
    )
    lmax = (
        cfg['ell_binning'][f'ell_max_{probe}']
        if probe in ['WL', 'GC']
        else cfg['ell_binning']['ell_max_3x2pt']
    )
    title = '%s, $\\ell_{\\rm max} = %i$, zbins %s%i, zsteps %s\n%s uncertainties' % (
        probe,
        lmax,
        ep_or_ed,
        zbins,
        len(z_grid),
        which_uncertainty,
    )

    # bar plot
    if include_fom:
        nparams_toplot = 8

    for i, case in enumerate(cases_to_plot):
        cases_to_plot[i] = case
        if 'OneCovariance' in cases_to_plot[i]:
            cases_to_plot[i] = cases_to_plot[i].replace('OneCovariance', 'OneCov')
        if f'PySSC_{probe}_G' in cases_to_plot[i]:
            cases_to_plot[i] = cases_to_plot[i].replace(
                f'PySSC_{probe}_G', f'{probe}_G'
            )

        cases_to_plot[i] = cases_to_plot[i].replace(f'_{probe}', f'')
        cases_to_plot[i] = cases_to_plot[i].replace(f'FM_', f'')
        cases_to_plot[i] = cases_to_plot[i].replace(f'_', f' ')
        cases_to_plot[i] = cases_to_plot[i].replace(f'GSSC', f'G+SSC')
        cases_to_plot[i] = cases_to_plot[i].replace(f'SSCcNG', f'SSC+cNG')

    plot_lib.bar_plot(
        uncert_array[:, :nparams_toplot],
        title,
        cases_to_plot,
        nparams=nparams_toplot,
        param_names_label=param_names_label,
        bar_width=0.13,
        include_fom=include_fom,
        divide_fom_by_10_plt=divide_fom_by_10_plt,
    )

# ! % diff for the 3 probes - careful about the plot title
perc_diff_probe.pop('XC')
plot_lib.bar_plot(
    np.array(list(perc_diff_probe.values())),
    title + r', % diff (G + SSC + cNG)/G',
    (list(perc_diff_probe.keys())),
    nparams=nparams_toplot,
    param_names_label=param_names_label,
    bar_width=0.13,
    include_fom=include_fom,
    divide_fom_by_10_plt=False,
)

# ! Print tables

# if include_fom:
#     nparams_toplot_ref = nparams_toplot
#     nparams_toplot = nparams_toplot_ref + 1
# titles = param_names_list[:nparams_toplot_ref] + ['FoM']

# # for uncert_dict, _, name in zip([uncert_dict, uncert_dict], [fm_dict, fm_dict_vin], ['Davide', 'Vincenzo']):
# print(f"G uncertainties [%]:")
# data = []
# for probe in probes:
#     uncerts = [f'{uncert:.3f}' for uncert in uncert_dict[f'FM_{probe}_G']]
#     fom = f'{fom_dict[f"FM_{probe}_G"]:.2f}'
#     data.append([probe] + uncerts + [fom])
# print(tabulate(data, headers=titles, tablefmt="pretty"))

# print(f"GSSC/G ratio  :")
# data = []
# table = []  # tor tex
# for probe in probes:
#     ratios = [f'{ratio:.3f}' for ratio in uncert_dict[f'ratio_{probe}_G']]
#     fom = f'{fom_dict[f"ratio_{probe}_G"]:.2f}'
#     data.append([probe] + ratios + [fom])
#     table.append(ratios + [fom])
# print(tabulate(data, headers=titles, tablefmt="pretty"))

# print(f"SSC % increase :")
# data = []
# for probe in probes:
#     ratios = [f'{ratio:.3f}' for ratio in uncert_dict[f'perc_diff_{probe}_G']]
#     fom = f'{fom_dict[f"perc_diff_{probe}_G"]:.2f}'
#     data.append([probe] + ratios + [fom])
# print(tabulate(data, headers=titles, tablefmt="pretty"))

# ! quickly compare two selected FMs
# TODO this is misleading, understand better why (comparing GSSC, not perc_diff)

fm_dict_of_dicts = {
    'simps': sl.load_pickle(f'{fm_folder}/FM_dict_sigma2b_simpsdav.pickle'),
    'levin': sl.load_pickle(f'{fm_folder}/FM_dict_sigma2b_levindav.pickle'),
    # 'current': fm_dict,
}


labels = list(fm_dict_of_dicts.keys())
fm_dict_list = list(fm_dict_of_dicts.values())
keys_toplot_in = ['FM_WL_Gtot', 'FM_GC_Gtot', 'FM_XC_Gtot', 'FM_3x2pt_Gtot']
# keys_toplot = 'all'
colors = [
    'tab:blue',
    'tab:green',
    'tab:orange',
    'tab:red',
    'tab:cyan',
    'tab:grey',
    'tab:olive',
    'tab:purple',
]

reference = 'first_key'
nparams_toplot_in = 8
normalize_by_gauss = True

sl.compare_fm_constraints(
    *fm_dict_list,
    labels=labels,
    keys_toplot_in=keys_toplot_in,
    normalize_by_gauss=True,
    which_uncertainty='marginal',
    reference=reference,
    colors=colors,
    abs_FoM=True,
    save_fig=False,
    fig_path='/home/davide/Scrivania/',
)

assert False, 'stop here'

fisher_matrices = (
    fm_dict_of_dicts['SB_hm_simpker']['FM_3x2pt_GSSC'],
    fm_dict_of_dicts['SB_KEapp_hm_simpker']['FM_3x2pt_GSSC'],
    fm_dict_of_dicts['OC_simpker']['FM_3x2pt_GSSC'],
    fm_dict_of_dicts['current']['FM_3x2pt_GSSC'],
)
fiducials = list(fm_dict_of_dicts['SB_hm_simpker']['fiducial_values_dict'].values())
# fiducials = (
# fm_dict_of_dicts['SB_hm_simpker']['fiducial_values_dict'].values(),
# fm_dict_of_dicts['SB_KEapp_hm_simpker']['fiducial_values_dict'].values(),
# fm_dict_of_dicts['OC_simpker']['fiducial_values_dict'].values(),
# fm_dict_of_dicts['current']['fiducial_values_dict'].values(),
# )
param_names_list = list(
    fm_dict_of_dicts['SB_hm_simpker']['fiducial_values_dict'].keys()
)
param_names_labels_toplot = param_names_list[:8]
plot_lib.triangle_plot(
    fisher_matrices,
    fiducials,
    title,
    labels,
    param_names_list,
    param_names_labels_toplot,
    param_names_labels_tex=None,
    rotate_param_labels=False,
    contour_colors=None,
    line_colors=None,
)


print(
    'Finished in {:.2f} minutes'.format((time.perf_counter() - script_start_time) / 60)
)



"""
this code snippet can be appended at the end of main.py for a quick and dirty FM
estimation. It is not intended to be used for a serious analysis.
"""

FM_ordered_params = {
    'Om': 0.32,
    'Ob': 0.05,
    'wz': -1.0,
    'wa': 0.0,
    'h': 0.6737,
    'ns': 0.966,
    's8': 0.816,
    'logT': 7.75,
    'Aia': 0.16,
    'eIA': 1.66,
    'm01': 0.0,
    'm02': 0.0,
    'm03': 0.0,
    'm04': 0.0,
    'm05': 0.0,
    'm06': 0.0,
    'm07': 0.0,
    'm08': 0.0,
    'm09': 0.0,
    'm10': 0.0,
    'm11': 0.0,
    'm12': 0.0,
    'm13': 0.0,
    'dzWL01': -0.025749,
    'dzWL02': 0.022716,
    'dzWL03': -0.026032,
    'dzWL04': 0.012594,
    'dzWL05': 0.019285,
    'dzWL06': 0.008326,
    'dzWL07': 0.038207,
    'dzWL08': 0.002732,
    'dzWL09': 0.034066,
    'dzWL10': 0.049479,
    'dzWL11': 0.06649,
    'dzWL12': 0.000815,
    'dzWL13': 0.04907,
    # coefficients for the polynomial magnification and galaxy bias fits
    'bG01': 1.33291,
    'bG02': -0.72414,
    'bG03': 1.0183,
    'bG04': -0.14913,
    'bM01': -1.50685,
    'bM02': 1.35034,
    'bM03': 0.08321,
    'bM04': 0.04279,
}


fm_cfg = {
    'GL_or_LG': 'GL',
    'compute_FM': True,
    'save_FM_txt': False,
    'save_FM_dict': True,
    'load_preprocess_derivatives': False,
    'which_derivatives': 'Vincenzo',  # Vincenzo or Spaceborne,
    'derivatives_folder': '{ROOT:s}/common_data/vincenzo/SPV3_07_2022/LiFEforSPV3_may24/OutputFiles/DataVecDers/{flat_or_nonflat:s}/{which_pk:s}/{EP_or_ED:s}{zbins:02d}',
    'derivatives_filename': 'dDVd{param_name:s}-{probe:s}-ML{magcut_lens:03d}-MS{magcut_source:03d}-{EP_or_ED:s}{zbins:02d}.dat',
    'derivatives_prefix': 'dDVd',
    'derivatives_BNT_transform': False,
    'deriv_ell_cuts': False,
    'fm_folder': '{ROOT:s}/common_data/Spaceborne/jobs/SPV3/output/Flagship_{flagship_version}/FM/BNT_{BNT_transform:s}/ell_cuts_{ell_cuts:s}',
    'fm_txt_filename': 'fm_txt_filename',
    'fm_dict_filename': f'FM_dict_sigma2b_bigchanges_.pickle',
    'test_against_vincenzo': False,
    'test_against_benchmarks': False,
    'FM_ordered_params': FM_ordered_params,
    'ind': ind,
    'block_index': 'ell',
    'zbins': zbins,
    'compute_SSC': True,
}


param_names_3x2pt = [param for param in FM_ordered_params.keys() if param != 'ODE']
nparams_tot = len(param_names_3x2pt)

flat_or_nonflat = 'Flat'
magcut_lens = 245  # valid for GCph
magcut_source = 245  # valid for WL
zmin_nz_lens = 2  # = 0.2
zmin_nz_source = 2  # = 0.2
zmax_nz = 25  # = 2.5
idIA = 2
idB = 3
idM = 3
idR = 1
idBM = 3  # for the SU responses
ep_or_ed = 'EP'
ROOT = '/home/davide/Documenti/Lavoro/Programmi'

variable_specs = {
    'flat_or_nonflat': flat_or_nonflat,
    'which_pk': 'HMCodeBar',
    'EP_or_ED': ep_or_ed,
    'zbins': zbins,
}

# list_params_to_vary = list(FM_ordered_params.keys())
list_params_to_vary = [param for param in FM_ordered_params.keys() if param != 'ODE']
# list_params_to_vary = ['h', 'wa', 'dzWL01', 'm06', 'bG02', 'bM02']
# list_params_to_vary = ['bM02', ]


# Vincenzo's derivatives
der_prefix = fm_cfg['derivatives_prefix']
derivatives_folder = fm_cfg['derivatives_folder'].format(**variable_specs, ROOT=ROOT)
fm_dict_filename = fm_cfg['fm_dict_filename'].format(**variable_specs, ROOT=ROOT)
# ! get vincenzo's derivatives' parameters, to check that they match with the yaml file
# check the parameter names in the derivatives folder, to see whether I'm setting the correct ones in the config file
vinc_filenames = sl.get_filenames_in_folder(derivatives_folder)
vinc_filenames = [
    vinc_filename
    for vinc_filename in vinc_filenames
    if vinc_filename.startswith(der_prefix)
]

# keep only the files corresponding to the correct magcut_lens, magcut_source and zbins
vinc_filenames = [
    filename
    for filename in vinc_filenames
    if all(
        x in filename
        for x in [f'ML{magcut_lens}', f'MS{magcut_source}', f'{ep_or_ed}{zbins:02d}']
    )
]
vinc_filenames = [filename.replace('.dat', '') for filename in vinc_filenames]

vinc_trimmed_filenames = [
    vinc_filename.split('-', 1)[0].strip() for vinc_filename in vinc_filenames
]
vinc_trimmed_filenames = [
    vinc_trimmed_filename[len(der_prefix) :]
    if vinc_trimmed_filename.startswith(der_prefix)
    else vinc_trimmed_filename
    for vinc_trimmed_filename in vinc_trimmed_filenames
]
vinc_param_names = list(set(vinc_trimmed_filenames))
vinc_param_names.sort()

# ! get fiducials names and values from the yaml file
# remove ODE if I'm studying only flat models
if flat_or_nonflat == 'Flat' and 'ODE' in FM_ordered_params:
    FM_ordered_params.pop('ODE')
fm_fid_dict = FM_ordered_params
param_names_3x2pt = list(fm_fid_dict.keys())
fm_cfg['param_names_3x2pt'] = param_names_3x2pt
fm_cfg['nparams_tot'] = len(param_names_3x2pt)

# sort them to compare with vincenzo's param names
my_sorted_param_names = param_names_3x2pt.copy()
my_sorted_param_names.sort()

for dzgc_param_name in [f'dzGC{zi:02d}' for zi in range(1, zbins + 1)]:
    if (
        dzgc_param_name in vinc_param_names
    ):  # ! added this if statement, not very elegant
        vinc_param_names.remove(dzgc_param_name)

# check whether the 2 lists match and print the elements that are in one list but not in the other
param_names_not_in_my_list = [
    vinc_param_name
    for vinc_param_name in vinc_param_names
    if vinc_param_name not in my_sorted_param_names
]
param_names_not_in_vinc_list = [
    my_sorted_param_name
    for my_sorted_param_name in my_sorted_param_names
    if my_sorted_param_name not in vinc_param_names
]

# Check if the parameter names match
if not np.all(vinc_param_names == my_sorted_param_names):
    # Print the mismatching parameters
    print(
        f'Params present in input folder but not in the cfg file: {param_names_not_in_my_list}'
    )
    print(
        f'Params present in cfg file but not in the input folder: {param_names_not_in_vinc_list}'
    )

# ! preprocess derivatives (or load the alreay preprocessed ones)
if fm_cfg['load_preprocess_derivatives']:
    warnings.warn(
        'loading preprocessed derivatives is faster but a bit more dangerous, make sure all the specs are taken into account'
    )
    dC_LL_4D = np.load(f'{derivatives_folder}/reshaped_into_4d_arrays/dC_LL_4D.npy')
    dC_GG_4D = np.load(f'{derivatives_folder}/reshaped_into_4d_arrays/dC_GG_4D.npy')
    dC_3x2pt_6D = np.load(
        f'{derivatives_folder}/reshaped_into_4d_arrays/dC_3x2pt_6D.npy'
    )

elif not fm_cfg['load_preprocess_derivatives']:
    der_prefix = fm_cfg['derivatives_prefix']
    dC_dict_1D = dict(sl.get_kv_pairs_v2(derivatives_folder, 'dat'))
    # check if dictionary is empty
    if not dC_dict_1D:
        raise ValueError(f'No derivatives found in folder {derivatives_folder}')

    # separate in 4 different dictionaries and reshape them (no interpolation needed in this case)
    dC_dict_LL_3D = {}
    dC_dict_GG_3D = {}
    dC_dict_3x2pt_5D = {}

    for key in vinc_filenames:  # loop over these, I already selected ML, MS and so on
        if not key.startswith('dDVddzGC'):
            if 'WLO' in key:
                dC_dict_LL_3D[key] = cl_utils.cl_SPV3_1D_to_3D(
                    dC_dict_1D[key], 'WL', nbl_WL_opt, zbins
                )[:nbl_WL, :, :]
            elif 'GCO' in key:
                dC_dict_GG_3D[key] = cl_utils.cl_SPV3_1D_to_3D(
                    dC_dict_1D[key], 'GC', nbl_GC, zbins
                )
            elif '3x2pt' in key:
                dC_dict_3x2pt_5D[key] = cl_utils.cl_SPV3_1D_to_3D(
                    dC_dict_1D[key], '3x2pt', nbl_3x2pt, zbins
                )

    # turn the dictionaries of derivatives into npy array of shape (nbl, zbins, zbins, nparams)
    dC_LL_4D_vin = sl.dC_dict_to_4D_array(
        dC_dict_LL_3D, param_names_3x2pt, nbl_WL, zbins, der_prefix
    )
    dC_GG_4D_vin = sl.dC_dict_to_4D_array(
        dC_dict_GG_3D, param_names_3x2pt, nbl_GC, zbins, der_prefix
    )
    dC_3x2pt_6D_vin = sl.dC_dict_to_4D_array(
        dC_dict_3x2pt_5D, param_names_3x2pt, nbl_3x2pt, zbins, der_prefix, is_3x2pt=True
    )

    # free up memory
    del dC_dict_1D, dC_dict_LL_3D, dC_dict_GG_3D, dC_dict_3x2pt_5D

    # save these so they can simply be imported!
    if not os.path.exists(f'{derivatives_folder}/reshaped_into_np_arrays'):
        os.makedirs(f'{derivatives_folder}/reshaped_into_np_arrays')
    np.save(f'{derivatives_folder}/reshaped_into_np_arrays/dC_LL_4D.npy', dC_LL_4D_vin)
    np.save(f'{derivatives_folder}/reshaped_into_np_arrays/dC_GG_4D.npy', dC_GG_4D_vin)
    np.save(
        f'{derivatives_folder}/reshaped_into_np_arrays/dC_3x2pt_6D.npy', dC_3x2pt_6D_vin
    )

deriv_dict_vin = {
    'dC_LL_4D': dC_LL_4D_vin,
    'dC_GG_4D': dC_GG_4D_vin,
    'dC_3x2pt_6D': dC_3x2pt_6D_vin,
}


print('Starting covariance matrix inversion...')
start_time = time.perf_counter()
cov_WL_GO_2D_inv = np.linalg.inv(cov_dict['cov_WL_g_2D'])
cov_GC_GO_2D_inv = np.linalg.inv(cov_dict['cov_GC_g_2D'])
cov_XC_GO_2D_inv = np.linalg.inv(cov_dict['cov_XC_g_2D'])
cov_3x2pt_GO_2D_inv = np.linalg.inv(cov_dict['cov_3x2pt_g_2D'])
cov_WL_tot_2D_inv = np.linalg.inv(cov_dict['cov_WL_tot_2D'])
cov_GC_tot_2D_inv = np.linalg.inv(cov_dict['cov_GC_tot_2D'])
cov_XC_tot_2D_inv = np.linalg.inv(cov_dict['cov_XC_tot_2D'])
cov_3x2pt_tot_2D_inv = np.linalg.inv(cov_dict['cov_3x2pt_tot_2D'])
print('done in %.2f seconds' % (time.perf_counter() - start_time))


# load reshaped derivatives, with shape (nbl, zbins, zbins, nparams)
dC_LL_4D = deriv_dict_vin['dC_LL_4D']
dC_GG_4D = deriv_dict_vin['dC_GG_4D']
dC_3x2pt_6D = deriv_dict_vin['dC_3x2pt_6D']


dC_LLfor3x2pt_4D = dC_3x2pt_6D[0, 0, :, :, :, :]
dC_XCfor3x2pt_4D = dC_3x2pt_6D[0, 1, :, :, :, :]
dC_GGfor3x2pt_4D = dC_3x2pt_6D[1, 1, :, :, :, :]

# flatten z indices, obviously following the ordering given in ind
# separate the ind for the different probes
dC_LL_3D = sl.dC_4D_to_3D(dC_LL_4D, nbl_WL, zpairs_auto, nparams_tot, ind_auto)
dC_GG_3D = sl.dC_4D_to_3D(dC_GG_4D, nbl_GC, zpairs_auto, nparams_tot, ind_auto)
dC_LLfor3x2pt_3D = sl.dC_4D_to_3D(
    dC_LLfor3x2pt_4D, nbl_3x2pt, zpairs_auto, nparams_tot, ind_auto
)
dC_XCfor3x2pt_3D = sl.dC_4D_to_3D(
    dC_XCfor3x2pt_4D, nbl_3x2pt, zpairs_cross, nparams_tot, ind_cross
)
dC_GGfor3x2pt_3D = sl.dC_4D_to_3D(
    dC_GGfor3x2pt_4D, nbl_3x2pt, zpairs_auto, nparams_tot, ind_auto
)

# concatenate the flattened components of the 3x2pt datavector
dC_3x2pt_3D = np.concatenate(
    (dC_LLfor3x2pt_3D, dC_XCfor3x2pt_3D, dC_GGfor3x2pt_3D), axis=1
)


# collapse ell and zpair - ATTENTION: np.reshape, like ndarray.flatten, accepts an 'ordering' parameter, which works
# in the same way not with the old datavector, which was ordered in a different way...
block_index = 'ell'
if block_index in ['ell', 'vincenzo', 'C-style']:
    which_flattening = 'C'
elif block_index in ['ij', 'sylvain', 'F-style']:
    which_flattening = 'F'
else:
    raise ValueError(
        "block_index should be either 'ell', 'vincenzo', 'C-style', 'ij', 'sylvain' or 'F-style'"
    )

dC_LL_2D = np.reshape(
    dC_LL_3D, (nbl_WL * zpairs_auto, nparams_tot), order=which_flattening
)
dC_GG_2D = np.reshape(
    dC_GG_3D, (nbl_GC * zpairs_auto, nparams_tot), order=which_flattening
)
dC_XC_2D = np.reshape(
    dC_XCfor3x2pt_3D, (nbl_3x2pt * zpairs_cross, nparams_tot), order=which_flattening
)
dC_3x2pt_2D = np.reshape(
    dC_3x2pt_3D, (nbl_3x2pt * zpairs_3x2pt, nparams_tot), order=which_flattening
)

# ! cut the *flattened* derivatives vector
# if FM_cfg['deriv_ell_cuts']:
#     print('Performing the ell cuts on the derivatives...')
#     dC_LL_2D = np.delete(dC_LL_2D, ell_dict['idxs_to_delete_dict']['LL'], axis=0)
#     dC_GG_2D = np.delete(dC_GG_2D, ell_dict['idxs_to_delete_dict']['GG'], axis=0)
#     dC_WA_2D = np.delete(dC_WA_2D, ell_dict['idxs_to_delete_dict']['WA'], axis=0)
#     dC_XC_2D = np.delete(dC_XC_2D, ell_dict['idxs_to_delete_dict'][GL_or_LG], axis=0)
#     dC_3x2pt_2D = np.delete(dC_3x2pt_2D, ell_dict['idxs_to_delete_dict']['3x2pt'], axis=0)
#     # raise ValueError('the above cuts are correct, but I should be careful when defining the 2x2pt datavector/covmat,\
#         # as n_elem_ll will be lower because of the cuts...')

# # if the ell cuts removed all WA bins (which is in fact the case)
# if dC_WA_2D.shape[0] == 0:
#     dC_WA_2D = np.ones((nbl_WA * zpairs_auto, nparams_tot))

######################### COMPUTE FM #####################################

start = time.perf_counter()
FM_WL_GO = np.einsum(
    'ia,ik,kb->ab', dC_LL_2D, cov_WL_GO_2D_inv, dC_LL_2D, optimize='optimal'
)
FM_GC_GO = np.einsum(
    'ia,ik,kb->ab', dC_GG_2D, cov_GC_GO_2D_inv, dC_GG_2D, optimize='optimal'
)
FM_XC_GO = np.einsum(
    'ia,ik,kb->ab', dC_XC_2D, cov_XC_GO_2D_inv, dC_XC_2D, optimize='optimal'
)
FM_3x2pt_GO = np.einsum(
    'ia,ik,kb->ab', dC_3x2pt_2D, cov_3x2pt_GO_2D_inv, dC_3x2pt_2D, optimize='optimal'
)
print(f'GO FM done in {(time.perf_counter() - start):.2f} s')

start = time.perf_counter()
FM_WL_GS = np.einsum(
    'ia,ik,kb->ab', dC_LL_2D, cov_WL_tot_2D_inv, dC_LL_2D, optimize='optimal'
)
FM_GC_GS = np.einsum(
    'ia,ik,kb->ab', dC_GG_2D, cov_GC_tot_2D_inv, dC_GG_2D, optimize='optimal'
)
FM_XC_GS = np.einsum(
    'ia,ik,kb->ab', dC_XC_2D, cov_XC_tot_2D_inv, dC_XC_2D, optimize='optimal'
)
FM_3x2pt_GS = np.einsum(
    'ia,ik,kb->ab', dC_3x2pt_2D, cov_3x2pt_tot_2D_inv, dC_3x2pt_2D, optimize='optimal'
)
print(f'GS FM done in {(time.perf_counter() - start):.2f} s')


# store the matrices in the dictionary
probe_names = ['WL', 'GC', 'XC', '3x2pt']
FMs_GO = [FM_WL_GO, FM_GC_GO, FM_XC_GO, FM_3x2pt_GO]
FMs_GS = [FM_WL_GS, FM_GC_GS, FM_XC_GS, FM_3x2pt_GS]


FM_dict = {}
for probe_name, FM_GO, FM_GS in zip(probe_names, FMs_GO, FMs_GS):
    FM_dict[f'FM_{probe_name}_G'] = FM_GO
    FM_dict[f'FM_{probe_name}_TOT'] = FM_GS

print('FMs computed in %.2f seconds' % (time.perf_counter() - start))


# ! ==================================== compute and save fisher matrix ================================================


fm_dict = FM_dict

# ordered fiducial parameters entering the FM
fm_dict['fiducial_values_dict'] = fm_cfg['FM_ordered_params']

fm_folder = '/home/davide/Documenti/Lavoro/Programmi/common_data/Spaceborne/jobs/SPV3/output/Flagship_2/FM/BNT_False/ell_cuts_False'
from spaceborne import plot_lib

fm_dict_filename = fm_cfg['fm_dict_filename']
if fm_cfg['save_FM_dict']:
    sl.save_pickle(f'{fm_folder}/{fm_dict_filename}', fm_dict)

# ! plot the results directly, as a quick check
nparams_toplot = 7
names_params_to_fix = []
divide_fom_by_10 = True
include_fom = True
which_uncertainty = 'marginal'

fix_dz = True
fix_shear_bias = True
fix_gal_bias = False
fix_mag_bias = False
shear_bias_prior = 5e-4
# dz_prior = np.array(2 * 1e-3 * (1 + np.array(cfg['covariance_cfg']['zbin_centers'])))

probes = ['WL', 'GC', 'XC', '3x2pt']
dz_param_names = [f'dzWL{(zi + 1):02d}' for zi in range(zbins)]
shear_bias_param_names = [f'm{(zi + 1):02d}' for zi in range(zbins)]
gal_bias_param_names = [f'bG{(zi + 1):02d}' for zi in range(4)]
mag_bias_param_names = [f'bM{(zi + 1):02d}' for zi in range(4)]
param_names_list = list(FM_ordered_params.keys())

if fix_dz:
    names_params_to_fix += dz_param_names

if fix_shear_bias:
    names_params_to_fix += shear_bias_param_names

if fix_gal_bias:
    names_params_to_fix += gal_bias_param_names

if fix_mag_bias:
    names_params_to_fix += mag_bias_param_names

fom_dict = {}
uncert_dict = {}
masked_fm_dict = {}
masked_fid_pars_dict = {}
perc_diff_probe = {}
fm_dict_toplot = deepcopy(fm_dict)
del fm_dict_toplot['fiducial_values_dict']
for key in list(fm_dict_toplot.keys()):
    if key != 'fiducial_values_dict' and '_WA_' not in key and '_2x2pt_' not in key:
        print(key)

        fm = deepcopy(fm_dict_toplot[key])

        masked_fm_dict[key], masked_fid_pars_dict[key] = sl.mask_fm_v2(
            fm,
            FM_ordered_params,
            names_params_to_fix=names_params_to_fix,
            remove_null_rows_cols=True,
        )

        if not fix_shear_bias and any(item in key for item in ['WL', 'XC', '3x2pt']):
            print(f'adding shear bias Gaussian prior to {key}')
            shear_bias_prior_values = np.array([shear_bias_prior] * zbins)
            masked_fm_dict[key] = sl.add_prior_to_fm(
                masked_fm_dict[key],
                masked_fid_pars_dict[key],
                shear_bias_param_names,
                shear_bias_prior_values,
            )

        if not fix_dz:
            print(f'adding dz Gaussian prior to {key}')
            masked_fm_dict[key] = sl.add_prior_to_fm(
                masked_fm_dict[key], masked_fid_pars_dict[key], dz_param_names, dz_prior
            )

        uncert_dict[key] = sl.uncertainties_fm_v2(
            masked_fm_dict[key],
            masked_fid_pars_dict[key],
            which_uncertainty=which_uncertainty,
            normalize=True,
            percent_units=True,
        )[:nparams_toplot]

        param_names = list(masked_fid_pars_dict[key].keys())
        cosmo_param_names = list(masked_fid_pars_dict[key].keys())[:nparams_toplot]

        w0wa_idxs = param_names.index('wz'), param_names.index('wa')
        fom_dict[key] = sl.compute_FoM(masked_fm_dict[key], w0wa_idxs=w0wa_idxs)

# compute percent diff btw Gauss and G+SSC, using the respective Gaussian covariance
for probe in probes:
    key_a = f'FM_{probe}_G'
    key_b = f'FM_{probe}_TOT'

    uncert_dict[f'perc_diff_{probe}_G'] = sl.percent_diff(
        uncert_dict[key_b], uncert_dict[key_a]
    )
    fom_dict[f'perc_diff_{probe}_G'] = np.abs(
        sl.percent_diff(fom_dict[key_b], fom_dict[key_a])
    )

    nparams_toplot = 7
    divide_fom_by_10_plt = False if probe in ('WLXC') else divide_fom_by_10

    cases_to_plot = [
        f'FM_{probe}_G',
        f'FM_{probe}_TOT',
        # f'FM_{probe}_GSSCcNG',
        f'perc_diff_{probe}_G',
        #  f'FM_{probe}_{which_ng_cov_suffix}',
        #  f'perc_diff_{probe}_{which_ng_cov_suffix}',
    ]

    # # transform dict. into an array and add the fom
    uncert_array, fom_array = [], []

    for case in cases_to_plot:
        uncert_array.append(uncert_dict[case])
        if divide_fom_by_10 and 'FM' in case and 'WL' not in case:
            fom_dict[case] /= 10
        fom_array.append(fom_dict[case])

    uncert_array = np.asarray(uncert_array)
    fom_array = np.asarray(fom_array)

    uncert_array = np.hstack((uncert_array, fom_array.reshape(-1, 1)))

    perc_diff_probe[probe] = np.append(
        uncert_dict[f'perc_diff_{probe}_G'], fom_dict[f'perc_diff_{probe}_G']
    )

    # label and title stuff
    fom_label = 'FoM/10\nperc_diff' if divide_fom_by_10 else 'FoM'
    param_names_label = (
        param_names_list[:nparams_toplot] + [fom_label]
        if include_fom
        else param_names_list[:nparams_toplot]
    )
    lmax = (
        cfg['ell_binning'][f'ell_max_{probe}']
        if probe in ['WL', 'GC']
        else cfg['ell_binning']['ell_max_3x2pt']
    )
    title = '%s, $\\ell_{\\rm max} = %i$, zbins %s%i, zsteps %s\n%s uncertainties' % (
        probe,
        lmax,
        ep_or_ed,
        zbins,
        len(z_grid),
        which_uncertainty,
    )

    # bar plot
    if include_fom:
        nparams_toplot = 8

    for i, case in enumerate(cases_to_plot):
        cases_to_plot[i] = case
        if 'OneCovariance' in cases_to_plot[i]:
            cases_to_plot[i] = cases_to_plot[i].replace('OneCovariance', 'OneCov')
        if f'PySSC_{probe}_G' in cases_to_plot[i]:
            cases_to_plot[i] = cases_to_plot[i].replace(
                f'PySSC_{probe}_G', f'{probe}_G'
            )

        cases_to_plot[i] = cases_to_plot[i].replace(f'_{probe}', f'')
        cases_to_plot[i] = cases_to_plot[i].replace(f'FM_', f'')
        cases_to_plot[i] = cases_to_plot[i].replace(f'_', f' ')
        cases_to_plot[i] = cases_to_plot[i].replace(f'GSSC', f'G+SSC')
        cases_to_plot[i] = cases_to_plot[i].replace(f'SSCcNG', f'SSC+cNG')

    plot_lib.bar_plot(
        uncert_array[:, :nparams_toplot],
        title,
        cases_to_plot,
        nparams=nparams_toplot,
        param_names_label=param_names_label,
        bar_width=0.13,
        include_fom=include_fom,
        divide_fom_by_10_plt=divide_fom_by_10_plt,
    )

# ! % diff for the 3 probes - careful about the plot title
perc_diff_probe.pop('XC')
plot_lib.bar_plot(
    np.array(list(perc_diff_probe.values())),
    title + r', % diff (G + SSC + cNG)/G',
    (list(perc_diff_probe.keys())),
    nparams=nparams_toplot,
    param_names_label=param_names_label,
    bar_width=0.13,
    include_fom=include_fom,
    divide_fom_by_10_plt=False,
)

# ! Print tables

# if include_fom:
#     nparams_toplot_ref = nparams_toplot
#     nparams_toplot = nparams_toplot_ref + 1
# titles = param_names_list[:nparams_toplot_ref] + ['FoM']

# # for uncert_dict, _, name in zip([uncert_dict, uncert_dict], [fm_dict, fm_dict_vin], ['Davide', 'Vincenzo']):
# print(f"G uncertainties [%]:")
# data = []
# for probe in probes:
#     uncerts = [f'{uncert:.3f}' for uncert in uncert_dict[f'FM_{probe}_G']]
#     fom = f'{fom_dict[f"FM_{probe}_G"]:.2f}'
#     data.append([probe] + uncerts + [fom])
# print(tabulate(data, headers=titles, tablefmt="pretty"))

# print(f"GSSC/G ratio  :")
# data = []
# table = []  # tor tex
# for probe in probes:
#     ratios = [f'{ratio:.3f}' for ratio in uncert_dict[f'ratio_{probe}_G']]
#     fom = f'{fom_dict[f"ratio_{probe}_G"]:.2f}'
#     data.append([probe] + ratios + [fom])
#     table.append(ratios + [fom])
# print(tabulate(data, headers=titles, tablefmt="pretty"))

# print(f"SSC % increase :")
# data = []
# for probe in probes:
#     ratios = [f'{ratio:.3f}' for ratio in uncert_dict[f'perc_diff_{probe}_G']]
#     fom = f'{fom_dict[f"perc_diff_{probe}_G"]:.2f}'
#     data.append([probe] + ratios + [fom])
# print(tabulate(data, headers=titles, tablefmt="pretty"))

# ! quickly compare two selected FMs
# TODO this is misleading, understand better why (comparing GSSC, not perc_diff)

fm_dict_of_dicts = {
    'develop': sl.load_pickle(f'{fm_folder}/FM_dict_sigma2b_bigchanges.pickle'),
    'levin': sl.load_pickle(f'{fm_folder}/FM_dict_sigma2b_levindav.pickle'),
    # 'current': fm_dict,
}


labels = list(fm_dict_of_dicts.keys())
fm_dict_list = list(fm_dict_of_dicts.values())
keys_toplot_in = ['FM_WL_TOT', 'FM_GC_TOT', 'FM_XC_TOT', 'FM_3x2pt_TOT']
# keys_toplot = 'all'
colors = [
    'tab:blue',
    'tab:green',
    'tab:orange',
    'tab:red',
    'tab:cyan',
    'tab:grey',
    'tab:olive',
    'tab:purple',
]

reference = 'first_key'
nparams_toplot_in = 8
normalize_by_gauss = True

sl.compare_fm_constraints(
    *fm_dict_list,
    labels=labels,
    keys_toplot_in=keys_toplot_in,
    normalize_by_gauss=True,
    which_uncertainty='marginal',
    reference=reference,
    colors=colors,
    abs_FoM=True,
    save_fig=False,
    fig_path='/home/davide/Scrivania/',
)

assert False, 'stop here'

fisher_matrices = (
    fm_dict_of_dicts['SB_hm_simpker']['FM_3x2pt_GSSC'],
    fm_dict_of_dicts['SB_KEapp_hm_simpker']['FM_3x2pt_GSSC'],
    fm_dict_of_dicts['OC_simpker']['FM_3x2pt_GSSC'],
    fm_dict_of_dicts['current']['FM_3x2pt_GSSC'],
)
fiducials = list(fm_dict_of_dicts['SB_hm_simpker']['fiducial_values_dict'].values())
# fiducials = (
# fm_dict_of_dicts['SB_hm_simpker']['fiducial_values_dict'].values(),
# fm_dict_of_dicts['SB_KEapp_hm_simpker']['fiducial_values_dict'].values(),
# fm_dict_of_dicts['OC_simpker']['fiducial_values_dict'].values(),
# fm_dict_of_dicts['current']['fiducial_values_dict'].values(),
# )
param_names_list = list(
    fm_dict_of_dicts['SB_hm_simpker']['fiducial_values_dict'].keys()
)
param_names_labels_toplot = param_names_list[:8]
plot_lib.triangle_plot(
    fisher_matrices,
    fiducials,
    title,
    labels,
    param_names_list,
    param_names_labels_toplot,
    param_names_labels_tex=None,
    rotate_param_labels=False,
    contour_colors=None,
    line_colors=None,
)


print(
    'Finished in {:.2f} minutes'.format((time.perf_counter() - script_start_time) / 60)
)