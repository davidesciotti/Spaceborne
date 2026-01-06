"""OneCovariance Interface Module

This module provides an interface to the OneCovariance (OC) covariance matrix
calculator.
It handles configuration, execution, and post-processing of covariance calculations for
cosmic shear, galaxy-galaxy lensing, and galaxy clustering.

Key Features:
- Configures and executes OneCovariance
- Manages IO
- Reshapes covariance matrices between different formats
- Optimizes ell binning to match target specifications
- Supports different precision settings for calculations
- Handles Gaussian, non-Gaussian, and SSC covariance terms


"""

import configparser
import os
import subprocess
from collections import defaultdict
from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize_scalar

from spaceborne import constants as const
from spaceborne import cov_dict as cd
from spaceborne import sb_lib as sl

_UNSET = object()


def compare_sb_cov_to_oc_list(
    cov_rs_obj,
    cov_oc_dict_6d: dict,
    probe_sb: str,
    term: str,
    ind_auto: np.ndarray,
    ind_cross: np.ndarray,
    zpairs_auto: int,
    zpairs_cross: int,
    scale_bins: int,
    title: str | None = None,
):
    # gt is gm in OneCov
    probe_oc = probe_sb.replace('gt', 'gm')

    # get probe names, ind and zpairs for 2D conversion
    probe_ab, probe_cd = sl.split_probe_name(probe_sb)  # TODO space?
    probe_ab_ix, probe_cd_ix = (
        const.RS_PROBE_NAME_TO_IX_DICT_SHORT[probe_ab],
        const.RS_PROBE_NAME_TO_IX_DICT_SHORT[probe_cd],
    )
    zpairs_ab = zpairs_cross if probe_ab_ix == 1 else zpairs_auto
    zpairs_cd = zpairs_cross if probe_cd_ix == 1 else zpairs_auto
    ind_ab = ind_cross if probe_ab_ix == 1 else ind_auto
    ind_cd = ind_cross if probe_cd_ix == 1 else ind_auto

    # get 6D covs
    cov_sb_6d = getattr(cov_rs_obj, f'cov_{probe_sb}_{term}_6d')

    # for cov OC, some blocks may be transposed
    try:
        cov_oc_6d = cov_oc_dict_6d[f'{probe_oc}_{term}']
    except KeyError:
        _probe_sb_inv = probe_cd + probe_ab
        _probe_oc_inv = _probe_sb_inv.replace('gt', 'gm')
        cov_oc_6d = cov_oc_dict_6d[f'{_probe_oc_inv}_{term}'].transpose(
            1, 0, 4, 5, 2, 3
        )

    # if both covs are null, exit the function
    if np.all(cov_sb_6d == 0) and np.all(cov_oc_6d == 0):
        print(f'OC and SB covs for {term = } {probe_sb = } are both identically 0')
        return

    # convert to 2D to compare
    cov_sb_4d = sl.cov_6D_to_4D_blocks(
        cov_sb_6d, scale_bins, zpairs_ab, zpairs_cd, ind_ab, ind_cd
    )
    cov_oc_4d = sl.cov_6D_to_4D_blocks(
        cov_oc_6d, scale_bins, zpairs_ab, zpairs_cd, ind_ab, ind_cd
    )

    cov_sb_2d = sl.cov_4D_to_2D(cov_sb_4d, block_index='zpair', optimize=True)
    cov_oc_2d = sl.cov_4D_to_2D(cov_oc_4d, block_index='zpair', optimize=True)

    sl.compare_arrays(
        cov_sb_2d,
        cov_oc_2d,
        'SB',
        'OC',
        log_array=True,
        log_diff=True,
        abs_val=True,
        plot_diff_threshold=10,
        title=title,
    )

    fig, axs = plt.subplots(
        2,
        2,
        figsize=(15, 6),
        sharex='col',
        height_ratios=[2, 1],
        gridspec_kw={'hspace': 0, 'wspace': 0.3},
    )

    # flatten to (2,2) shape
    axs = axs.reshape(2, 2)

    sl.compare_funcs(
        None,
        {'SB diag': np.abs(np.diag(cov_sb_2d)), 'OC diag': np.abs(np.diag(cov_oc_2d))},
        logscale_y=[True, False],
        title=title,
        ylim_diff=[-100, 100],
        ax=axs[:, 0],
    )

    sl.compare_funcs(
        None,
        {
            'SB flat': np.abs(cov_sb_2d).flatten(),
            'OC flat': np.abs(cov_oc_2d).flatten(),
        },
        logscale_y=[True, False],
        title=title,
        ylim_diff=[-100, 100],
        ax=axs[:, 1],
    )


def print_cfg_onecov_ini(cfg_onecov_ini):
    """This is necessary since cfg_onecov_ini is not simply a dict (because I first
    load) an example .ini file..."""

    for section in cfg_onecov_ini.sections():
        print(f'[{section}]')
        for key, value in cfg_onecov_ini.items(section):
            print(f'{key} = {value}')
        print()  # Add a blank line for readability between sections


def process_cov_from_list_file(
    cov_dict, oc_output_covlist_fname, zbins, obs_space, df_chunk_size=5_000_000
):
    import re

    import pandas as pd

    # read df column names
    with open(oc_output_covlist_fname) as file:
        header = (
            file.readline().strip()
        )  # Read the first line and strip newline characters
    header_list = re.split(
        '\t', header.strip().replace('\t\t', '\t').replace('\t\t', '\t')
    )
    column_names = header_list

    # Determine scale type
    if 'theta1' in column_names:
        scale_ix_name = 'theta'
    elif 'ell1' in column_names:
        scale_ix_name = 'ell'
    elif 'n1' in column_names:
        scale_ix_name = 'n'
    else:
        raise ValueError('OneCov column names not recognised')

    usecols = ['#obs', 'tomoi', f'{scale_ix_name}1']

    # partial (much quicker) import of the .list file, to get info about thetas, probes
    # and to perform checks on the tomo bin idxs
    data = pd.read_csv(oc_output_covlist_fname, usecols=usecols, sep=r'\s+')

    scales_oc_load = data[f'{scale_ix_name}1'].unique()
    cov_scale_indices = {scale: idx for idx, scale in enumerate(scales_oc_load)}
    nbx_oc = len(scales_oc_load)  # 'nbx' = nbt or nbl

    # check tomo idxs: SB tomographic indices start from 0
    tomoi_oc_load = data['tomoi'].unique()
    subtract_one_from_z_ix = min(tomoi_oc_load) == 1

    # Setup probe translation
    if obs_space == 'harmonic':
        valid_probes = const.HS_DIAG_PROBES_OC
        probe_transl_dict = const.HS_DIAG_PROBES_OC_TO_SB
        assert scale_ix_name == 'ell', 'scale_ix_name must be "ell" for harmonic space'
    elif obs_space == 'real':
        valid_probes = const.RS_DIAG_PROBES_OC
        probe_transl_dict = const.RS_DIAG_PROBES_OC_TO_SB
        assert scale_ix_name == 'theta', 'scale_ix_name must be "theta" for real space'
    elif obs_space == 'cosebis':
        valid_probes = const.CS_DIAG_PROBES_OC
        probe_transl_dict = const.CS_DIAG_PROBES_OC_TO_SB
        assert scale_ix_name == 'n', 'scale_ix_name must be "n" for cosebis space'
    else:
        raise ValueError('obs_space must be either "harmonic", "real" or "cosebis"')

    print(f'Loading OneCovariance output from {oc_output_covlist_fname} file...')

    # Initialize arrays
    # in this case it's necessary to preallocate the 6D shapes, since the arrays
    # are filled partially at each iteration of the for loop
    temp_cov_arrays = defaultdict(
        lambda: defaultdict(
            lambda: np.zeros((nbx_oc, nbx_oc, zbins, zbins, zbins, zbins))
        )
    )

    # Column mapping for covariance terms
    term_columns = {
        'sva': 'covg sva',
        'mix': 'covg mix',
        'sn': 'covg sn',
        'ssc': 'covssc',
        'cng': 'covng',
    }

    for df_chunk in pd.read_csv(
        oc_output_covlist_fname,
        sep=r'\s+',
        names=column_names,
        skiprows=1,
        chunksize=df_chunk_size,
    ):
        for probe_abcd_oc, subdf in df_chunk.groupby('#obs'):
            probe_ab_oc, probe_cd_oc = sl.split_probe_name(
                probe_abcd_oc, space=None, valid_probes=valid_probes
            )
            probe_ab, probe_cd = (
                probe_transl_dict[probe_ab_oc],
                probe_transl_dict[probe_cd_oc],
            )
            probe_2tpl = (probe_ab, probe_cd)

            # Pre-compute indices once
            theta1_idx = subdf[f'{scale_ix_name}1'].map(cov_scale_indices).values
            theta2_idx = subdf[f'{scale_ix_name}2'].map(cov_scale_indices).values

            if subtract_one_from_z_ix:
                z_ixs = subdf[['tomoi', 'tomoj', 'tomok', 'tomol']].sub(1).values
            else:
                z_ixs = subdf[['tomoi', 'tomoj', 'tomok', 'tomol']].values

            index_tuple = (
                theta1_idx,
                theta2_idx,
                z_ixs[:, 0],
                z_ixs[:, 1],
                z_ixs[:, 2],
                z_ixs[:, 3],
            )

            # Assign individual terms
            for term, col in term_columns.items():
                temp_cov_arrays[term][probe_2tpl][index_tuple] = subdf[col].values

            # Compute 'g' term as sum of sva, sn and mix
            temp_cov_arrays['g'][probe_2tpl][index_tuple] = (
                subdf['covg sva'].values
                + subdf['covg sn'].values
                + subdf['covg mix'].values
            )

            temp_cov_arrays['tot'][probe_2tpl][index_tuple] = (
                temp_cov_arrays['g'][probe_2tpl][index_tuple]
                + temp_cov_arrays['ssc'][probe_2tpl][index_tuple]
                + temp_cov_arrays['cng'][probe_2tpl][index_tuple]
            )

    # store in cov_dict
    for term in cov_dict:
        for probe_2tpl in cov_dict[term]:
            if probe_2tpl != '3x2pt':
                cov_dict[term][probe_2tpl]['6d'] = temp_cov_arrays[term][probe_2tpl]

    print('...done')


class OneCovarianceInterface:
    def __init__(self, cfg, pvt_cfg, do_g, do_ssc, do_cng):
        """Initializes the OneCovarianceInterface class with the provided configuration
        and private configuration
        dictionaries.

        Args:
            cfg (dict): The configuration dictionary.
            pvt_cfg (dict): The private specifications dictionary.
            do_ssc (bools): Whether to compute the SSC term.
            do_cng (bool): Whether to compute the connected non-Gaussian
            covariance term.

        Attributes:
            cfg (dict): The configuration dictionary.
            oc_cfg (dict): The OneCovariance configuration dictionary.
            pvt_cfg (dict): The private specifications dictionary.
            zbins (int): The number of redshift bins.
            nbl_3x2pt (int): The number of ell bins for the 3x2pt analysis.
            compute_ssc (bool): Whether to compute the super-sample
            covariance (SSC) term.
            compute_cng (bool): Whether to compute the connected non-Gaussian
            covariance (cNG) term.
            oc_path (str): The path to the OneCovariance output directory.
            path_to_oc_executable (str): The path to the OneCovariance executable.
            path_to_config_oc_ini (str): The path to the OneCovariance configuration
            INI file.

        """
        self.cfg = cfg
        self.oc_cfg = self.cfg['OneCovariance']
        self.pvt_cfg = pvt_cfg
        self.n_probes = cfg['covariance']['n_probes']
        self.nbl_3x2pt = pvt_cfg['nbl_3x2pt']
        self.zbins = pvt_cfg['zbins']
        self.ind = pvt_cfg['ind']
        self.probe_ordering = pvt_cfg['probe_ordering']
        self.GL_OR_LG = pvt_cfg['GL_OR_LG']

        # set which cov terms to compute from cfg file
        self.compute_g = do_g
        self.compute_ssc = do_ssc
        self.compute_cng = do_cng

        self.obs_space = self.cfg['probe_selection']['space']

        if self.obs_space == 'harmonic':
            prefix = 'hs'
        elif self.obs_space == 'real':
            prefix = 'rs'
        elif self.obs_space == 'cosebis':
            prefix = 'cs'
        else:
            raise ValueError('self.obs_space must be "harmonic", "real" or "cosebis"')

        # instantiate cov dict with the required terms and probe combinations
        self.req_terms = pvt_cfg['req_terms']
        self.req_probe_combs_2d = pvt_cfg[f'req_probe_combs_{prefix}_2d']
        self.nonreq_probe_combs = pvt_cfg[f'nonreq_probe_combs_{prefix}']
        dims = ['6d', '4d', '2d']

        _req_probe_combs_2d = [
            sl.split_probe_name(probe, space=self.obs_space)
            for probe in self.req_probe_combs_2d
        ]
        _req_probe_combs_2d.append('3x2pt')
        self.cov_dict = cd.create_cov_dict(
            self.req_terms, _req_probe_combs_2d, dims=dims
        )

        # paths and filenems
        self.path_to_oc_env = cfg['OneCovariance']['path_to_oc_env']
        self.path_to_oc_executable = cfg['OneCovariance']['path_to_oc_executable']

        self.oc_path: str = _UNSET
        self.z_grid_trisp_sb: np.ndarray = _UNSET
        self.path_to_config_oc_ini: str = _UNSET
        self.ells_sb: np.ndarray = _UNSET
        self.cov_3x2pt_sva_10d: np.ndarray = _UNSET
        self.cov_3x2pt_sn_10d: np.ndarray = _UNSET
        self.cov_3x2pt_mix_10d: np.ndarray = _UNSET
        self.cov_3x2pt_g_10d: np.ndarray = _UNSET
        self.cov_3x2pt_ssc_10d: np.ndarray = _UNSET
        self.cov_3x2pt_cng_10d: np.ndarray = _UNSET
        self.cov_3x2pt_tot_10d: np.ndarray = _UNSET

    def build_save_oc_ini(self, ascii_filenames_dict, h, print_ini=True):
        # this is just to preserve case sensitivity
        class CaseConfigParser(configparser.ConfigParser):
            def optionxform(self, optionstr):
                return optionstr

        cl_ll_oc_filename = ascii_filenames_dict['cl_ll_ascii_filename']
        cl_gl_oc_filename = ascii_filenames_dict['cl_gl_ascii_filename']
        cl_gg_oc_filename = ascii_filenames_dict['cl_gg_ascii_filename']
        nz_src_filename_ascii = ascii_filenames_dict['nz_src_ascii_filename']
        nz_lns_filename_ascii = ascii_filenames_dict['nz_lns_ascii_filename']

        # Read the .ini file selected in cfg
        cfg_oc_ini = CaseConfigParser()
        # cfg_oc_ini.read(self.path_to_oc_ini)

        # set useful lists
        mult_shear_bias_list = np.array(self.cfg['C_ell']['mult_shear_bias'])
        n_eff_clust_list = self.cfg['nz']['ngal_lenses']
        n_eff_lensing_list = self.cfg['nz']['ngal_sources']
        ellipticity_dispersion_list = self.cfg['covariance']['sigma_eps_i']

        # set headers
        cfg_oc_ini['covariance terms'] = {}
        cfg_oc_ini['observables'] = {}
        cfg_oc_ini['output settings'] = {}
        cfg_oc_ini['covELLspace settings'] = {}
        cfg_oc_ini['covTHETAspace settings'] = {}  # For real space case
        cfg_oc_ini['covCOSEBI settings'] = {}  # For COSEBIs case
        cfg_oc_ini['survey specs'] = {}
        cfg_oc_ini['redshift'] = {}
        cfg_oc_ini['cosmo'] = {}
        cfg_oc_ini['bias'] = {}
        cfg_oc_ini['IA'] = {}
        cfg_oc_ini['hod'] = {}
        cfg_oc_ini['halomodel evaluation'] = {}
        cfg_oc_ini['powspec evaluation'] = {}
        cfg_oc_ini['trispec evaluation'] = {}
        cfg_oc_ini['tabulated inputs files'] = {}
        cfg_oc_ini['misc'] = {}

        # ! [covariance terms]
        cfg_oc_ini['covariance terms']['gauss'] = str(True)
        cfg_oc_ini['covariance terms']['split_gauss'] = str(True)
        cfg_oc_ini['covariance terms']['nongauss'] = str(self.compute_cng)
        cfg_oc_ini['covariance terms']['ssc'] = str(self.compute_ssc)

        # ! [observables]
        if self.obs_space == 'harmonic':
            est_shear = 'C_ell'
            est_ggl = 'C_ell'
            est_clust = 'C_ell'

            cosmic_shear = self.cfg['probe_selection']['LL']
            ggl = self.cfg['probe_selection']['GL']
            clustering = self.cfg['probe_selection']['GG']

        elif self.obs_space == 'real':
            est_shear = 'xi_pm'
            est_ggl = 'gamma_t'
            est_clust = 'w'

            cosmic_shear = (
                self.cfg['probe_selection']['xip'] or self.cfg['probe_selection']['xim']
            )
            ggl = self.cfg['probe_selection']['gt']
            clustering = self.cfg['probe_selection']['w']

        elif self.obs_space == 'cosebis':
            est_shear = est_ggl = est_clust = 'cosebi'

            cosmic_shear = (
                self.cfg['probe_selection']['En'] or self.cfg['probe_selection']['Bn']
            )
            ggl = self.cfg['probe_selection']['Psigl']
            clustering = self.cfg['probe_selection']['Psigg']
        else:
            raise ValueError('self.which_obs must he "harmonic" or "real"')

        cfg_oc_ini['observables']['cosmic_shear'] = str(cosmic_shear)
        cfg_oc_ini['observables']['est_shear'] = est_shear
        cfg_oc_ini['observables']['ggl'] = str(ggl)
        cfg_oc_ini['observables']['est_ggl'] = est_ggl
        cfg_oc_ini['observables']['clustering'] = str(clustering)
        cfg_oc_ini['observables']['est_clust'] = est_clust
        cfg_oc_ini['observables']['cstellar_mf'] = str(False)
        cfg_oc_ini['observables']['cross_terms'] = str(True)
        cfg_oc_ini['observables']['unbiased_clustering'] = str(False)

        # ! [output settings]
        self.cov_oc_fname = self.cfg['OneCovariance']['oc_output_filename']
        cfg_oc_ini['output settings']['directory'] = self.oc_path
        cfg_oc_ini['output settings']['file'] = ', '.join(
            [f'{self.cov_oc_fname}_list.dat', f'{self.cov_oc_fname}_matrix.mat']
        )
        cfg_oc_ini['output settings']['style'] = ', '.join(['list', 'matrix'])
        cfg_oc_ini['output settings']['list_style_spatial_first'] = str(True)
        cfg_oc_ini['output settings']['corrmatrix_plot'] = (
            f'{self.cov_oc_fname}_corrplot.pdf'
        )
        cfg_oc_ini['output settings']['save_configs'] = 'save_configs.ini'
        cfg_oc_ini['output settings']['save_Cells'] = str(True)
        cfg_oc_ini['output settings']['save_trispectra'] = str(False)
        cfg_oc_ini['output settings']['save_alms'] = str(True)
        cfg_oc_ini['output settings']['use_tex'] = str(False)

        # ! [covELLspace settings]
        np.testing.assert_allclose(
            np.diff(self.z_grid_trisp_sb)[0],
            np.diff(self.z_grid_trisp_sb),
            atol=0,
            rtol=1e-7,
            err_msg='The redshift grid is not uniform.',
        )
        delta_z = np.diff(self.z_grid_trisp_sb)[0]

        ell_binning_type = self.cfg['binning']['binning_type']
        if self.cfg['binning']['binning_type'] == 'ref_cut':
            ell_binning_type = 'log'

        # settings common to both observables
        cfg_oc_ini['covELLspace settings']['limber'] = str(True)
        cfg_oc_ini['covELLspace settings']['nglimber'] = str(True)
        cfg_oc_ini['covELLspace settings']['delta_z'] = str(delta_z)
        cfg_oc_ini['covELLspace settings']['tri_delta_z'] = str(0.5)
        # * PRECISION PARAMETER MODIFIED IN THE PAST (500 -> 1000)
        cfg_oc_ini['covELLspace settings']['integration_steps'] = str(500)
        cfg_oc_ini['covELLspace settings']['nz_interpolation_polynom_order'] = str(1)
        cfg_oc_ini['covELLspace settings']['mult_shear_bias'] = ', '.join(
            map(str, mult_shear_bias_list)
        )
        cfg_oc_ini['covELLspace settings']['ell_type_clustering'] = ell_binning_type
        cfg_oc_ini['covELLspace settings']['ell_type_lensing'] = ell_binning_type

        # settings specific to both observables
        if self.obs_space == 'harmonic':
            cfg_oc_ini['covELLspace settings']['ell_min'] = str(
                self.pvt_cfg['ell_min_3x2pt']
            )
            cfg_oc_ini['covELLspace settings']['ell_min_lensing'] = str(
                self.pvt_cfg['ell_min_3x2pt']
            )
            cfg_oc_ini['covELLspace settings']['ell_min_clustering'] = str(
                self.pvt_cfg['ell_min_3x2pt']
            )
            cfg_oc_ini['covELLspace settings']['ell_bins'] = str(
                self.pvt_cfg['nbl_3x2pt']
            )
            cfg_oc_ini['covELLspace settings']['ell_bins_lensing'] = str(
                self.pvt_cfg['nbl_3x2pt']
            )
            cfg_oc_ini['covELLspace settings']['ell_bins_clustering'] = str(
                self.pvt_cfg['nbl_3x2pt']
            )

            # find best ell_max for OC, since it uses a slightly different recipe
            self.find_optimal_ellmax_oc(target_ell_array=self.ells_sb)
            cfg_oc_ini['covELLspace settings']['ell_max'] = str(self.optimal_ellmax)
            cfg_oc_ini['covELLspace settings']['ell_max_lensing'] = str(
                self.optimal_ellmax
            )
            cfg_oc_ini['covELLspace settings']['ell_max_clustering'] = str(
                self.optimal_ellmax
            )

        elif self.obs_space == 'real':
            cfg_oc_ini['covELLspace settings']['ell_min'] = str(
                self.cfg['precision']['ell_min_rs']
            )
            cfg_oc_ini['covELLspace settings']['ell_max'] = str(
                self.cfg['precision']['ell_max_rs']
            )
            cfg_oc_ini['covELLspace settings']['ell_bins'] = str(
                self.cfg['precision']['ell_bins_rs']
            )
            cfg_oc_ini['covELLspace settings']['ell_type'] = 'log'

        # ! [survey specs]
        # commented out to avoid loading mask file by accident
        cfg_oc_ini['survey specs']['mask_directory'] = str(
            self.cfg['mask']['mask_path']
        )  # TODO test this!!
        cfg_oc_ini['survey specs']['survey_area_lensing_in_deg2'] = str(
            self.cfg['mask']['survey_area_deg2']
        )
        cfg_oc_ini['survey specs']['survey_area_ggl_in_deg2'] = str(
            self.cfg['mask']['survey_area_deg2']
        )
        cfg_oc_ini['survey specs']['survey_area_clust_in_deg2'] = str(
            self.cfg['mask']['survey_area_deg2']
        )
        cfg_oc_ini['survey specs']['n_eff_clust'] = ', '.join(
            map(str, n_eff_clust_list)
        )
        cfg_oc_ini['survey specs']['n_eff_lensing'] = ', '.join(
            map(str, n_eff_lensing_list)
        )
        cfg_oc_ini['survey specs']['ellipticity_dispersion'] = ', '.join(
            map(str, ellipticity_dispersion_list)
        )

        # ! [redshift]
        cfg_oc_ini['redshift']['z_directory'] = self.oc_path
        # TODO re-check that the OC documentation is correct
        cfg_oc_ini['redshift']['zclust_file'] = nz_lns_filename_ascii
        cfg_oc_ini['redshift']['zlens_file'] = nz_src_filename_ascii
        cfg_oc_ini['redshift']['value_loc_in_clustbin'] = 'mid'
        cfg_oc_ini['redshift']['value_loc_in_lensbin'] = 'mid'

        # ! [cosmo]
        cfg_oc_ini['cosmo']['h'] = str(self.cfg['cosmology']['h'])
        cfg_oc_ini['cosmo']['ns'] = str(self.cfg['cosmology']['ns'])
        cfg_oc_ini['cosmo']['omega_m'] = str(self.cfg['cosmology']['Om'])
        cfg_oc_ini['cosmo']['omega_b'] = str(self.cfg['cosmology']['Ob'])
        cfg_oc_ini['cosmo']['omega_de'] = str(self.cfg['cosmology']['ODE'])
        cfg_oc_ini['cosmo']['sigma8'] = str(self.cfg['cosmology']['s8'])
        cfg_oc_ini['cosmo']['w0'] = str(self.cfg['cosmology']['wz'])
        cfg_oc_ini['cosmo']['wa'] = str(self.cfg['cosmology']['wa'])
        cfg_oc_ini['cosmo']['neff'] = str(self.cfg['cosmology']['N_eff'])
        cfg_oc_ini['cosmo']['m_nu'] = str(self.cfg['cosmology']['m_nu'])
        cfg_oc_ini['cosmo']['tcmb0'] = str(2.7255)

        # ! [bias]
        if self.cfg['covariance']['which_b1g_in_resp'] == 'from_input':
            gal_bias_ascii_filename = ascii_filenames_dict['gal_bias_ascii_filename']
            cfg_oc_ini['bias']['bias_files'] = gal_bias_ascii_filename

        # ! [IA]
        cfg_oc_ini['IA']['A_IA'] = str(self.cfg['intrinsic_alignment']['Aia'])
        cfg_oc_ini['IA']['eta_IA'] = str(self.cfg['intrinsic_alignment']['eIA'])
        cfg_oc_ini['IA']['z_pivot_IA'] = str(
            self.cfg['intrinsic_alignment']['z_pivot_IA']
        )

        # ! [hod]
        cfg_oc_ini['hod']['model_mor_cen'] = 'double_powerlaw'
        cfg_oc_ini['hod']['model_mor_sat'] = 'double_powerlaw'
        cfg_oc_ini['hod']['dpow_logm0_cen'] = str(10.51)
        cfg_oc_ini['hod']['dpow_logm1_cen'] = str(11.38)
        cfg_oc_ini['hod']['dpow_a_cen'] = str(7.096)
        cfg_oc_ini['hod']['dpow_b_cen'] = str(0.2)
        cfg_oc_ini['hod']['dpow_norm_cen'] = str(1.0)
        cfg_oc_ini['hod']['dpow_norm_sat'] = str(0.56)
        cfg_oc_ini['hod']['model_scatter_cen'] = 'lognormal'
        cfg_oc_ini['hod']['model_scatter_sat'] = 'modschechter'
        cfg_oc_ini['hod']['logn_sigma_c_cen'] = str(0.35)
        cfg_oc_ini['hod']['modsch_logmref_sat'] = str(13.0)
        cfg_oc_ini['hod']['modsch_alpha_s_sat'] = str(-0.858)
        cfg_oc_ini['hod']['modsch_b_sat'] = ', '.join([str(-0.024), str(1.149)])

        # ! [covTHETAspace settings]
        if self.obs_space == 'real':
            cfg_oc_ini['covTHETAspace settings']['theta_min_clustering'] = str(
                self.cfg['binning']['theta_min_arcmin']
            )
            cfg_oc_ini['covTHETAspace settings']['theta_max_clustering'] = str(
                self.cfg['binning']['theta_max_arcmin']
            )
            cfg_oc_ini['covTHETAspace settings']['theta_bins_clustering'] = str(
                self.cfg['binning']['theta_bins']
            )
            cfg_oc_ini['covTHETAspace settings']['theta_type_clustering'] = 'lin'
            cfg_oc_ini['covTHETAspace settings']['theta_min_lensing'] = str(
                self.cfg['binning']['theta_min_arcmin']
            )
            cfg_oc_ini['covTHETAspace settings']['theta_max_lensing'] = str(
                self.cfg['binning']['theta_max_arcmin']
            )
            cfg_oc_ini['covTHETAspace settings']['theta_bins_lensing'] = str(
                self.cfg['binning']['theta_bins']
            )
            cfg_oc_ini['covTHETAspace settings']['theta_type_lensing'] = 'lin'

            cfg_oc_ini['covTHETAspace settings']['theta_min'] = str(
                self.cfg['binning']['theta_min_arcmin']
            )
            cfg_oc_ini['covTHETAspace settings']['theta_max'] = str(
                self.cfg['binning']['theta_max_arcmin']
            )
            cfg_oc_ini['covTHETAspace settings']['theta_type'] = 'lin'

            cfg_oc_ini['covTHETAspace settings']['xi_pp'] = str(True)
            cfg_oc_ini['covTHETAspace settings']['xi_mm'] = str(True)
            cfg_oc_ini['covTHETAspace settings']['theta_accuracy'] = str(1e-3)
            cfg_oc_ini['covTHETAspace settings']['integration_intervals'] = str(40)

        # ! [covTHETAspace settings]
        if self.obs_space == 'cosebis':
            for _probe in ['', '_clustering', '_lensing']:
                cfg_oc_ini['covCOSEBI settings'][f'En_modes{_probe}'] = str(
                    self.cfg['precision']['n_modes_cosebis']
                )
            cfg_oc_ini['covCOSEBI settings']['En_accuracy'] = str(1e-4)

            for _probe in ['', '_clustering', '_lensing']:
                for _type in ['_min', '_max']:
                    # print(f'theta{_type}{_probe}', f'theta{_type}_arcmin_cosebis')
                    cfg_oc_ini['covCOSEBI settings'][f'theta{_type}{_probe}'] = str(
                        self.cfg['precision'][f'theta{_type}_arcmin_cosebis']
                    )

            cfg_oc_ini['covCOSEBI settings']['wn_style'] = 'log'
            cfg_oc_ini['covCOSEBI settings']['wn_accuracy'] = str(1e-6)
            cfg_oc_ini['covCOSEBI settings']['dimensionless_cosebi'] = str(False)
                    

        # ! [halomodel evaluation]
        if ('Tinker10' not in self.cfg['halo_model']['mass_function']) or (
            'Tinker10' not in self.cfg['halo_model']['halo_bias']
        ):
            raise ValueError(
                'Only Tinker10 mass function and halo bias are supported '
                f'by OneCovariance. Got {self.cfg['halo_model']['mass_function']=}'
                f'and {self.cfg['halo_model']['halo_bias']=} instead'
            )

        # * PRECISION PARAMETER MODIFIED IN THE PAST (900 -> 1500)
        cfg_oc_ini['halomodel evaluation']['m_bins'] = str(900)
        cfg_oc_ini['halomodel evaluation']['log10m_min'] = str(6)
        cfg_oc_ini['halomodel evaluation']['log10m_max'] = str(18)
        cfg_oc_ini['halomodel evaluation']['hmf_model'] = 'Tinker10'
        cfg_oc_ini['halomodel evaluation']['mdef_model'] = 'SOMean'
        cfg_oc_ini['halomodel evaluation']['mdef_params'] = ', '.join(
            ['overdensity', str(200)]
        )

        cfg_oc_ini['halomodel evaluation']['disable_mass_conversion'] = str(True)
        cfg_oc_ini['halomodel evaluation']['delta_c'] = str(1.686)
        cfg_oc_ini['halomodel evaluation']['transfer_model'] = 'CAMB'
        cfg_oc_ini['halomodel evaluation']['small_k_damping_for1h'] = 'damped'

        # ! [powspec evaluation]
        cfg_oc_ini['powspec evaluation']['non_linear_model'] = str(
            self.cfg['extra_parameters']['camb']['halofit_version']
        )
        cfg_oc_ini['powspec evaluation']['HMCode_logT_AGN'] = str(
            self.cfg['extra_parameters']['camb']['HMCode_logT_AGN']
        )
        cfg_oc_ini['powspec evaluation']['log10k_min'] = str(
            self.cfg['covariance']['log10_k_min'] * h
        )
        cfg_oc_ini['powspec evaluation']['log10k_max'] = str(
            self.cfg['covariance']['log10_k_max'] * h
        )
        cfg_oc_ini['powspec evaluation']['log10k_bins'] = str(
            self.cfg['covariance']['k_steps']
        )

        # ! [trispec evaluation]
        cfg_oc_ini['trispec evaluation']['log10k_min'] = str(
            self.cfg['covariance']['log10_k_min'] * h
        )
        cfg_oc_ini['trispec evaluation']['log10k_max'] = str(
            self.cfg['covariance']['log10_k_max'] * h
        )
        cfg_oc_ini['trispec evaluation']['log10k_bins'] = str(
            self.cfg['covariance']['k_steps']
        )
        cfg_oc_ini['trispec evaluation']['matter_klim'] = str(0.001)
        cfg_oc_ini['trispec evaluation']['matter_mulim'] = str(0.001)
        cfg_oc_ini['trispec evaluation']['small_k_damping_for1h'] = 'damped'
        cfg_oc_ini['trispec evaluation']['lower_calc_limit'] = str(1e-200)

        # ! [tabulated inputs files]
        cfg_oc_ini['tabulated inputs files']['Cell_directory'] = self.oc_path
        cfg_oc_ini['tabulated inputs files']['Cmm_file'] = f'{cl_ll_oc_filename}.ascii'
        cfg_oc_ini['tabulated inputs files']['Cgm_file'] = f'{cl_gl_oc_filename}.ascii'
        cfg_oc_ini['tabulated inputs files']['Cgg_file'] = f'{cl_gg_oc_filename}.ascii'

        # ! [misc]
        cfg_oc_ini['misc']['num_cores'] = str(self.cfg['misc']['num_threads'])

        # TODO integration_steps is similar to len(z_grid), but OC works in
        # TODO log space
        # TODO + it would signficantly slow down the code if using SB values
        # TODO (e.g. 3000)
        # TODO so I leave it like this for the time being

        # print the updated ini
        if print_ini:
            print_cfg_onecov_ini(cfg_oc_ini)

        # Save the updated configuration to a new .ini file
        with open(f'{self.oc_path}/input_configs.ini', 'w') as configfile:
            cfg_oc_ini.write(configfile)

        # store in self for good measure
        self.cfg_onecov_ini = cfg_oc_ini

    def call_oc_from_bash(self) -> None:
        """This function runs OneCovariance"""
        try:
            # Set MPLBACKEND to prevent display errors in subprocess
            env = os.environ.copy()
            env['MPLBACKEND'] = 'Agg'

            subprocess.run(
                [
                    self.path_to_oc_env,
                    self.path_to_oc_executable,
                    self.path_to_config_oc_ini,
                ],
                check=True,
                capture_output=False,
                text=True,
                env=env,
            )
        except subprocess.CalledProcessError as e:
            print('OneCovariance failed with error:')
            raise e

    def call_oc_from_class(self):
        """This interface was originally created by Robert Reischke.
        Pros:
            - Streamlines the call to the code by instantiating and calling the
            CovELLSpace class directly
            (as done in OneCovariance main file)

            - Returns outputs which are in a more similar format as Spaceborne
            - Returns outputs with more significant digits
        Cons:
            - Less maintainable than the bash call
        """
        import sys

        sys.path.append(os.path.dirname(self.path_to_oc_executable))
        import platform

        from onecov.cov_ell_space import CovELLSpace
        from onecov.cov_input import FileInput, Input

        if len(platform.mac_ver()[0]) > 0 and (
            platform.processor() == 'arm'
            or int(platform.mac_ver()[0][: (platform.mac_ver()[0]).find('.')]) > 13
        ):
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        from onecov.cov_setup import Setup

        print('READING OneCovariance INPUT')
        print('#############')

        inp = Input()

        covterms, observables, output, cosmo, bias, iA, hod, survey_params, prec = (
            inp.read_input(f'{self.oc_path}/input_configs.ini')
        )
        fileinp = FileInput(bias)
        read_in_tables = fileinp.read_input(f'{self.oc_path}/input_configs.ini')
        _setup = Setup(
            cosmo_dict=cosmo,
            bias_dict=bias,
            survey_params_dict=survey_params,
            prec=prec,
            read_in_tables=read_in_tables,
        )
        covell = CovELLSpace(
            cov_dict=covterms,
            obs_dict=observables,
            output_dict=output,
            cosmo_dict=cosmo,
            bias_dict=bias,
            iA_dict=iA,
            hod_dict=hod,
            survey_params_dict=survey_params,
            prec=prec,
            read_in_tables=read_in_tables,
        )
        covariance_in_ell_space = covell.calc_covELL(
            observables, output, bias, hod, survey_params, prec, read_in_tables
        )

        if len(covariance_in_ell_space) == 3:
            self.cov_g, self.cov_cng, self.cov_ssc = covariance_in_ell_space
        else:
            raise ValueError(
                'The length of the output covariance list should be 3 (g, ng, ssc)'
            )

    def oc_cov_to_10d(self, cov_tuple_in, nbl, compute_cov):
        assert len(cov_tuple_in) == 6, (
            'For the moment, OC cov tuple should have 6 entries (for 3 probes)'
        )

        cov_10d_out = np.zeros(
            (
                self.n_probes,
                self.n_probes,
                self.n_probes,
                self.n_probes,
                nbl,
                nbl,
                self.zbins,
                self.zbins,
                self.zbins,
                self.zbins,
            )
        )

        # guard
        if not compute_cov:
            return cov_10d_out

        # Ensure covariance shapes are correct
        for cov in cov_tuple_in:
            if isinstance(cov, int):
                assert cov == 0, "cov must be == 0 if it's a single integer"
            elif isinstance(cov, np.ndarray):
                assert cov.shape == (
                    nbl,
                    nbl,
                    1,
                    1,
                    self.zbins,
                    self.zbins,
                    self.zbins,
                    self.zbins,
                )
            else:
                raise ValueError('cov must be either an integer or a numpy array')

        # Update the cov_oc_3x2pt_10D for the given covariance type
        # the order of the tuple is gggg, gggm, ggmm, gmgm, mmgm, mmmm
        cov_order = [
            (1, 1, 1, 1),
            (1, 1, 1, 0),
            (1, 1, 0, 0),
            (1, 0, 1, 0),
            (0, 0, 1, 0),
            (0, 0, 0, 0),
        ]

        # Update the cov_oc_3x2pt_10D for the given covariance type
        for idx, (a, b, c, d) in enumerate(cov_order):
            if isinstance(cov_tuple_in[idx], np.ndarray):
                cov_10d_out[a, b, c, d, :, :, :, :, :, :] = (
                    cov_tuple_in[idx][:, :, 0, 0, :, :, :, :]
                ).copy()

        # Transpose to get the remaining blocks
        # ell1 <-> ell2 and zi, zj <-> zk, zl, but ell1 <-> ell2 should have no effect!
        cov_10d_out[0, 0, 1, 1, :, :, :, :, :, :] = (
            np.transpose(cov_10d_out[1, 1, 0, 0, :, :, :, :, :, :], (1, 0, 4, 5, 2, 3))
        ).copy()
        cov_10d_out[1, 0, 1, 1, :, :, :, :, :, :] = (
            np.transpose(cov_10d_out[1, 1, 1, 0, :, :, :, :, :, :], (1, 0, 4, 5, 2, 3))
        ).copy()
        cov_10d_out[1, 0, 0, 0, :, :, :, :, :, :] = (
            np.transpose(cov_10d_out[0, 0, 1, 0, :, :, :, :, :, :], (1, 0, 4, 5, 2, 3))
        ).copy()

        # check that the diagonal blocks (only the diagonal!!) are symmetric in
        # ell1, ell2
        for a, b, c, d in ((0, 0, 0, 0), (1, 0, 1, 0), (1, 1, 1, 1)):
            np.testing.assert_allclose(
                cov_10d_out[a, b, c, d, :, :, :, :, :, :],
                cov_10d_out[a, b, c, d, :, :, :, :, :, :].transpose(1, 0, 2, 3, 4, 5),
                atol=0,
                rtol=1e-3,
                err_msg='Diagonal blocks should be symmetric in ell1, ell2',
            )

        return cov_10d_out

    def process_cov_from_class(self):
        cov_sva_tuple = [self.cov_g[idx * 3] for idx in range(6)]
        cov_mix_tuple = [self.cov_g[idx * 3 + 1] for idx in range(6)]
        cov_sn_tuple = [self.cov_g[idx * 3 + 2] for idx in range(6)]

        self.cov_sva_oc_3x2pt_10D = self.oc_cov_to_10d(
            cov_tuple_in=cov_sva_tuple, nbl=self.nbl_3x2pt, compute_cov=self.compute_g
        )
        self.cov_mix_oc_3x2pt_10D = self.oc_cov_to_10d(
            cov_tuple_in=cov_mix_tuple, nbl=self.nbl_3x2pt, compute_cov=self.compute_g
        )
        self.cov_sn_oc_3x2pt_10D = self.oc_cov_to_10d(
            cov_tuple_in=cov_sn_tuple, nbl=self.nbl_3x2pt, compute_cov=self.compute_g
        )
        self.cov_ssc_oc_3x2pt_10D = self.oc_cov_to_10d(
            cov_tuple_in=self.cov_ssc, nbl=self.nbl_3x2pt, compute_cov=self.compute_ssc
        )
        self.cov_cng_oc_3x2pt_10D = self.oc_cov_to_10d(
            cov_tuple_in=self.cov_cng, nbl=self.nbl_3x2pt, compute_cov=self.compute_cng
        )

        self.cov_g_oc_3x2pt_10D = (
            self.cov_sva_oc_3x2pt_10D
            + self.cov_mix_oc_3x2pt_10D
            + self.cov_sn_oc_3x2pt_10D
        )

    def process_cov_from_mat_file(self):
        self.zpairs_auto, self.zpairs_cross, self.zpairs_3x2pt = sl.get_zpairs(
            self.zbins
        )

        self.cov_dict_matfmt = defaultdict(lambda: defaultdict(dict))

        elem_auto = self.zpairs_auto * self.nbl_3x2pt
        elem_cross = self.zpairs_cross * self.nbl_3x2pt

        if self.compute_g:
            cov_in = np.genfromtxt(
                f'{self.oc_path}/{self.cov_oc_fname}_matrix_gauss.mat'
            )
            self.cov_dict_matfmt['g']['3x2pt']['2d'] = self.cov_ggglll_to_llglgg(
                cov_in, elem_auto, elem_cross
            )

        if self.compute_ssc:
            cov_in = np.genfromtxt(f'{self.oc_path}/{self.cov_oc_fname}_matrix_SSC.mat')
            self.cov_dict_matfmt['ssc']['3x2pt']['2d'] = self.cov_ggglll_to_llglgg(
                cov_in, elem_auto, elem_cross
            )

        if self.compute_cng:
            cov_in = np.genfromtxt(
                f'{self.oc_path}/{self.cov_oc_fname}_matrix_nongauss.mat'
            )
            self.cov_dict_matfmt['cng']['3x2pt']['2d'] = self.cov_ggglll_to_llglgg(
                cov_in, elem_auto, elem_cross
            )

        cov_in = np.genfromtxt(f'{self.oc_path}/{self.cov_oc_fname}_matrix.mat')
        self.cov_dict_matfmt['tot']['3x2pt']['2d'] = self.cov_ggglll_to_llglgg(
            cov_in, elem_auto, elem_cross
        )

    def output_sanity_check(
        self,
        req_probe_combs_2d: list,
        cov_dict_6d_to_4d_and_2d_kw: dict,
        rtol: float = 1e-4,
    ):
        """
        Checks that the .dat and .mat outputs give consistent results
        """

        # process the covariance from the mat file.
        # This creates the 3x2pt 2D cov for the different terms
        self.process_cov_from_mat_file()

        if self.obs_space == 'harmonic':
            cov_4d_to_2dcloe_func = sl.cov_4D_to_2DCLOE_3x2pt_hs
        elif self.obs_space == 'real':
            cov_4d_to_2dcloe_func = sl.cov_4D_to_2DCLOE_3x2pt_rs

        # NOTE: 3x2pt 4d and 2d is created on-the-fly for this check,
        # and not stored in self. This is because of 2 reasons:
        # 1. The reshaping is centralized in the SB cov (hs/rs) classes.
        # 2. The zpair ordering is hardcoded in OC
        # TODO check point number 2, there is some option in the ini file...

        # create a copy to avoid polluting the original dict,
        # which has only 6d and no 3x2pt
        cov_dict_tmplist = deepcopy(self.cov_dict)
        # reshape individual blocks to 4d and 2d
        cov_dict_tmplist = sl.cov_dict_6d_probe_blocks_to_4d_and_2d(
            cov_dict_tmplist, **cov_dict_6d_to_4d_and_2d_kw
        )

        for term in self.cov_dict_matfmt:
            # create 3x2pt 4d
            cov_term_3x2pt_list_4d = sl.cov_dict_4d_probeblocks_to_3x2pt_4d_array(
                cov_dict_tmplist[term], self.obs_space
            )
            # create 3x2pt 2d
            cov_term_3x2pt_list_2d = cov_4d_to_2dcloe_func(
                cov_term_3x2pt_list_4d,
                zbins=self.zbins,
                req_probe_combs_2d=req_probe_combs_2d,
                block_index='zpair',
            )
            # compare with mat fmt
            np.testing.assert_allclose(
                cov_term_3x2pt_list_2d,
                self.cov_dict_matfmt[term]['3x2pt']['2d'],
                rtol=rtol,
                atol=0,
                err_msg=f'{term} covariance matrix from .mat file is'
                ' not consistent with .dat output',
            )

    def cov_ggglll_to_llglgg(
        self, cov_ggglll_2d: np.ndarray, elem_auto: int, elem_cross: int
    ) -> np.ndarray:
        """Transforms a covariance matrix from gg-gl-ll format to llglgg format.

        Parameters
        ----------
        cov_ggglll_2d : np.ndarray
            Input covariance matrix in gg-gl-ll format.
        elem_auto : int
            Number of auto elements in the covariance matrix.
        elem_cross : int
            Number of cross elements in the covariance matrix.

        Returns
        -------
        np.ndarray
            Transformed covariance matrix in mm-gm-gg format.

        """
        elem_apc = elem_auto + elem_cross

        cov_gggg_2d = cov_ggglll_2d[:elem_auto, :elem_auto]
        cov_gggl_2d = cov_ggglll_2d[:elem_auto, elem_auto:elem_apc]
        cov_ggll_2d = cov_ggglll_2d[:elem_auto, elem_apc:]
        cov_glgg_2d = cov_ggglll_2d[elem_auto:elem_apc, :elem_auto]
        cov_glgl_2d = cov_ggglll_2d[elem_auto:elem_apc, elem_auto:elem_apc]
        cov_glll_2d = cov_ggglll_2d[elem_auto:elem_apc, elem_apc:]
        cov_llgg_2d = cov_ggglll_2d[elem_apc:, :elem_auto]
        cov_llgl_2d = cov_ggglll_2d[elem_apc:, elem_auto:elem_apc]
        cov_llll_2d = cov_ggglll_2d[elem_apc:, elem_apc:]

        row_1 = np.concatenate((cov_llll_2d, cov_llgl_2d, cov_llgg_2d), axis=1)
        row_2 = np.concatenate((cov_glll_2d, cov_glgl_2d, cov_glgg_2d), axis=1)
        row_3 = np.concatenate((cov_ggll_2d, cov_gggl_2d, cov_gggg_2d), axis=1)

        cov_llglgg_2d = np.concatenate((row_1, row_2, row_3), axis=0)

        return cov_llglgg_2d

    def find_optimal_ellmax_oc(self, target_ell_array):
        upper_lim = self.ells_sb[-1] + 300
        lower_lim = self.ells_sb[-1] - 300
        lower_lim = max(lower_lim, 0)

        # Perform the minimization
        result = minimize_scalar(
            self.objective_function, bounds=[lower_lim, upper_lim], method='bounded'
        )

        # Check the result
        if result.success:
            self.optimal_ellmax = result.x
            print(f'Optimal ellmax found: {self.optimal_ellmax}')
        else:
            print('Optimization failed.')

        self.new_ells_oc = self.compute_ells_oc(
            nbl=int(self.pvt_cfg['nbl_3x2pt']),
            ell_min=float(self.pvt_cfg['ell_min_3x2pt']),
            ell_max=self.optimal_ellmax,
        )

        fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        fig.subplots_adjust(hspace=0)
        ax[0].plot(target_ell_array, label='target ells (SB)', marker='o', alpha=0.6)
        ax[0].plot(self.new_ells_oc, label='ells OC', marker='o', alpha=0.6)
        ax[1].plot(
            sl.percent_diff(target_ell_array, self.new_ells_oc),
            label='% diff',
            marker='o',
        )
        ax[0].legend()
        ax[1].legend()
        ax[0].set_ylabel('$\\ell$')
        ax[1].set_ylabel('SB/OC - 1 [%]')
        fig.supxlabel('ell idx')

    def compute_ells_oc(self, nbl, ell_min, ell_max):
        ell_bin_edges_oc_int = np.unique(
            np.geomspace(ell_min, ell_max, nbl + 1)
        ).astype(int)
        ells_oc_int = np.exp(
            0.5 * (np.log(ell_bin_edges_oc_int[1:]) + np.log(ell_bin_edges_oc_int[:-1]))
        )  # it's the same if I take base 10 log
        return ells_oc_int

    def objective_function(self, ell_max):
        ells_oc = self.compute_ells_oc(
            nbl=int(self.pvt_cfg['nbl_3x2pt']),
            ell_min=float(self.pvt_cfg['ell_min_3x2pt']),
            ell_max=ell_max,
        )
        ssd = np.sum((self.ells_sb - ells_oc) ** 2)
        # ssd = np.sum(sl.percent_diff(self.ells_sb, ells_oc)**2)  # TODO test this
        return ssd

    def get_oc_responses(self, ini_filename, h):
        import sys

        sys.path.append('/home/davide/Documenti/Lavoro/Programmi/OneCovariance')
        import os
        import platform

        from onecov.cov_ell_space import CovELLSpace
        from onecov.cov_input import FileInput, Input

        if len(platform.mac_ver()[0]) > 0 and (
            platform.processor() == 'arm'
            or int(platform.mac_ver()[0][: (platform.mac_ver()[0]).find('.')]) > 13
        ):
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        from onecov.cov_setup import Setup

        inp = Input()
        covterms, observables, output, cosmo, bias, iA, hod, survey_params, prec = (
            inp.read_input(ini_filename)
        )
        covterms['gauss'] = True  # in principle it could be False, but I get an error
        covterms['ssc'] = True
        covterms['nongauss'] = False
        fileinp = FileInput(bias)
        read_in_tables = fileinp.read_input(ini_filename)
        _setup = Setup(cosmo, bias, survey_params, prec, read_in_tables)
        ellspace = CovELLSpace(
            covterms,
            observables,
            output,
            cosmo,
            bias,
            iA,
            hod,
            survey_params,
            prec,
            read_in_tables,
        )
        _ssc = ellspace.covELL_ssc(
            bias, hod, prec, survey_params, observables['ELLspace']
        )

        dPmm_ddeltab = ellspace.aux_response_mm[:, :, 0] / h**3
        dPgm_ddeltab = ellspace.aux_response_gm[:, :, 0] / h**3
        dPgg_ddeltab = ellspace.aux_response_gg[:, :, 0] / h**3

        # all these results are *not* in h units
        resp_dict = {
            'dPmm_ddeltab': dPmm_ddeltab.T,
            'dPgm_ddeltab': dPgm_ddeltab.T,
            'dPgg_ddeltab': dPgg_ddeltab.T,
            'k_1Mpc': ellspace.mass_func.k * h,
            'z': ellspace.los_z,
        }

        return resp_dict
