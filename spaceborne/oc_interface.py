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
import time
import warnings
from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize_scalar

from spaceborne import sb_lib as sl


def print_cfg_onecov_ini(cfg_onecov_ini):
    """This is necessary since cfg_onecov_ini is not simply a dict (because I first
    load) an example .ini file..."""

    for section in cfg_onecov_ini.sections():
        print(f'[{section}]')
        for key, value in cfg_onecov_ini.items(section):
            print(f'{key} = {value}')
        print()  # Add a blank line for readability between sections


def process_cov_from_list_file_rs(
    oc_output_covlist_fname,
    n_probes_rs,
    probe_idx_dict_short_oc,
    zbins,
    df_chunk_size=5_000_000,
):
    import pandas as pd
    import re

    # set df column names
    with open(f'{oc_output_covlist_fname}') as file:
        header = (
            file.readline().strip()
        )  # Read the first line and strip newline characters
    header_list = re.split(
        '\t', header.strip().replace('\t\t', '\t').replace('\t\t', '\t')
    )
    column_names = header_list

    data = pd.read_csv(
        f'{oc_output_covlist_fname}', usecols=['theta1', 'tomoi'], sep='\s+'
    )

    thetas_oc_load = data['theta1'].unique()
    thetas_oc_load_rad = np.deg2rad(thetas_oc_load / 60)
    cov_theta_indices = {theta_out: idx for idx, theta_out in enumerate(thetas_oc_load)}
    nbt_oc = len(thetas_oc_load)

    # SB tomographic indices start from 0
    tomoi_oc_load = data['tomoi'].unique()
    subtract_one = False
    if min(tomoi_oc_load) == 1:
        subtract_one = True

    # ! import .list covariance file
    shape = (
        n_probes_rs,
        n_probes_rs,
        nbt_oc,
        nbt_oc,
        zbins,
        zbins,
        zbins,
        zbins,
    )
    cov_g_oc_3x2pt_8D = np.zeros(shape)
    cov_sva_oc_3x2pt_8D = np.zeros(shape)
    cov_mix_oc_3x2pt_8D = np.zeros(shape)
    cov_sn_oc_3x2pt_8D = np.zeros(shape)
    cov_ssc_oc_3x2pt_8D = np.zeros(shape)
    cov_cng_oc_3x2pt_8D = np.zeros(shape)
    # cov_tot_oc_3x2pt_8D = np.zeros(shape)

    print(f'Loading OneCovariance output from {oc_output_covlist_fname} file...')
    for df_chunk in pd.read_csv(
        f'{oc_output_covlist_fname}',
        sep='\s+',
        names=column_names,
        skiprows=1,
        chunksize=df_chunk_size,
    ):
        # Vectorize the extraction of probe indices
        probe_idx = df_chunk['#obs'].str[:].map(probe_idx_dict_short_oc).values
        probe_idx_arr = np.array(probe_idx.tolist())  # now shape is (N, 4)

        # Map 'ell' values to their corresponding indices
        theta1_idx = df_chunk['theta1'].map(cov_theta_indices).values
        theta2_idx = df_chunk['theta2'].map(cov_theta_indices).values

        # Compute z indices
        if subtract_one:
            z_indices = df_chunk[['tomoi', 'tomoj', 'tomok', 'tomol']].sub(1).values
        else:
            z_indices = df_chunk[['tomoi', 'tomoj', 'tomok', 'tomol']].values

        # Vectorized assignment to the arrays
        index_tuple = (  # fmt: skip
            probe_idx_arr[:, 0], probe_idx_arr[:, 1], theta1_idx, theta2_idx,
            z_indices[:, 0], z_indices[:, 1], z_indices[:, 2], z_indices[:, 3],
        )  # fmt: skip

        cov_sva_oc_3x2pt_8D[index_tuple] = df_chunk['covg sva'].values
        cov_mix_oc_3x2pt_8D[index_tuple] = df_chunk['covg mix'].values
        cov_sn_oc_3x2pt_8D[index_tuple] = df_chunk['covg sn'].values
        cov_g_oc_3x2pt_8D[index_tuple] = (
            df_chunk['covg sva'].values
            + df_chunk['covg mix'].values
            + df_chunk['covg sn'].values
        )
        cov_ssc_oc_3x2pt_8D[index_tuple] = df_chunk['covssc'].values
        cov_cng_oc_3x2pt_8D[index_tuple] = df_chunk['covng'].values
        # cov_tot_oc_3x2pt_8D[index_tuple] = df_chunk['cov'].values

    covs_8d_dict = {
        'cov_sva_oc_3x2pt_8D': cov_sva_oc_3x2pt_8D,
        'cov_mix_oc_3x2pt_8D': cov_mix_oc_3x2pt_8D,
        'cov_sn_oc_3x2pt_8D': cov_sn_oc_3x2pt_8D,
        'cov_g_oc_3x2pt_8D': cov_g_oc_3x2pt_8D,
        'cov_ssc_oc_3x2pt_8D': cov_ssc_oc_3x2pt_8D,
        'cov_cng_oc_3x2pt_8D': cov_cng_oc_3x2pt_8D,
        # cov_tot_oc_3x2pt_8D
    }

    # for cov_8d in covs_8d_dict:
    #     cov_8d[0, 0, 1, 1] = deepcopy(
    #         np.transpose(cov_8d[1, 1, 0, 0], (1, 0, 4, 5, 2, 3))
    #     )
    #     cov_8d[1, 0, 0, 0] = deepcopy(
    #         np.transpose(cov_8d[0, 0, 1, 0], (1, 0, 4, 5, 2, 3))
    #     )
    #     cov_8d[1, 0, 1, 1] = deepcopy(
    #         np.transpose(cov_8d[1, 1, 1, 0], (1, 0, 4, 5, 2, 3))
    #     )

    print('...done')
    return covs_8d_dict


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
            conda_base_path (str): The base path of the OneCovariance Conda environment.
            oc_path (str): The path to the OneCovariance output directory.
            path_to_oc_executable (str): The path to the OneCovariance executable.
            path_to_config_oc_ini (str): The path to the OneCovariance configuration
            INI file.

        """
        self.cfg = cfg
        self.oc_cfg = self.cfg['OneCovariance']
        self.cov_rs_cfg = self.cfg['cov_real_space']
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

        self.obs_space = self.cfg['covariance']['space']

        # paths and filenems
        self.conda_base_path = self.get_conda_base_path()
        self.path_to_oc_executable = cfg['OneCovariance']['path_to_oc_executable']

    def get_conda_base_path(self):
        try:
            # Run the conda info --base command and capture the output
            result = subprocess.run(
                ['conda', 'info', '--base'],
                stdout=subprocess.PIPE,
                check=True,
                text=True,
            )
            # Extract and return the base path
            return result.stdout.strip() + '/bin'
        except FileNotFoundError:
            return '/home/cosmo/davide.sciotti/software/anaconda3/bin'
        except subprocess.CalledProcessError as e:
            print(f'Error occurred: {e}')
            return None

    def build_save_oc_ini(self, ascii_filenames_dict, print_ini=True):
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
        cfg_oc_ini['covTHETAspace settings'] = {}  # For real_space case
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
        if self.obs_space == 'harmonic_space':
            est_shear = 'C_ell'
            est_ggl = 'C_ell'
            est_clust = 'C_ell'
        elif self.obs_space == 'real_space':
            est_shear = 'xi_pm'
            est_ggl = 'gamma_t'
            est_clust = 'w'
        else:
            raise ValueError('self.which_obs must he "harmonic_space" or "real_space"')

        cfg_oc_ini['observables']['cosmic_shear'] = str(True)
        cfg_oc_ini['observables']['est_shear'] = est_shear
        cfg_oc_ini['observables']['ggl'] = str(True)
        cfg_oc_ini['observables']['est_ggl'] = est_ggl
        cfg_oc_ini['observables']['clustering'] = str(True)
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

        ell_binning_type = self.cfg['ell_binning']['binning_type']
        if self.cfg['ell_binning']['binning_type'] == 'ref_cut':
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
        if self.obs_space == 'harmonic_space':
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

        elif self.obs_space == 'real_space':
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
        if self.obs_space == 'real_space':
            cfg_oc_ini['covTHETAspace settings']['theta_min_clustering'] = str(
                self.cov_rs_cfg['theta_min_arcmin']
            )
            cfg_oc_ini['covTHETAspace settings']['theta_max_clustering'] = str(
                self.cov_rs_cfg['theta_max_arcmin']
            )
            cfg_oc_ini['covTHETAspace settings']['theta_bins_clustering'] = str(
                self.cfg['cov_real_space']['theta_bins']
            )
            cfg_oc_ini['covTHETAspace settings']['theta_type_clustering'] = 'lin'
            cfg_oc_ini['covTHETAspace settings']['theta_min_lensing'] = str(
                self.cov_rs_cfg['theta_min_arcmin']
            )
            cfg_oc_ini['covTHETAspace settings']['theta_max_lensing'] = str(
                self.cov_rs_cfg['theta_max_arcmin']
            )
            cfg_oc_ini['covTHETAspace settings']['theta_bins_lensing'] = str(
                self.cov_rs_cfg['theta_bins']
            )
            cfg_oc_ini['covTHETAspace settings']['theta_type_lensing'] = 'lin'

            cfg_oc_ini['covTHETAspace settings']['theta_min'] = str(
                self.cov_rs_cfg['theta_min_arcmin']
            )
            cfg_oc_ini['covTHETAspace settings']['theta_max'] = str(
                self.cov_rs_cfg['theta_max_arcmin']
            )
            cfg_oc_ini['covTHETAspace settings']['theta_type'] = 'lin'

            cfg_oc_ini['covTHETAspace settings']['xi_pp'] = str(True)
            cfg_oc_ini['covTHETAspace settings']['xi_mm'] = str(True)
            cfg_oc_ini['covTHETAspace settings']['theta_accuracy'] = str(1e-3)
            cfg_oc_ini['covTHETAspace settings']['integration_intervals'] = str(40)

        # ! [halomodel evaluation]
        if ('Tinker10' not in self.cfg['halo_model']['mass_function']) or (
            'Tinker10' not in self.cfg['halo_model']['halo_bias']
        ):
            warnings.warn('Only Tinker10 supported by OC at the moment', stacklevel=2)

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
            self.cfg['covariance']['log10_k_min']
        )
        cfg_oc_ini['powspec evaluation']['log10k_max'] = str(
            self.cfg['covariance']['log10_k_max']
        )
        cfg_oc_ini['powspec evaluation']['log10k_bins'] = str(
            self.cfg['covariance']['k_steps']
        )

        # ! [trispec evaluation]
        cfg_oc_ini['trispec evaluation']['log10k_min'] = str(
            self.cfg['covariance']['log10_k_min']
        )
        cfg_oc_ini['trispec evaluation']['log10k_max'] = str(
            self.cfg['covariance']['log10_k_max']
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

    def call_oc_from_bash(self):
        """This function runs OneCovariance"""
        activate_and_run = f"""
        source {self.conda_base_path}/activate cov20_env
        python {self.path_to_oc_executable} {self.path_to_config_oc_ini}
        source {self.conda_base_path}/deactivate
        source {self.conda_base_path}/activate spaceborne-dav
        """
        # python {self.path_to_oc_executable.replace('covariance.py', 'reshape_cov_list_Cl_callable.py')} {self.path_to_config_oc_ini.replace('input_configs.ini', '')}

        process = subprocess.Popen(activate_and_run, shell=True, executable='/bin/bash')
        process.communicate()

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
                cov_10d_out[a, b, c, d, :, :, :, :, :, :] = deepcopy(
                    cov_tuple_in[idx][:, :, 0, 0, :, :, :, :]
                )

        # Transpose to get the remaining blocks
        # ell1 <-> ell2 and zi, zj <-> zk, zl, but ell1 <-> ell2 should have no effect!
        cov_10d_out[0, 0, 1, 1, :, :, :, :, :, :] = deepcopy(
            np.transpose(cov_10d_out[1, 1, 0, 0, :, :, :, :, :, :], (1, 0, 4, 5, 2, 3))
        )
        cov_10d_out[1, 0, 1, 1, :, :, :, :, :, :] = deepcopy(
            np.transpose(cov_10d_out[1, 1, 1, 0, :, :, :, :, :, :], (1, 0, 4, 5, 2, 3))
        )
        cov_10d_out[1, 0, 0, 0, :, :, :, :, :, :] = deepcopy(
            np.transpose(cov_10d_out[0, 0, 1, 0, :, :, :, :, :, :], (1, 0, 4, 5, 2, 3))
        )

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

        elem_auto = self.zpairs_auto * self.nbl_3x2pt
        elem_cross = self.zpairs_cross * self.nbl_3x2pt

        if self.compute_g:
            cov_in = np.genfromtxt(f'{self.oc_path}/covariance_matrix_gauss.mat')
            self.cov_mat_g_2d = self.cov_ggglll_to_llglgg(cov_in, elem_auto, elem_cross)

        if self.compute_ssc:
            cov_in = np.genfromtxt(f'{self.oc_path}/covariance_matrix_SSC.mat')
            self.cov_mat_ssc_2d = self.cov_ggglll_to_llglgg(
                cov_in, elem_auto, elem_cross
            )

        if self.compute_cng:
            cov_in = np.genfromtxt(f'{self.oc_path}/covariance_matrix_nongauss.mat')
            self.cov_mat_cng_2d = self.cov_ggglll_to_llglgg(
                cov_in, elem_auto, elem_cross
            )

        cov_in = np.genfromtxt(f'{self.oc_path}/covariance_matrix.mat')
        self.cov_mat_tot_2d = self.cov_ggglll_to_llglgg(cov_in, elem_auto, elem_cross)

    def output_sanity_check(self, rtol=1e-4):
        """
        Checks that the .dat and .mat outputs give consistent results
        """

        # TODO why am I processing the output twice?
        # TODO this should be generalised to real space

        self.process_cov_from_mat_file()

        cov_list_g_4d = sl.cov_3x2pt_10D_to_4D(
            self.cov_g_oc_3x2pt_10D,
            self.probe_ordering,
            self.nbl_3x2pt,
            self.zbins,
            self.ind,
            self.GL_OR_LG,
        )
        cov_list_ssc_4d = sl.cov_3x2pt_10D_to_4D(
            self.cov_ssc_oc_3x2pt_10D,
            self.probe_ordering,
            self.nbl_3x2pt,
            self.zbins,
            self.ind,
            self.GL_OR_LG,
        )
        cov_list_cng_4d = sl.cov_3x2pt_10D_to_4D(
            self.cov_cng_oc_3x2pt_10D,
            self.probe_ordering,
            self.nbl_3x2pt,
            self.zbins,
            self.ind,
            self.GL_OR_LG,
        )
        cov_list_tot_4d = sl.cov_3x2pt_10D_to_4D(
            self.cov_tot_oc_3x2pt_10D,
            self.probe_ordering,
            self.nbl_3x2pt,
            self.zbins,
            self.ind,
            self.GL_OR_LG,
        )

        cov_list_g_2d = sl.cov_4D_to_2DCLOE_3x2pt(
            cov_list_g_4d, self.zbins, block_index='zpair'
        )
        cov_list_ssc_2d = sl.cov_4D_to_2DCLOE_3x2pt(
            cov_list_ssc_4d, self.zbins, block_index='zpair'
        )
        cov_list_cng_2d = sl.cov_4D_to_2DCLOE_3x2pt(
            cov_list_cng_4d, self.zbins, block_index='zpair'
        )
        cov_list_tot_2d = sl.cov_4D_to_2DCLOE_3x2pt(
            cov_list_tot_4d, self.zbins, block_index='zpair'
        )

        if self.compute_g:
            np.testing.assert_allclose(
                cov_list_g_2d,
                self.cov_mat_g_2d,
                atol=0,
                rtol=rtol,
                err_msg='Gaussian covariance matrix from .mat file is'
                ' not consistent with .dat output',
            )

        if self.compute_ssc:
            np.testing.assert_allclose(
                cov_list_ssc_2d,
                self.cov_mat_ssc_2d,
                atol=0,
                rtol=rtol,
                err_msg='SSC covariance matrix from .mat file is'
                ' not consistent with .dat output',
            )

        if self.compute_cng:
            np.testing.assert_allclose(
                cov_list_cng_2d,
                self.cov_mat_cng_2d,
                atol=0,
                rtol=rtol,
                err_msg='cNG covariance matrix from .mat file is'
                ' not consistent with .dat output',
            )

        np.testing.assert_allclose(
            cov_list_tot_2d,
            self.cov_mat_tot_2d,
            atol=0,
            rtol=rtol,
            err_msg='Gaussian covariance matrix from .mat file is'
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

    def process_cov_from_list_file_hs(self, df_chunk_size=5_000_000):
        """
        Import and reshape the output of the OneCovariance (OC) .dat
        (aka "list") file into a set of 10d arrays.
        The function also performs some additional processing,
        such as symmetrizing the output dictionary.
        """
        import re

        import pandas as pd

        # set df column names
        with open(f'{self.oc_path}/{self.cov_oc_fname}_list.dat') as file:
            header = (
                file.readline().strip()
            )  # Read the first line and strip newline characters
        header_list = re.split(
            '\t', header.strip().replace('\t\t', '\t').replace('\t\t', '\t')
        )
        column_names = header_list

        # ell values actually used in OC; save in self to be able to compare to
        # the SB ell values
        # note use delim_whitespace=True instead of sep='\s+' if this gives
        # compatibility issues
        self.ells_oc_load = pd.read_csv(
            f'{self.oc_path}/{self.cov_oc_fname}_list.dat', usecols=['ell1'], sep='\s+'
        )['ell1'].unique()

        # check if the saved ells are within 1% of the required ones;
        # I think the saved values are truncated to only
        # 2 decimals, so this is a rough comparison (rtol is 1%)
        try:
            np.testing.assert_allclose(
                self.new_ells_oc, self.ells_oc_load, atol=0, rtol=1e-2
            )
        except AssertionError as err:
            print('ell values computed vs loaded for OC are not the same')
            print(err)

        cov_ell_indices = {
            ell_out: idx for idx, ell_out in enumerate(self.ells_oc_load)
        }

        probe_idx_dict = {
            'm': 0,
            'g': 1,
        }

        # ! import .list covariance file
        shape = (
            self.n_probes,
            self.n_probes,
            self.n_probes,
            self.n_probes,
            self.nbl_3x2pt,
            self.nbl_3x2pt,
            self.zbins,
            self.zbins,
            self.zbins,
            self.zbins,
        )
        self.cov_g_oc_3x2pt_10D = np.zeros(shape)
        self.cov_sva_oc_3x2pt_10D = np.zeros(shape)
        self.cov_mix_oc_3x2pt_10D = np.zeros(shape)
        self.cov_sn_oc_3x2pt_10D = np.zeros(shape)
        self.cov_ssc_oc_3x2pt_10D = np.zeros(shape)
        self.cov_cng_oc_3x2pt_10D = np.zeros(shape)
        self.cov_tot_oc_3x2pt_10D = np.zeros(shape)

        print('Loading OneCovariance output from covariance_list.dat file...')
        start = time.perf_counter()
        for df_chunk in pd.read_csv(
            f'{self.oc_path}/covariance_list.dat',
            sep='\s+',
            names=column_names,
            skiprows=1,
            chunksize=df_chunk_size,
        ):
            # Vectorize the extraction of probe indices
            probe_idx_a = df_chunk['#obs'].str[0].map(probe_idx_dict).values
            probe_idx_b = df_chunk['#obs'].str[1].map(probe_idx_dict).values
            probe_idx_c = df_chunk['#obs'].str[2].map(probe_idx_dict).values
            probe_idx_d = df_chunk['#obs'].str[3].map(probe_idx_dict).values

            # Map 'ell' values to their corresponding indices
            ell1_idx = df_chunk['ell1'].map(cov_ell_indices).values
            ell2_idx = df_chunk['ell2'].map(cov_ell_indices).values

            # Compute z indices
            if np.min(df_chunk[['tomoi', 'tomoj', 'tomok', 'tomol']].values) == 1:
                z_indices = df_chunk[['tomoi', 'tomoj', 'tomok', 'tomol']].sub(1).values
            else:
                warnings.warn('tomo indices seem to start from 0...', stacklevel=2)
                z_indices = df_chunk[['tomoi', 'tomoj', 'tomok', 'tomol']].values

            # Vectorized assignment to the arrays
            index_tuple = (
                probe_idx_a,
                probe_idx_b,
                probe_idx_c,
                probe_idx_d,
                ell1_idx,
                ell2_idx,
                z_indices[:, 0],
                z_indices[:, 1],
                z_indices[:, 2],
                z_indices[:, 3],
            )

            self.cov_sva_oc_3x2pt_10D[index_tuple] = df_chunk['covg sva'].values
            self.cov_mix_oc_3x2pt_10D[index_tuple] = df_chunk['covg mix'].values
            self.cov_sn_oc_3x2pt_10D[index_tuple] = df_chunk['covg sn'].values
            self.cov_g_oc_3x2pt_10D[index_tuple] = (
                df_chunk['covg sva'].values
                + df_chunk['covg mix'].values
                + df_chunk['covg sn'].values
            )
            self.cov_ssc_oc_3x2pt_10D[index_tuple] = df_chunk['covssc'].values
            self.cov_cng_oc_3x2pt_10D[index_tuple] = df_chunk['covng'].values
            self.cov_tot_oc_3x2pt_10D[index_tuple] = df_chunk['cov'].values

        covs_10d = [
            self.cov_sva_oc_3x2pt_10D,
            self.cov_mix_oc_3x2pt_10D,
            self.cov_sn_oc_3x2pt_10D,
            self.cov_g_oc_3x2pt_10D,
            self.cov_ssc_oc_3x2pt_10D,
            self.cov_cng_oc_3x2pt_10D,
            self.cov_tot_oc_3x2pt_10D,
        ]

        for cov_10d in covs_10d:
            cov_10d[0, 0, 1, 1] = deepcopy(
                np.transpose(cov_10d[1, 1, 0, 0], (1, 0, 4, 5, 2, 3))
            )
            cov_10d[1, 0, 0, 0] = deepcopy(
                np.transpose(cov_10d[0, 0, 1, 0], (1, 0, 4, 5, 2, 3))
            )
            cov_10d[1, 0, 1, 1] = deepcopy(
                np.transpose(cov_10d[1, 1, 1, 0], (1, 0, 4, 5, 2, 3))
            )

        print(
            f'OneCovariance output loaded in {time.perf_counter() - start:.2f} seconds'
        )

    def _oc_output_to_dict_or_array(
        self, which_ng_cov, output_type, ind_dict=None, symmetrize_output_dict=None
    ):
        # ! THIS FUNCTION IS DEPRECATED

        # import
        filename = self.cov_filename.format(
            which_ng_cov=which_ng_cov,
            probe_a='{probe_a:s}',
            probe_b='{probe_b:s}',
            probe_c='{probe_c:s}',
            probe_d='{probe_d:s}',
        )
        cov_ng_oc_3x2pt_dict_8D = sl.load_cov_from_probe_blocks(
            path=self.oc_path,
            filename=filename,
            probe_ordering=self.cfg['covariance']['probe_ordering'],
        )

        # reshape
        if output_type == '8D_dict':
            return cov_ng_oc_3x2pt_dict_8D

        elif output_type in ['10D_dict', '10D_array']:
            cov_ng_oc_3x2pt_dict_10D = sl.cov_3x2pt_dict_8d_to_10d(
                cov_3x2pt_dict_8D=cov_ng_oc_3x2pt_dict_8D,
                nbl=self.pvt_cfg['nbl_3x2pt'],
                zbins=self.zbins,
                ind_dict=ind_dict,
                probe_ordering=self.cfg['covariance']['probe_ordering'],
                symmetrize_output_dict=symmetrize_output_dict,
            )

            if output_type == '10D_dict':
                return cov_ng_oc_3x2pt_dict_10D

            elif output_type == '10D_array':
                return sl.cov_10D_dict_to_array(
                    cov_ng_oc_3x2pt_dict_10D,
                    nbl=self.pvt_cfg['nbl_3x2pt'],
                    zbins=self.zbins,
                    n_probes=self.cfg['covariance']['n_probes'],
                )

        else:
            raise ValueError('output_dict_dim must be 8D or 10D')

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
