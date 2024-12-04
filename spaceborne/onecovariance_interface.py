"""
OneCovariance Interface Module

This module provides an interface to the OneCovariance (OC) covariance matrix calculator.
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

import gc
import os
import multiprocessing
import re
num_cores = multiprocessing.cpu_count()
os.environ['OMP_NUM_THREADS'] = '32'
os.environ['NUMBA_NUM_THREADS'] = '32'
os.environ['NUMBA_PARALLEL_DIAGNOSTICS'] = '4'

from matplotlib import pyplot as plt
import numpy as np
import time
import configparser
import warnings
from copy import deepcopy
import pandas as pd

import spaceborne.ell_utils as ell_utils
import spaceborne.my_module as mm
import subprocess
from scipy.optimize import minimize_scalar


class OneCovarianceInterface():

    def __init__(self, ROOT, cfg, variable_specs, do_ssc, do_cng):
        """
        Initializes the OneCovarianceInterface class with the provided configuration and variable specifications.

        Args:
            ROOT (str): The root directory of the project.
            cfg (dict): The configuration dictionary.
            variable_specs (dict): The variable specifications dictionary.
            do_ssc (bools): Whether to compute the SSC term.
            do_cng (bool): Whether to compute the connected non-Gaussian covariance term.

        Attributes:
            cfg (dict): The configuration dictionary.
            oc_cfg (dict): The OneCovariance configuration dictionary.
            variable_specs (dict): The variable specifications dictionary.
            zbins (int): The number of redshift bins.
            nbl_3x2pt (int): The number of ell bins for the 3x2pt analysis.
            compute_ssc (bool): Whether to compute the super-sample covariance (SSC) term.
            compute_cng (bool): Whether to compute the connected non-Gaussian covariance term.
            ROOT (str): The root directory of the project.
            conda_base_path (str): The base path of the OneCovariance Conda environment.
            oc_path (str): The path to the OneCovariance output directory.
            path_to_oc_executable (str): The path to the OneCovariance executable.
            path_to_config_oc_ini (str): The path to the OneCovariance configuration INI file.
        """

        self.cfg = cfg
        self.oc_cfg = self.cfg['OneCovariance']
        self.variable_specs = variable_specs
        self.n_probes = cfg['covariance']['n_probes']

        # set which cov terms to compute from cfg file
        self.compute_ssc = do_ssc
        self.compute_cng = do_cng

        # paths
        # TODO do we really need ROOT?
        self.ROOT = ROOT
        self.conda_base_path = self.get_conda_base_path()
        self.path_to_oc_executable = cfg['OneCovariance']['path_to_oc_executable'].format(
            ROOT=ROOT)
        self.cov_filename = 'cov_OC_{which_ng_cov:s}_{probe_a:s}{probe_b:s}{probe_c:s}{probe_d:s}.npz'

    def get_conda_base_path(self):
        try:
            # Run the conda info --base command and capture the output
            result = subprocess.run(['conda', 'info', '--base'], stdout=subprocess.PIPE, check=True, text=True)
            # Extract and return the base path
            return result.stdout.strip() + '/bin'
        except FileNotFoundError as e:
            return '/home/cosmo/davide.sciotti/software/anaconda3/bin'
        except subprocess.CalledProcessError as e:
            print(f"Error occurred: {e}")
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

        # TODO import another file??
        # Read the existing reference .ini file
        cfg_onecov_ini = CaseConfigParser()
        cfg_onecov_ini.read(f'{self.ROOT}/Spaceborne/input/config_3x2pt_pure_Cell_general.ini')

        # set useful lists
        mult_shear_bias_list = np.array(self.cfg['C_ell']['mult_shear_bias'])
        n_eff_clust_list = self.cfg['nz']['ngal_lenses']
        n_eff_lensing_list = self.cfg['nz']['ngal_sources']
        ellipticity_dispersion_list = [self.cfg['covariance']['sigma_eps_i']] * self.zbins

        cfg_onecov_ini['covariance terms']['gauss'] = str(True)
        cfg_onecov_ini['covariance terms']['split_gauss'] = str(False)
        cfg_onecov_ini['covariance terms']['nongauss'] = str(self.compute_cng)
        cfg_onecov_ini['covariance terms']['ssc'] = str(self.compute_ssc)
        cfg_onecov_ini['output settings']['directory'] = self.oc_path

        # [observables]
        cfg_onecov_ini['observables']['cosmic_shear'] = str(True)
        cfg_onecov_ini['observables']['est_shear'] = 'C_ell'
        cfg_onecov_ini['observables']['ggl'] = str(True)
        cfg_onecov_ini['observables']['est_ggl'] = 'C_ell'
        cfg_onecov_ini['observables']['clustering'] = str(True)
        cfg_onecov_ini['observables']['est_clust'] = 'C_ell'
        cfg_onecov_ini['observables']['cstellar_mf'] = str(False)
        cfg_onecov_ini['observables']['cross_terms'] = str(True)
        cfg_onecov_ini['observables']['unbiased_clustering'] = str(False)

        cfg_onecov_ini['covELLspace settings']['ell_min'] = str(self.variable_specs['ell_min'])
        cfg_onecov_ini['covELLspace settings']['ell_bins'] = str(self.variable_specs['nbl_3x2pt'])
        cfg_onecov_ini['covELLspace settings']['ell_min_lensing'] = str(self.variable_specs['ell_min'])
        cfg_onecov_ini['covELLspace settings']['ell_bins_lensing'] = str(self.variable_specs['nbl_3x2pt'])
        cfg_onecov_ini['covELLspace settings']['ell_min_clustering'] = str(self.variable_specs['ell_min'])
        cfg_onecov_ini['covELLspace settings']['ell_bins_clustering'] = str(self.variable_specs['nbl_3x2pt'])
        cfg_onecov_ini['covELLspace settings']['mult_shear_bias'] = ', '.join(map(str, mult_shear_bias_list))

        # find best ell_max for OC, since it uses a slightly different recipe
        self.find_optimal_ellmax_oc(target_ell_array=self.ells_sb)
        cfg_onecov_ini['covELLspace settings']['ell_max'] = str(self.optimal_ellmax)
        cfg_onecov_ini['covELLspace settings']['ell_max_lensing'] = str(self.optimal_ellmax)
        cfg_onecov_ini['covELLspace settings']['ell_max_clustering'] = str(self.optimal_ellmax)

        # commented out to avoid loading mask file by accident
        cfg_onecov_ini['survey specs']['mask_directory'] = str(
            self.cfg['mask']['mask_path'])  # TODO test this!!
        cfg_onecov_ini['survey specs']['survey_area_lensing_in_deg2'] = str(
            self.cfg['mask']['survey_area_deg2'])
        cfg_onecov_ini['survey specs']['survey_area_ggl_in_deg2'] = str(
            self.cfg['mask']['survey_area_deg2'])
        cfg_onecov_ini['survey specs']['survey_area_clust_in_deg2'] = str(
            self.cfg['mask']['survey_area_deg2'])
        cfg_onecov_ini['survey specs']['n_eff_clust'] = ', '.join(map(str, n_eff_clust_list))
        cfg_onecov_ini['survey specs']['n_eff_lensing'] = ', '.join(map(str, n_eff_lensing_list))
        cfg_onecov_ini['survey specs']['ellipticity_dispersion'] = ', '.join(map(str, ellipticity_dispersion_list))

        cfg_onecov_ini['redshift']['z_directory'] = self.oc_path
        # TODO re-check that the OC documentation is correct
        cfg_onecov_ini['redshift']['zclust_file'] = nz_lns_filename_ascii
        cfg_onecov_ini['redshift']['zlens_file'] = nz_src_filename_ascii

        cfg_onecov_ini['cosmo']['h'] = str(self.cfg['cosmology']['h'])
        cfg_onecov_ini['cosmo']['ns'] = str(self.cfg['cosmology']['ns'])
        cfg_onecov_ini['cosmo']['omega_m'] = str(self.cfg['cosmology']['Om'])
        cfg_onecov_ini['cosmo']['omega_b'] = str(self.cfg['cosmology']['Ob'])
        cfg_onecov_ini['cosmo']['omega_de'] = str(self.cfg['cosmology']['ODE'])
        cfg_onecov_ini['cosmo']['sigma8'] = str(self.cfg['cosmology']['s8'])
        cfg_onecov_ini['cosmo']['w0'] = str(self.cfg['cosmology']['wz'])
        cfg_onecov_ini['cosmo']['wa'] = str(self.cfg['cosmology']['wa'])
        cfg_onecov_ini['cosmo']['neff'] = str(self.cfg['cosmology']['N_eff'])
        cfg_onecov_ini['cosmo']['m_nu'] = str(self.cfg['cosmology']['m_nu'])

        if self.cfg["covariance"]["which_b1g_in_resp"] == 'from_input':
            gal_bias_ascii_filename = ascii_filenames_dict['gal_bias_ascii_filename']
            cfg_onecov_ini['bias']['bias_files'] = gal_bias_ascii_filename

        cfg_onecov_ini['IA']['A_IA'] = str(self.cfg['intrinsic_alignment']['Aia'])
        cfg_onecov_ini['IA']['eta_IA'] = str(self.cfg['intrinsic_alignment']['eIA'])
        cfg_onecov_ini['IA']['z_pivot_IA'] = str(self.cfg['intrinsic_alignment']['z_pivot_IA'])

        cfg_onecov_ini['powspec evaluation']['non_linear_model'] = str(
            self.cfg['extra_parameters']['camb']['halofit_version'])
        cfg_onecov_ini['powspec evaluation']['HMCode_logT_AGN'] = str(
            self.cfg['extra_parameters']['camb']['HMCode_logT_AGN'])

        cfg_onecov_ini['tabulated inputs files']['Cell_directory'] = self.oc_path
        cfg_onecov_ini['tabulated inputs files']['Cmm_file'] = f'{cl_ll_oc_filename}.ascii'
        cfg_onecov_ini['tabulated inputs files']['Cgm_file'] = f'{cl_gl_oc_filename}.ascii'
        cfg_onecov_ini['tabulated inputs files']['Cgg_file'] = f'{cl_gg_oc_filename}.ascii'

        cfg_onecov_ini['misc']['num_cores'] = str(self.cfg['misc']['num_threads'])

        # ! precision settings
        if self.oc_cfg['precision_settings'] == 'high_precision':
            delta_z = 0.04
            tri_delta_z = 0.25
            integration_steps = 1000
            m_bins = 1500  # 900 or 1500
            log10k_bins = 200  # 150 or 200
        elif self.oc_cfg['precision_settings'] == 'default':  # these are the default values, used by Robert as well
            delta_z = 0.08
            tri_delta_z = 0.5
            integration_steps = 500
            m_bins = 900
            log10k_bins = 100
        else:
            raise ValueError(f"Unknown precision settings: {self.oc_cfg['precision_settings']}")

        cfg_onecov_ini['covELLspace settings']['delta_z'] = str(delta_z)
        cfg_onecov_ini['covELLspace settings']['tri_delta_z'] = str(tri_delta_z)
        cfg_onecov_ini['covELLspace settings']['integration_steps'] = str(integration_steps)
        cfg_onecov_ini['halomodel evaluation']['m_bins'] = str(m_bins)
        cfg_onecov_ini['trispec evaluation']['log10k_bins'] = str(log10k_bins)

        # print the updated ini
        if print_ini:
            for section in cfg_onecov_ini.sections():
                print(f"[{section}]")
                for key, value in cfg_onecov_ini[section].items():
                    print(f"{key} = {value}")
                print()

        # Save the updated configuration to a new .ini file
        with open(f'{self.oc_path}/input_configs.ini', 'w') as configfile:
            cfg_onecov_ini.write(configfile)

        # store in self for good measure
        self.cfg_onecov_ini = cfg_onecov_ini

    def _call_onecovariance(self):
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

    def call_onecovariance(self):
        """This interface was originally created by Robert Reischke"""
        import sys
        sys.path.append(os.path.dirname(self.path_to_oc_executable))
        from onecov.cov_input import Input, FileInput
        from onecov.cov_ell_space import CovELLSpace
        from onecov.cov_polyspectra import PolySpectra
        import platform
        if len(platform.mac_ver()[0]) > 0 and (platform.processor() == 'arm' or int(platform.mac_ver()[0][:(platform.mac_ver()[0]).find(".")]) > 13):
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        from onecov.cov_setup import Setup

        print("READING OneCovariance INPUT")
        print("#############")

        inp = Input()

        covterms, observables, output, cosmo, bias, iA, hod, survey_params, prec = inp.read_input(
            f"{self.oc_path}/input_configs.ini")
        fileinp = FileInput(bias)
        read_in_tables = fileinp.read_input(f"{self.oc_path}/input_configs.ini")
        setup = Setup(cosmo_dict=cosmo,
                      bias_dict=bias,
                      survey_params_dict=survey_params,
                      prec=prec,
                      read_in_tables=read_in_tables)
        ellspace = CovELLSpace(cov_dict=covterms,
                               obs_dict=observables,
                               output_dict=output,
                               cosmo_dict=cosmo,
                               bias_dict=bias,
                               iA_dict=iA,
                               hod_dict=hod,
                               survey_params_dict=survey_params,
                               prec=prec,
                               read_in_tables=read_in_tables)
        self.cov_g = ellspace.covELL_gaussian(
            covELLspacesettings=observables['ELLspace'],
            survey_params_dict=survey_params,
            calc_prefac=True)
        self.cov_ssc = ellspace.covELL_ssc(bias_dict=bias,
                                           hod_dict=hod,
                                           prec=prec,
                                           survey_params_dict=survey_params,
                                           covELLspacesettings=observables['ELLspace'])
        self.cov_cng = ellspace.covELL_non_gaussian(covELLspacesettings=observables['ELLspace'],
                                                    output_dict=output,
                                                    bias_dict=bias,
                                                    hod_dict=hod,
                                                    prec=prec,
                                                    tri_tab=read_in_tables['tri'])

    def reshape_oc_output(self):
        
        cov_g_oc_10d = np.zeros((self.n_probes, self.n_probes, self.n_probes, self.n_probes,
                                cov_nbl, cov_nbl, self.zbins, self.zbins, self.zbins, self.zbins))
        cov_sva_oc_10d = np.zeros((self.n_probes, self.n_probes, self.n_probes, self.n_probes,
                                  cov_nbl, cov_nbl, self.zbins, self.zbins, self.zbins, self.zbins))
        cov_mix_oc_10d = np.zeros((self.n_probes, self.n_probes, self.n_probes, self.n_probes,
                                  cov_nbl, cov_nbl, self.zbins, self.zbins, self.zbins, self.zbins))
        cov_sn_oc_10d = np.zeros((self.n_probes, self.n_probes, self.n_probes, self.n_probes,
                                 cov_nbl, cov_nbl, self.zbins, self.zbins, self.zbins, self.zbins))
        cov_ssc_oc_10d = np.zeros((self.n_probes, self.n_probes, self.n_probes, self.n_probes,
                                  cov_nbl, cov_nbl, self.zbins, self.zbins, self.zbins, self.zbins))
        cov_cng_oc_10d = np.zeros((self.n_probes, self.n_probes, self.n_probes, self.n_probes,
                                  cov_nbl, cov_nbl, self.zbins, self.zbins, self.zbins, self.zbins))
        cov_tot_oc_10d = np.zeros((self.n_probes, self.n_probes, self.n_probes, self.n_probes,
                                  cov_nbl, cov_nbl, self.zbins, self.zbins, self.zbins, self.zbins))

        assert len(self.cov_ssc) == 6, 'For the moment, SSC OC tuple should have 6 entries (for 3 probes)'
        assert len(self.cov_cng) == 6, 'For the moment, cNG OC tuple should have 6 entries (for 3 probes)'
        for i in len(self.cov_ssc):
            assert self.cov_ssc[i].shape == (cov_nbl, cov_nbl, self.zbins, self.zbins, self.zbins, self.zbins)
        for i in len(self.cov_cng):
            assert self.cov_cng[i].shape == (cov_nbl, cov_nbl, self.zbins, self.zbins, self.zbins, self.zbins)

        # the order of the tuple is gggg, gggm, ggmm, gmgm, mmgm, mmmm (confirmed by Robert)
        self.cov_ssc_oc_10d[1, 1, 1, 1, :, :, :, :, :, :] = self.cov_ssc[0][:, :, 0, 0, :, :, :, :]
        self.cov_ssc_oc_10d[1, 1, 1, 0, :, :, :, :, :, :] = self.cov_ssc[1][:, :, 0, 0, :, :, :, :]
        self.cov_ssc_oc_10d[1, 1, 0, 0, :, :, :, :, :, :] = self.cov_ssc[2][:, :, 0, 0, :, :, :, :]
        self.cov_ssc_oc_10d[1, 0, 1, 0, :, :, :, :, :, :] = self.cov_ssc[3][:, :, 0, 0, :, :, :, :]
        self.cov_ssc_oc_10d[0, 0, 1, 0, :, :, :, :, :, :] = self.cov_ssc[4][:, :, 0, 0, :, :, :, :]
        self.cov_ssc_oc_10d[0, 0, 0, 0, :, :, :, :, :, :] = self.cov_ssc[5][:, :, 0, 0, :, :, :, :]

        self.cov_cng_oc_10d[1, 1, 1, 1, :, :, :, :, :, :] = self.cov_ssc[0][:, :, 0, 0, :, :, :, :]
        self.cov_cng_oc_10d[1, 1, 1, 0, :, :, :, :, :, :] = self.cov_ssc[1][:, :, 0, 0, :, :, :, :]
        self.cov_cng_oc_10d[1, 1, 0, 0, :, :, :, :, :, :] = self.cov_ssc[2][:, :, 0, 0, :, :, :, :]
        self.cov_cng_oc_10d[1, 0, 1, 0, :, :, :, :, :, :] = self.cov_ssc[3][:, :, 0, 0, :, :, :, :]
        self.cov_cng_oc_10d[0, 0, 1, 0, :, :, :, :, :, :] = self.cov_ssc[4][:, :, 0, 0, :, :, :, :]
        self.cov_cng_oc_10d[0, 0, 0, 0, :, :, :, :, :, :] = self.cov_ssc[5][:, :, 0, 0, :, :, :, :]

        # transpose to get the remaining blocks
        # ell1 <-> ell2 and zi, zj <-> zk, zl, but ell1 <-> ell2 should have no effect!
        self.cov_ssc_oc_10d[0, 0, 1, 1, :, :, :, :, :, :] = self.cov_ssc_oc_10d[1,
                                                                                1, 0, 0, :, :, :, :, :, :].transpose(1, 0, 4, 5, 2, 3)
        self.cov_ssc_oc_10d[1, 0, 1, 1, :, :, :, :, :, :] = self.cov_ssc_oc_10d[1,
                                                                                1, 1, 0, :, :, :, :, :, :].transpose(1, 0, 4, 5, 2, 3)
        self.cov_ssc_oc_10d[1, 0, 0, 0, :, :, :, :, :, :] = self.cov_ssc_oc_10d[0,
                                                                                0, 1, 0, :, :, :, :, :, :].transpose(1, 0, 4, 5, 2, 3)

        self.cov_cng_oc_10d[0, 0, 1, 1, :, :, :, :, :, :] = self.cov_cng_oc_10d[1,
                                                                                1, 0, 0, :, :, :, :, :, :].transpose(1, 0, 4, 5, 2, 3)
        self.cov_cng_oc_10d[1, 0, 1, 1, :, :, :, :, :, :] = self.cov_cng_oc_10d[1,
                                                                                1, 1, 0, :, :, :, :, :, :].transpose(1, 0, 4, 5, 2, 3)
        self.cov_cng_oc_10d[1, 0, 0, 0, :, :, :, :, :, :] = self.cov_cng_oc_10d[0,
                                                                                0, 1, 0, :, :, :, :, :, :].transpose(1, 0, 4, 5, 2, 3)

        for zi, zj, zk, zl in itertools.product(range(2), repeat=4):
            np.testing.assert_allclose(self.cov_ssc_oc_10d[zi, zj, zk, zl, :, :, :, :, :, :].transpose(1, 0, 4, 5, 2, 3),
                                       self.cov_ssc_oc_10d[zi, zj, zk, zl, :, :, :, :, :, :].transpose(0, 1, 4, 5, 2, 3), atol=0, rtol=1e-5)

    def _reshape_oc_output(self, variable_specs, ind_dict, symmetrize_output_dict):
        """
        Reshape the output of the OneCovariance (OC) calculation into a dictionary or array format.

        This function takes the raw output from the OC calculation and reshapes it into a more
        convenient format for further processing. It supports both 8D and 10D output formats,
        and can return the data as either a dictionary or a numpy array.

        The function also performs some additional processing, such as symmetrizing the output
        dictionary and saving the reshaped covariance matrices to compressed numpy files.
        """

        zpairs_auto, zpairs_cross, zpairs_3x2pt = mm.get_zpairs(self.zbins)

        ind_auto = ind_dict['G', 'G']
        ind_cross = ind_dict['G', 'L']

        chunk_size = 5000000
        cov_nbl = variable_specs['nbl_3x2pt']
        # column_names = [
        #     '#obs', 'ell1', 'ell2', 's1', 's2', 'tomoi', 'tomoj', 'tomok', 'tomol',
        #     'cov', 'covg sva', 'covg mix', 'covg sn', 'covng', 'covssc',
        # ]

        # check that column names are correct
        with open(f'{self.oc_path}/covariance_list.dat', 'r') as file:
            header = file.readline().strip()  # Read the first line and strip newline characters
        print('.dat file header: ')
        print(header)
        header_list = re.split('\t', header.strip().replace('\t\t', '\t').replace('\t\t', '\t'))

        # assert column_names == header_list, 'column names from .dat file do not match with the expected ones'

        column_names = header_list

        # ell values actually used in OC; save in self to be able to compare to the SB ell values
        # note use delim_whitespace=True instead of sep='\s+' if this gives compatibility issues
        self.ells_oc_load = pd.read_csv(f'{self.oc_path}/covariance_list.dat',
                                        usecols=['ell1'], sep='\s+')['ell1'].unique()

        # check if the saved ells are within 1% of the required ones; I think the saved values are truncated to only
        # 2 decimals, so this is a rough comparison
        try:
            np.testing.assert_allclose(self.new_ells_oc, self.ells_oc_load, atol=0, rtol=1e-2)
        except AssertionError as err:
            print('ell values computed vs loaded for OC are not the same')
            print(err)

        cov_ell_indices = {ell_out: idx for idx, ell_out in enumerate(self.ells_oc_load)}

        probe_idx_dict = {
            'm': 0,
            'g': 1,
        }

        # ! import .list covariance file
        cov_g_oc_10d = np.zeros((2, 2, 2, 2, cov_nbl, cov_nbl, self.zbins, self.zbins, self.zbins, self.zbins))
        cov_sva_oc_10d = np.zeros((2, 2, 2, 2, cov_nbl, cov_nbl, self.zbins, self.zbins, self.zbins, self.zbins))
        cov_mix_oc_10d = np.zeros((2, 2, 2, 2, cov_nbl, cov_nbl, self.zbins, self.zbins, self.zbins, self.zbins))
        cov_sn_oc_10d = np.zeros((2, 2, 2, 2, cov_nbl, cov_nbl, self.zbins, self.zbins, self.zbins, self.zbins))
        cov_ssc_oc_10d = np.zeros((2, 2, 2, 2, cov_nbl, cov_nbl, self.zbins, self.zbins, self.zbins, self.zbins))
        cov_cng_oc_10d = np.zeros((2, 2, 2, 2, cov_nbl, cov_nbl, self.zbins, self.zbins, self.zbins, self.zbins))
        cov_tot_oc_10d = np.zeros((2, 2, 2, 2, cov_nbl, cov_nbl, self.zbins, self.zbins, self.zbins, self.zbins))

        print('loading dataframe in chunks...')
        start = time.perf_counter()
        for df_chunk in pd.read_csv(f'{self.oc_path}/covariance_list.dat', sep='\s+', names=column_names, skiprows=1, chunksize=chunk_size):

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
                warnings.warn('tomo indices seem to start from 0...')
                z_indices = df_chunk[['tomoi', 'tomoj', 'tomok', 'tomol']].values

            # Vectorized assignment to the arrays
            index_tuple = (probe_idx_a, probe_idx_b, probe_idx_c, probe_idx_d, ell1_idx, ell2_idx,
                           z_indices[:, 0], z_indices[:, 1], z_indices[:, 2], z_indices[:, 3])

            cov_sva_oc_10d[index_tuple] = df_chunk['covg sva'].values
            cov_mix_oc_10d[index_tuple] = df_chunk['covg mix'].values
            cov_sn_oc_10d[index_tuple] = df_chunk['covg sn'].values
            cov_g_oc_10d[index_tuple] = df_chunk['covg sva'].values + \
                df_chunk['covg mix'].values + df_chunk['covg sn'].values
            cov_ssc_oc_10d[index_tuple] = df_chunk['covssc'].values
            cov_cng_oc_10d[index_tuple] = df_chunk['covng'].values
            cov_tot_oc_10d[index_tuple] = df_chunk['cov'].values

        print(f"df loaded in {time.perf_counter() - start:.2f} seconds")

        # ! load 2d covs for CLOE runs
        cov_oc_10d_dict = {'SVA': cov_sva_oc_10d,
                           'MIX': cov_mix_oc_10d,
                           'SN': cov_sn_oc_10d,
                           'G': cov_g_oc_10d,
                           'SSC': cov_ssc_oc_10d,
                           'cNG': cov_cng_oc_10d,
                           'tot': cov_tot_oc_10d,
                           }

        for cov_term in cov_oc_10d_dict.keys():

            print(f'working on {cov_term}')

            cov_10d = cov_oc_10d_dict[cov_term]

            cov_llll_4d = mm.cov_6D_to_4D_blocks(cov_10d[0, 0, 0, 0, ...], cov_nbl,
                                                 zpairs_auto, zpairs_auto, ind_auto, ind_auto)
            cov_llgl_4d = mm.cov_6D_to_4D_blocks(cov_10d[0, 0, 1, 0, ...], cov_nbl,
                                                 zpairs_auto, zpairs_cross, ind_auto, ind_cross)
            cov_ggll_4d = mm.cov_6D_to_4D_blocks(cov_10d[1, 1, 0, 0, ...], cov_nbl,
                                                 zpairs_auto, zpairs_auto, ind_auto, ind_auto)
            cov_glgl_4d = mm.cov_6D_to_4D_blocks(cov_10d[1, 0, 1, 0, ...], cov_nbl,
                                                 zpairs_cross, zpairs_cross, ind_cross, ind_cross)
            cov_gggl_4d = mm.cov_6D_to_4D_blocks(cov_10d[1, 1, 1, 0, ...], cov_nbl,
                                                 zpairs_auto, zpairs_cross, ind_auto, ind_cross)
            cov_gggg_4d = mm.cov_6D_to_4D_blocks(cov_10d[1, 1, 1, 1, ...], cov_nbl,
                                                 zpairs_auto, zpairs_auto, ind_auto, ind_auto)

            cov_llgg_4d = np.transpose(cov_ggll_4d, (1, 0, 3, 2))
            cov_glll_4d = np.transpose(cov_llgl_4d, (1, 0, 3, 2))
            cov_glgg_4d = np.transpose(cov_gggl_4d, (1, 0, 3, 2))

            cov_oc_10d_dict[cov_term][0, 0, 1, 1] = mm.cov_4D_to_6D_blocks(cov_llgg_4d, cov_nbl, self.zbins, ind_auto, ind_auto,
                                                                           symmetrize_output_dict['L', 'L'], symmetrize_output_dict['G', 'G'])
            cov_oc_10d_dict[cov_term][1, 0, 0, 0] = mm.cov_4D_to_6D_blocks(cov_glll_4d, cov_nbl, self.zbins, ind_cross, ind_auto,
                                                                           symmetrize_output_dict['G', 'L'], symmetrize_output_dict['L', 'L'])
            cov_oc_10d_dict[cov_term][1, 0, 1, 1] = mm.cov_4D_to_6D_blocks(cov_glgg_4d, cov_nbl, self.zbins, ind_cross, ind_auto,
                                                                           symmetrize_output_dict['G', 'L'], symmetrize_output_dict['G', 'G'])

            np.savez_compressed(
                f'{self.oc_path}/{self.cov_filename.format(which_ng_cov=cov_term, probe_a="L", probe_b="L", probe_c="L", probe_d="L")}', cov_llll_4d)
            np.savez_compressed(
                f'{self.oc_path}/{self.cov_filename.format(which_ng_cov=cov_term, probe_a="L", probe_b="L", probe_c="G", probe_d="L")}', cov_llgl_4d)
            np.savez_compressed(
                f'{self.oc_path}/{self.cov_filename.format(which_ng_cov=cov_term, probe_a="L", probe_b="L", probe_c="G", probe_d="G")}', cov_llgg_4d)
            np.savez_compressed(
                f'{self.oc_path}/{self.cov_filename.format(which_ng_cov=cov_term, probe_a="G", probe_b="L", probe_c="G", probe_d="L")}', cov_glgl_4d)
            np.savez_compressed(
                f'{self.oc_path}/{self.cov_filename.format(which_ng_cov=cov_term, probe_a="G", probe_b="L", probe_c="G", probe_d="G")}', cov_glgg_4d)
            np.savez_compressed(
                f'{self.oc_path}/{self.cov_filename.format(which_ng_cov=cov_term, probe_a="G", probe_b="G", probe_c="G", probe_d="G")}', cov_gggg_4d)

            del cov_llll_4d, cov_llgl_4d, cov_llgg_4d, cov_glgl_4d, cov_glgg_4d, cov_gggg_4d, cov_ggll_4d, cov_glll_4d, cov_gggl_4d
            gc.collect()

    def oc_output_to_dict_or_array(self, which_ng_cov, output_type, ind_dict=None, symmetrize_output_dict=None):

        # import
        filename = self.cov_filename.format(which_ng_cov=which_ng_cov,
                                            probe_a='{probe_a:s}',
                                            probe_b='{probe_b:s}',
                                            probe_c='{probe_c:s}',
                                            probe_d='{probe_d:s}')
        cov_ng_oc_3x2pt_dict_8D = mm.load_cov_from_probe_blocks(path=self.oc_path,
                                                                filename=filename,
                                                                probe_ordering=self.cfg['covariance']['probe_ordering'])

        # reshape
        if output_type == '8D_dict':
            return cov_ng_oc_3x2pt_dict_8D

        elif output_type in ['10D_dict', '10D_array']:
            cov_ng_oc_3x2pt_dict_10D = mm.cov_3x2pt_dict_8d_to_10d(
                cov_3x2pt_dict_8D=cov_ng_oc_3x2pt_dict_8D,
                nbl=self.variable_specs['nbl_3x2pt'],
                zbins=self.zbins,
                ind_dict=ind_dict,
                probe_ordering=self.cfg['covariance']['probe_ordering'],
                symmetrize_output_dict=symmetrize_output_dict)

            if output_type == '10D_dict':
                return cov_ng_oc_3x2pt_dict_10D

            elif output_type == '10D_array':
                return mm.cov_10D_dict_to_array(cov_ng_oc_3x2pt_dict_10D,
                                                nbl=self.variable_specs['nbl_3x2pt'],
                                                zbins=self.zbins,
                                                n_probes=self.cfg['covariance']['n_probes'])

        else:
            raise ValueError('output_dict_dim must be 8D or 10D')

    def find_optimal_ellmax_oc(self, target_ell_array):
        # Perform the minimization
        result = minimize_scalar(self.objective_function, bounds=[2000, 9000], method='bounded')

        # Check the result
        if result.success:
            self.optimal_ellmax = result.x
            print(f"Optimal ellmax found: {self.optimal_ellmax}")
        else:
            print("Optimization failed.")

        self.new_ells_oc = self.compute_ells_oc(nbl=int(self.variable_specs['nbl_3x2pt']),
                                                ell_min=float(self.variable_specs['ell_min']),
                                                ell_max=self.optimal_ellmax)

        fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        ax[0].plot(target_ell_array, label='target ells (SB)', marker='o', alpha=.6)
        ax[0].plot(self.new_ells_oc, label='ells OC', marker='o', alpha=.6)
        ax[1].plot(mm.percent_diff(target_ell_array, self.new_ells_oc), label='% diff', marker='o')

        ax[0].legend()
        ax[1].legend()
        ax[0].set_ylabel('$\\ell$')
        ax[1].set_ylabel('% diff')
        fig.supxlabel('ell idx')

    def compute_ells_oc(self, nbl, ell_min, ell_max):
        ell_bin_edges_oc_int = np.unique(np.geomspace(ell_min, ell_max, nbl + 1)).astype(int)
        ells_oc_int = np.exp(.5 * (np.log(ell_bin_edges_oc_int[1:])
                                   + np.log(ell_bin_edges_oc_int[:-1])))  # it's the same if I take base 10 log
        return ells_oc_int

    def objective_function(self, ell_max):
        ells_oc = self.compute_ells_oc(nbl=int(self.variable_specs['nbl_3x2pt']),
                                       ell_min=float(self.variable_specs['ell_min']),
                                       ell_max=ell_max)
        ssd = np.sum((self.ells_sb - ells_oc) ** 2)
        # ssd = np.sum(mm.percent_diff(self.ells_sb, ells_oc)**2)  # TODO test this
        return ssd
