import os
import multiprocessing
num_cores = multiprocessing.cpu_count()
os.environ['OMP_NUM_THREADS'] = '32'
os.environ['NUMBA_NUM_THREADS'] = '32'
os.environ['NUMBA_PARALLEL_DIAGNOSTICS'] = '4'

from tqdm import tqdm
from functools import partial
from collections import OrderedDict
import numpy as np
import time
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import configparser
import warnings
import gc
import yaml
import pprint
from copy import deepcopy
import numpy.testing as npt
from scipy.interpolate import interp1d, RegularGridInterpolator
from tabulate import tabulate


import spaceborne.ell_utils as ell_utils
import spaceborne.cl_preprocessing as cl_utils
import spaceborne.compute_Sijkl as Sijkl_utils
import spaceborne.covariance as covmat_utils
import spaceborne.fisher_matrix as fm_utils
import spaceborne.my_module as mm
import spaceborne.cosmo_lib as cosmo_lib
import spaceborne.wf_cl_lib as wf_cl_lib
import spaceborne.pyccl_cov_class as pyccl_cov_class
import spaceborne.plot_lib as plot_lib
import spaceborne.sigma2_SSC as sigma2_SSC
import subprocess


class OneCovarianceInterface():

    def __init__(self, ROOT, cfg, variable_specs):
        self.cfg = cfg
        self.variable_specs = variable_specs

        # paths
        self.ROOT = ROOT
        self.conda_base_path = self.get_conda_base_path()
        self.oc_path = cfg['covariance_cfg']['OneCovariance_cfg']['onecovariance_folder'].format(
            ROOT=self.ROOT, **self.variable_specs)
        self.path_to_oc_executable = cfg['covariance_cfg']['OneCovariance_cfg']['path_to_oc_executable'].format(
            ROOT=ROOT)
        self.path_to_config_oc_ini = f'{self.oc_path }/input_configs.ini'

    def get_conda_base_path(self):
        try:
            # Run the conda info --base command and capture the output
            result = subprocess.run(['conda', 'info', '--base'], stdout=subprocess.PIPE, check=True, text=True)
            # Extract and return the base path
            return result.stdout.strip() + '/bin'
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
        nofz_filename_ascii = ascii_filenames_dict['nofz_ascii_filename']
        gal_bias_ascii_filename = ascii_filenames_dict['gal_bias_ascii_filename']

        # TODO import another file??
        # Read the existing reference .ini file
        cfg_onecov_ini = CaseConfigParser()
        cfg_onecov_ini.read(f'{self.ROOT}/OneCovariance/config_files_dav/config_3x2pt_pure_Cell_dav.ini')
        zbins = self.cfg['general_cfg']['zbins']
        general_cfg = self.cfg['general_cfg']

        # set useful lists
        mult_shear_bias_list = [self.cfg['cosmology']['FM_ordered_params'][f'm{zi:02d}'] for zi in range(1, zbins + 1)]
        n_eff_clust_list = self.cfg['covariance_cfg']['ngal_clustering']
        n_eff_lensing_list = self.cfg['covariance_cfg']['ngal_lensing']
        ellipticity_dispersion_list = [self.cfg['covariance_cfg']['sigma_eps_i']] * zbins

        cfg_onecov_ini['covariance terms']['nongauss'] = str(True)
        cfg_onecov_ini['output settings']['directory'] = self.oc_path

        cfg_onecov_ini['covELLspace settings']['mult_shear_bias'] = ', '.join(map(str, mult_shear_bias_list))
        cfg_onecov_ini['covELLspace settings']['ell_min'] = str(general_cfg['ell_min'])
        cfg_onecov_ini['covELLspace settings']['ell_max'] = str(
            general_cfg['ell_max_3x2pt'])  # TODO slightly different ell_max for 3000?
        cfg_onecov_ini['covELLspace settings']['ell_bins'] = str(general_cfg['nbl_3x2pt'])

        cfg_onecov_ini['survey specs']['mask_directory'] = '/home/cosmo/davide.sciotti/data/common_data/mask/'
        cfg_onecov_ini['survey specs']['which_cov_binning'] = 'OneCovariance'  # TODO test diff with EC20

        # TODO import this in a cleaner way
        cfg_onecov_ini['survey specs']['mask_file_clust'] = 'mask_circular_1pole_14700deg2_nside2048.fits'
        cfg_onecov_ini['survey specs']['mask_file_lensing'] = 'mask_circular_1pole_14700deg2_nside2048.fits'
        cfg_onecov_ini['survey specs']['mask_file_ggl'] = 'mask_circular_1pole_14700deg2_nside2048.fits'
        cfg_onecov_ini['survey specs']['n_eff_clust'] = ', '.join(map(str, n_eff_clust_list))
        cfg_onecov_ini['survey specs']['n_eff_lensing'] = ', '.join(map(str, n_eff_lensing_list))
        cfg_onecov_ini['survey specs']['ellipticity_dispersion'] = ', '.join(map(str, ellipticity_dispersion_list))

        cfg_onecov_ini['redshift']['z_directory'] = self.oc_path
        cfg_onecov_ini['redshift']['zclust_file'] = nofz_filename_ascii
        cfg_onecov_ini['redshift']['zlens_file'] = nofz_filename_ascii

        cfg_onecov_ini['cosmo']['h'] = str(self.cfg['cosmology']['FM_ordered_params']['h'])
        cfg_onecov_ini['cosmo']['ns'] = str(self.cfg['cosmology']['FM_ordered_params']['ns'])
        cfg_onecov_ini['cosmo']['omega_m'] = str(self.cfg['cosmology']['FM_ordered_params']['Om'])
        cfg_onecov_ini['cosmo']['omega_b'] = str(self.cfg['cosmology']['FM_ordered_params']['Ob'])
        cfg_onecov_ini['cosmo']['omega_de'] = str(self.cfg['cosmology']['other_params']['ODE'])
        cfg_onecov_ini['cosmo']['sigma8'] = str(self.cfg['cosmology']['FM_ordered_params']['s8'])
        cfg_onecov_ini['cosmo']['w0'] = str(self.cfg['cosmology']['FM_ordered_params']['wz'])
        cfg_onecov_ini['cosmo']['wa'] = str(self.cfg['cosmology']['FM_ordered_params']['wa'])
        cfg_onecov_ini['cosmo']['neff'] = str(self.cfg['cosmology']['other_params']['N_eff'])
        cfg_onecov_ini['cosmo']['m_nu'] = str(self.cfg['cosmology']['other_params']['m_nu'])

        cfg_onecov_ini['bias']['bias_files'] = gal_bias_ascii_filename

        cfg_onecov_ini['IA']['A_IA'] = str(self.cfg['cosmology']['FM_ordered_params']['Aia'])
        cfg_onecov_ini['IA']['eta_IA'] = str(self.cfg['cosmology']['FM_ordered_params']['eIA'])

        cfg_onecov_ini['powspec evaluation']['non_linear_model'] = str(
            self.cfg['cosmology']['other_params']['camb_extra_parameters']['camb']['halofit_version'])
        cfg_onecov_ini['powspec evaluation']['HMCode_logT_AGN'] = str(
            self.cfg['cosmology']['other_params']['camb_extra_parameters']['camb']['HMCode_logT_AGN'])

        cfg_onecov_ini['tabulated inputs files']['Cell_directory'] = self.oc_path
        cfg_onecov_ini['tabulated inputs files']['Cmm_file'] = f'{cl_ll_oc_filename}.ascii'
        cfg_onecov_ini['tabulated inputs files']['Cgm_file'] = f'{cl_gl_oc_filename}.ascii'
        cfg_onecov_ini['tabulated inputs files']['Cgg_file'] = f'{cl_gg_oc_filename}.ascii'

        cfg_onecov_ini['misc']['num_cores'] = str(general_cfg['num_threads'])

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

    def call_onecovariance(self):
        """This function runs OneCovariance and reshapes the output"""

        activate_and_run = f"""
        source {self.conda_base_path}/activate cov20_env
        python {self.path_to_oc_executable} {self.path_to_config_oc_ini}
        source {self.conda_base_path}/deactivate
        source {self.conda_base_path}/activate spaceborne-dav
        python {self.path_to_oc_executable.replace('covariance.py', 'reshape_cov_list_Cl_callable.py')} {self.path_to_config_oc_ini.replace('input_configs.ini', '')}
        """

        process = subprocess.Popen(activate_and_run, shell=True, executable='/bin/bash')
        process.communicate()

    def oc_output_to_dict_or_array(self, which_ng_cov, output_type, ind_dict=None, symmetrize_output_dict=None):
        variable_specs = deepcopy(self.variable_specs)
        variable_specs.pop('which_ng_cov')
        filename = self.cfg['covariance_cfg']['OneCovariance_cfg']['cov_filename'].format(ROOT=self.ROOT,
                                                                                          which_ng_cov=which_ng_cov,
                                                                                          probe_a='{probe_a:s}',
                                                                                          probe_b='{probe_b:s}',
                                                                                          probe_c='{probe_c:s}',
                                                                                          probe_d='{probe_d:s}',
                                                                                          nbl=variable_specs['nbl_3x2pt'],
                                                                                          lmax=variable_specs['ell_max_3x2pt'],
                                                                                          **variable_specs)
        cov_ng_oc_3x2pt_dict_8D = mm.load_cov_from_probe_blocks(path=self.oc_path,
                                                                filename=filename,
                                                                probe_ordering=self.cfg['covariance_cfg']['probe_ordering'])

        if output_type == '8D_dict':
            return cov_ng_oc_3x2pt_dict_8D

        elif output_type in ['10D_dict', '10D_array']:
            cov_ng_oc_3x2pt_dict_10D = mm.cov_3x2pt_dict_8d_to_10d(
                cov_3x2pt_dict_8D=cov_ng_oc_3x2pt_dict_8D,
                nbl=variable_specs['nbl_3x2pt'],
                zbins=self.cfg['general_cfg']['zbins'],
                ind_dict=ind_dict,
                probe_ordering=self.cfg['covariance_cfg']['probe_ordering'],
                symmetrize_output_dict=symmetrize_output_dict)

            if output_type == '10D_dict':
                return cov_ng_oc_3x2pt_dict_10D

            elif output_type == '10D_array':
                return mm.cov_10D_dict_to_array(cov_ng_oc_3x2pt_dict_10D,
                                                nbl=variable_specs['nbl_3x2pt'],
                                                zbins=self.cfg['general_cfg']['zbins'],
                                                n_probes=self.cfg['general_cfg']['n_probes'])

        else:
            raise ValueError('output_dict_dim must be 8D or 10D')
