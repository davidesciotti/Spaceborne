import gc
import os
import multiprocessing
import re
num_cores = multiprocessing.cpu_count()
os.environ['OMP_NUM_THREADS'] = '32'
os.environ['NUMBA_NUM_THREADS'] = '32'
os.environ['NUMBA_PARALLEL_DIAGNOSTICS'] = '4'

import numpy as np
import time
import configparser
import warnings
from copy import deepcopy
import pandas as pd

import spaceborne.ell_utils as ell_utils
import spaceborne.my_module as mm
import subprocess


class OneCovarianceInterface():

    def __init__(self, ROOT, cfg, variable_specs):
        self.cfg = cfg
        self.variable_specs = variable_specs
        self.which_gauss_cov_binning = self.cfg['covariance_cfg']['OneCovariance_cfg']['which_gauss_cov_binning']
        self.zbins = self.cfg['general_cfg']['zbins']

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
        gal_bias_ascii_filename = ascii_filenames_dict['gal_bias_ascii_filename']

        # TODO import another file??
        # Read the existing reference .ini file
        cfg_onecov_ini = CaseConfigParser()
        cfg_onecov_ini.read(f'{self.ROOT}/OneCovariance/config_files_dav/config_3x2pt_pure_Cell_dav.ini')
        general_cfg = self.cfg['general_cfg']

        # set useful lists
        mult_shear_bias_list = [self.cfg['cosmology']['FM_ordered_params']
                                [f'm{zi:02d}'] for zi in range(1, self.zbins + 1)]
        n_eff_clust_list = self.cfg['covariance_cfg']['ngal_clustering']
        n_eff_lensing_list = self.cfg['covariance_cfg']['ngal_lensing']
        ellipticity_dispersion_list = [self.cfg['covariance_cfg']['sigma_eps_i']] * self.zbins

        cfg_onecov_ini['covariance terms']['gauss'] = str(True)
        cfg_onecov_ini['covariance terms']['split_gauss'] = str(True)
        cfg_onecov_ini['covariance terms']['nongauss'] = str(True)
        cfg_onecov_ini['covariance terms']['ssc'] = str(True)
        cfg_onecov_ini['output settings']['directory'] = self.oc_path

        warnings.warn('Setting same ell binning for all probes in OC')

        cfg_onecov_ini['covELLspace settings']['mult_shear_bias'] = ', '.join(map(str, mult_shear_bias_list))
        cfg_onecov_ini['covELLspace settings']['ell_min'] = str(general_cfg['ell_min'])
        cfg_onecov_ini['covELLspace settings']['ell_min_clustering'] = str(general_cfg['ell_min'])
        cfg_onecov_ini['covELLspace settings']['ell_min_lensing'] = str(general_cfg['ell_min'])
        # TODO slightly different ell_max for 3000?
        cfg_onecov_ini['covELLspace settings']['ell_max'] = str(general_cfg['ell_max_3x2pt'])
        cfg_onecov_ini['covELLspace settings']['ell_max_clustering'] = str(general_cfg['ell_max_3x2pt'])
        cfg_onecov_ini['covELLspace settings']['ell_max_lensing'] = str(general_cfg['ell_max_3x2pt'])

        cfg_onecov_ini['covELLspace settings']['ell_bins'] = str(general_cfg['nbl_3x2pt'])
        cfg_onecov_ini['covELLspace settings']['ell_bins_clustering'] = str(general_cfg['nbl_3x2pt'])
        cfg_onecov_ini['covELLspace settings']['ell_bins_lensing'] = str(general_cfg['nbl_3x2pt'])

        cfg_onecov_ini['survey specs']['mask_directory'] = '/home/cosmo/davide.sciotti/data/common_data/mask/'
        cfg_onecov_ini['survey specs']['mask_file_clust'] = ''
        cfg_onecov_ini['survey specs']['mask_file_lensing'] = ''
        cfg_onecov_ini['survey specs']['mask_file_ggl'] = ''
        # TODO test diff with EC20 binning
        cfg_onecov_ini['survey specs']['which_cov_binning'] = self.which_gauss_cov_binning

        cfg_onecov_ini['survey specs']['survey_area_lensing_in_deg2'] = str(
            self.cfg['covariance_cfg']['survey_area_deg2'])
        cfg_onecov_ini['survey specs']['survey_area_ggl_in_deg2'] = str(
            self.cfg['covariance_cfg']['survey_area_deg2'])
        cfg_onecov_ini['survey specs']['survey_area_clust_in_deg2'] = str(
            self.cfg['covariance_cfg']['survey_area_deg2'])
        cfg_onecov_ini['survey specs']['n_eff_clust'] = ', '.join(map(str, n_eff_clust_list))
        cfg_onecov_ini['survey specs']['n_eff_lensing'] = ', '.join(map(str, n_eff_lensing_list))
        cfg_onecov_ini['survey specs']['ellipticity_dispersion'] = ', '.join(map(str, ellipticity_dispersion_list))

        cfg_onecov_ini['redshift']['z_directory'] = self.oc_path
        # TODO re-check that the OC documentation is correct
        cfg_onecov_ini['redshift']['zclust_file'] = nz_lns_filename_ascii
        cfg_onecov_ini['redshift']['zlens_file'] = nz_src_filename_ascii

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

        # ! precision settings
        if self.cfg['covariance_cfg']['OneCovariance_cfg']['high_precision']:
            delta_z = 0.04
            tri_delta_z = 0.25
            integration_steps = 1000
            m_bins = 1500  # 1500
            log10k_bins = 200  # 200
        else:  # these are the default values
            delta_z = 0.08
            tri_delta_z = 0.5
            integration_steps = 500
            m_bins = 900
            log10k_bins = 100

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

    def call_onecovariance(self):
        """This function runs OneCovariance and reshapes the output"""

        activate_and_run = f"""
        source {self.conda_base_path}/activate cov20_env
        python {self.path_to_oc_executable} {self.path_to_config_oc_ini}
        source {self.conda_base_path}/deactivate
        source {self.conda_base_path}/activate spaceborne-dav
        """
        # python {self.path_to_oc_executable.replace('covariance.py', 'reshape_cov_list_Cl_callable.py')} {self.path_to_config_oc_ini.replace('input_configs.ini', '')}

        process = subprocess.Popen(activate_and_run, shell=True, executable='/bin/bash')
        process.communicate()

    def reshape_oc_output(self, variable_specs, ind_dict, symmetrize_output_dict):

        zpairs_auto, zpairs_cross, zpairs_3x2pt = mm.get_zpairs(self.zbins)

        probe_ordering = self.cfg['covariance_cfg']['probe_ordering']
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
        self.ells_oc_load = pd.read_csv(f'{self.oc_path}/covariance_list.dat',
                                        usecols=['ell1'], delim_whitespace=True)['ell1'].unique()
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
        for df_chunk in pd.read_csv(f'{self.oc_path}/covariance_list.dat', delim_whitespace=True, names=column_names, skiprows=1, chunksize=chunk_size):

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

            variable_specs = deepcopy(self.variable_specs)
            variable_specs.pop('which_ng_cov')
            cov_filename = self.cfg['covariance_cfg']['OneCovariance_cfg']['cov_filename'].format(ROOT=self.ROOT,
                                                                                                  which_ng_cov='{which_ng_cov:s}',
                                                                                                  probe_a='{probe_a:s}',
                                                                                                  probe_b='{probe_b:s}',
                                                                                                  probe_c='{probe_c:s}',
                                                                                                  probe_d='{probe_d:s}',
                                                                                                  nbl=variable_specs['nbl_3x2pt'],
                                                                                                  lmax=variable_specs['ell_max_3x2pt'],
                                                                                                  which_gauss_cov_binning=self.which_gauss_cov_binning,
                                                                                                  **variable_specs)
            np.savez_compressed(
                f'{self.oc_path}/{cov_filename.format(which_ng_cov=cov_term, probe_a="L", probe_b="L", probe_c="L", probe_d="L")}', cov_llll_4d)
            np.savez_compressed(
                f'{self.oc_path}/{cov_filename.format(which_ng_cov=cov_term, probe_a="L", probe_b="L", probe_c="G", probe_d="L")}', cov_llgl_4d)
            np.savez_compressed(
                f'{self.oc_path}/{cov_filename.format(which_ng_cov=cov_term, probe_a="L", probe_b="L", probe_c="G", probe_d="G")}', cov_llgg_4d)
            np.savez_compressed(
                f'{self.oc_path}/{cov_filename.format(which_ng_cov=cov_term, probe_a="G", probe_b="L", probe_c="G", probe_d="L")}', cov_glgl_4d)
            np.savez_compressed(
                f'{self.oc_path}/{cov_filename.format(which_ng_cov=cov_term, probe_a="G", probe_b="L", probe_c="G", probe_d="G")}', cov_glgg_4d)
            np.savez_compressed(
                f'{self.oc_path}/{cov_filename.format(which_ng_cov=cov_term, probe_a="G", probe_b="G", probe_c="G", probe_d="G")}', cov_gggg_4d)

            del cov_llll_4d, cov_llgl_4d, cov_llgg_4d, cov_glgl_4d, cov_glgg_4d, cov_gggg_4d, cov_ggll_4d, cov_glll_4d, cov_gggl_4d
            gc.collect()

    def oc_output_to_dict_or_array(self, which_ng_cov, output_type, ind_dict=None, symmetrize_output_dict=None):

        # import
        which_gauss_cov_binning = self.cfg['covariance_cfg']['OneCovariance_cfg']['which_gauss_cov_binning']
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
                                                                                          which_gauss_cov_binning=which_gauss_cov_binning,
                                                                                          **variable_specs)

        cov_ng_oc_3x2pt_dict_8D = mm.load_cov_from_probe_blocks(path=self.oc_path,
                                                                filename=filename,
                                                                probe_ordering=self.cfg['covariance_cfg']['probe_ordering'])

        # reshape
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
