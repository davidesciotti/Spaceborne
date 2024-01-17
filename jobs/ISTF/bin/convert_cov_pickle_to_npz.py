"""
This is just to uniform the format used when loading the covariance matrix in the covariance.py module"""

import numpy as np
import sys
sys.path.append('/home/cosmo/davide.sciotti/data/Spaceborne')
import bin.my_module as mm


path = '/home/cosmo/davide.sciotti/data/PyCCL_SSC/output/covmat/ISTF'

filenames = mm.get_filenames_in_folder(path)

for filename in filenames:
    if filename.endswith('.pickle'):
        dictionary = mm.load_pickle(f'{path}/{filename}')
        
        for key in dictionary.keys():
            probe_id = key[0] + key[1] + key[2] + key[3] 
            
            if 'SSC' in filename:
                block_name = filename.replace('cov_PyCCL_SSC', f'cov_SSC_pyccl_{probe_id}')
            elif 'cNG' in filename:
                block_name = filename.replace('cov_PyCCL_cNG', f'cov_cNG_pyccl_{probe_id}')
            np.savez_compressed(f'{path}/{block_name}', dictionary[key])


