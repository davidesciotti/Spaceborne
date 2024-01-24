import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
from joblib import Parallel, delayed


import os
ROOT = os.getenv('ROOT')
SB_ROOT = f'{ROOT}/Spaceborne'

# project modules
sys.path.append(SB_ROOT)
import bin.my_module as mm


input_extension = 'dat'
output_extension = 'npz'


def load_save_func(folder_path, filename, input_extension, output_extension):
    file = load_func(f'{folder_path}/{filename}')
    filename = filename.replace('.' + input_extension, '')
    save_func(f'{folder_path}/{filename}.{output_extension}', file)
    print(f'File {folder_path}/{filename}.{output_extension} saved')


if input_extension == 'dat':
    load_func = np.genfromtxt
elif input_extension == 'npz' or input_extension == 'npy':
    load_func = np.load
else:
    raise ValueError('Input extension not recognized')

if output_extension == 'dat':
    save_func = np.savetxt
elif output_extension == 'npy':
    save_func = np.save
elif output_extension == 'npz':
    save_func = np.savez_compressed
else:
    raise ValueError('Output extension not recognized')


folder_path = '/home/davide/Documenti/Lavoro/Programmi/common_data/vincenzo/SPV3_07_2022/LiFEforSPV3/OutputFiles/CovMats/GaussOnly/Full'
filenames = mm.get_filenames_in_folder(folder_path)

start_time = time.perf_counter()

Parallel(n_jobs=1)(delayed(load_save_func)(folder_path, filename,
                                           input_extension, output_extension) for filename in filenames)

print(f'Elapsed time: {time.perf_counter() - start_time:.2f} s')
