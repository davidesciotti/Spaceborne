import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys


ROOT = '/home/davide/Documenti/Lavoro/Programmi'
SB_ROOT = f'{ROOT}/Spaceborne'

# project modules
sys.path.append(SB_ROOT)
import bin.my_module as mm


folder_path = '/home/davide/Documenti/Lavoro/Programmi/common_data/vincenzo/SPV3_07_2022/LiFEforSPV3/OutputFiles/CovMats/GaussOnly/Full'
npy_dict = dict(mm.get_kv_pairs(folder_path, extension='dat'))

for key in npy_dict.keys():
    np.savez_compressed(f'{folder_path}/{key}.npz', npy_dict[key])
    
