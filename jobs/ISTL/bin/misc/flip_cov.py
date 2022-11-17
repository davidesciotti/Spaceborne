import sys
import time
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

project_path = Path.cwd().parent.parent.parent
job_path = Path.cwd().parent

sys.path.append(f'{project_path.parent.parent}/common_data/common_lib')
import my_module as mm
import cosmo_lib as csmlib

sys.path.append(f'{project_path.parent.parent}/common_data/common_config')
import ISTF_fid_params
import mpl_cfg

matplotlib.use('Qt5Agg')
plt.rcParams.update(mpl_cfg.mpl_rcParams_dict)
start_time = time.perf_counter()

########################################################################################################################

# ! settings
zbins = 10
# ! end settings

zpairs_auto, zpairs_cross, zpairs_3x2pt = mm.get_pairs(zbins)

cov_path = f'/Users/davide/Documents/Lavoro/Programmi/likelihood-implementation/data/ExternalBenchmark/Photometric/data'
cov_WL_2D = np.load(f'{cov_path}/CovMat-ShearShear-Gauss-20Bins.npy')
cov_GC_2D = np.load(f'{cov_path}/CovMat-PosPos-Gauss-20Bins.npy')
cov_3x2pt_2D = np.load(f'{cov_path}/CovMat-3x2pt-Gauss-20Bins.npy')

cov_WL_4D = mm.cov_2D_to_4D(cov_WL_2D, nbl=20, block_index='vincenzo')
cov_GC_4D = mm.cov_2D_to_4D(cov_GC_2D, nbl=20, block_index='vincenzo')
cov_3x2pt_4D = mm.cov_2D_to_4D(cov_3x2pt_2D, nbl=20, block_index='vincenzo')

cov_WL_2Dflip = mm.cov_4D_to_2D_old_2(cov_WL_4D, nbl=20, zpairs_AB=zpairs_auto, zpairs_CD=zpairs_auto, block_index='ij')
cov_GC_2Dflip = mm.cov_4D_to_2D_old_2(cov_GC_4D, nbl=20, zpairs_AB=zpairs_auto, zpairs_CD=zpairs_auto, block_index='ij')
cov_3x2pt_2Dflip = mm.cov_4D_to_2D_old_2(cov_3x2pt_4D, nbl=20, zpairs_AB=zpairs_3x2pt, zpairs_CD=zpairs_3x2pt, block_index='ij')

cov_WL_2Dflip_new = mm.cov_4D_to_2D(cov_WL_4D, block_index='ij')
cov_GC_2Dflip_new = mm.cov_4D_to_2D(cov_GC_4D, block_index='ij')
cov_3x2pt_2Dflip_new = mm.cov_4D_to_2D(cov_3x2pt_4D, block_index='ij')

assert np.array_equal(cov_WL_2Dflip, cov_WL_2Dflip_new)
assert np.array_equal(cov_GC_2Dflip, cov_GC_2Dflip_new)
assert np.array_equal(cov_3x2pt_2Dflip, cov_3x2pt_2Dflip_new)



np.save(f'{cov_path}/CovMat-ShearShear-Gauss-20Bins-zpair_blocks.npy', cov_WL_2Dflip)
np.save(f'{cov_path}/CovMat-PosPos-Gauss-20Bins-zpair_blocks.npy', cov_GC_2Dflip)
np.save(f'{cov_path}/CovMat-3x2pt-Gauss-20Bins-zpair_blocks.npy', cov_3x2pt_2Dflip)

mm.matshow(cov_3x2pt_2D, title='cov_3x2pt_2D', log=True)
mm.matshow(cov_3x2pt_2Dflip, title='cov_3x2pt_2Dflip', log=True)
