import numpy as np
import sys
sys.path.append('/home/cosmo/davide.sciotti/data/Spaceborne/')
import bin.my_module as mm

path = '/home/cosmo/davide.sciotti/data/common_data/Spaceborne/jobs/SPV3/output/Flagship_2/covmat/PyCCL/jan_2024'
probe_combinations = (('L', 'L'), ('G', 'L'), ('G', 'G'))
nbl = 29

for a, b in probe_combinations:
    for c, d in probe_combinations:
        print(a, b, c, d)
        cov_dense = np.load(
            f'{path}/cov_SSC_pyccl_{a}{b}{c}{d}_4D_nbl32_ellmax5000_zbinsEP13_sigma2_None_densegrids.npz')['arr_0']
        cov_dense_sigmanone = np.load(
            f'{path}/cov_SSC_pyccl_{a}{b}{c}{d}_4D_nbl29_ellmax3000_zbinsEP13_densegrids.npz')['arr_0']


        cov_dense_sigmanone = cov_dense_sigmanone[:nbl, :nbl, :, :]
        zpairs_ab = cov_dense.shape[2]
        zpairs_cd = cov_dense.shape[3]

        mm.compare_arrays(cov_dense.reshape((nbl * zpairs_ab, nbl * zpairs_cd)),
                          cov_dense_sigmanone.reshape((nbl * zpairs_ab, nbl * zpairs_cd)))
