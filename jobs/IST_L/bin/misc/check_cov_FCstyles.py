import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as spar

probe = '3x2pt'
for NL_flag in range(1, 4):
    Cstyle = spar.load_npz(f'/Users/davide/Documents/Lavoro/Programmi/SSC_restructured/jobs/IST_NL/output/covmat/CovMat'
                           f'-{probe}-Gauss-20bins-NL_flag_{NL_flag + 1}-2D-Cstyle-Sparse.npz').toarray()
    Fstyle = spar.load_npz(f'/Users/davide/Documents/Lavoro/Programmi/SSC_restructured/jobs/IST_NL/output/covmat/CovMat'
                           f'-{probe}-Gauss-20bins-NL_flag_{NL_flag + 1}-2D-Fstyle-Sparse.npz').toarray()

    plt.matshow(np.log10(Cstyle))
    plt.title('Cstyle')
    plt.matshow(np.log10(Fstyle))
    plt.title('Fstyle')
