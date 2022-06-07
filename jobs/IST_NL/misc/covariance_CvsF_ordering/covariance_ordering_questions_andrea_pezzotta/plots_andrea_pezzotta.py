import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Qt5Agg')

"""
produce the plots showed in the mail to Andrea Pezzotta
"""

path = '/jobs/IST_NL/output/covmat'
cov_WL_Fstyle = np.load(f'{path}/CovMat-ShearShear-Gauss-20bins-NL_flag_1-2D-Fstyle.npy')
cov_WL_Cstyle = np.load(f'{path}/CovMat-ShearShear-Gauss-20bins-NL_flag_1-2D-Cstyle.npy')

cov_3x2pt_Cstyle_concflat = np.load(f'{path}/CovMat-3x2pt-Gauss-20bins-NL_flag_1-2D-Cstyle.npy')
cov_3x2pt_Cstyle_flatconc = np.load(f'{path}/CovMat-3x2pt-Gauss-20bins-NL_flag_1_2DCLOE.npy')

# CovMat-3x2pt-Gauss-20bins-NL_flag_4_2DCLOE.npy

plt.matshow(np.log10(cov_WL_Fstyle))
plt.colorbar()
plt.title('cov_WL_Fstyle')

plt.matshow(np.log10(cov_WL_Cstyle))
plt.colorbar()
plt.title('cov_WL_Cstyle')

plt.matshow(np.log10(cov_3x2pt_Cstyle_concflat))
plt.colorbar()
plt.title('cov_3x2pt_Cstyle_concflat')

plt.matshow(np.log10(cov_3x2pt_Cstyle_flatconc))
plt.colorbar()
plt.title('cov_3x2pt_Cstyle_flatconc')

