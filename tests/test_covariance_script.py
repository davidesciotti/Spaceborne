import numpy as np
import scipy
import spaceborne.my_module as mm
import matplotlib.pyplot as plt

cov = np.load('/home/davide/Documenti/Lavoro/Programmi/CLOE_benchmarks/CovMat-3x2pt-GaussSSC-32Bins-13245deg2.npy')

np_inv = np.linalg.inv(cov)
scipy_inv = scipy.linalg.inv(cov)

diff = mm.percent_diff(np_inv, scipy_inv)
mm.matshow(diff, abs_val=True, title='np vs scipy inv, \\% diff', log=False, threshold=5)
