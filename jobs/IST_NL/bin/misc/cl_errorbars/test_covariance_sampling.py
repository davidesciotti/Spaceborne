import sys, os
from getdist import plots, MCSamples
import getdist
import matplotlib.pyplot as plt
import IPython
plt.rcParams['text.usetex']=True
import numpy as np


ndim = 3
nsamp = 100_000

random_state = np.random.default_rng(10) # seed random generator

# clever way to construct random covariance matrix
A = random_state.random((ndim,ndim))
cov = np.dot(A, A.T)
samps = random_state.multivariate_normal([0]*ndim, cov, size=nsamp)

# pick a parameter index
i = 0

# compute sigma in both ways
classical_sigma = np.sqrt(cov[i, i])
sampled_sigma = np.sqrt(np.var(samps[:, i]))

# make a histogram (a triangle plot is better 👇)
plt.hist(samps[:, i], bins=40)


# Get the getdist MCSamples objects for the samples, specifying same parameter
# names and labels; if not specified weights are assumed to all be unity
names = ["x%s"%i for i in range(ndim)]
labels =  ["x_%s"%i for i in range(ndim)]
# this line is just to pass the right object to g.triangle_plot
samples = MCSamples(samples=samps,names = names, labels = labels)

g = plots.get_subplot_plotter()
g.triangle_plot(samples, filled=True)

# count the elements in a very inefficient way, just to be sure 
p = 0
for j in range(samps.shape[0]):
    if -classical_sigma < samps[j, i] < classical_sigma:
        p +=1

# print param index and % of elements in the interval
print(i, p/nsamp * 100)
