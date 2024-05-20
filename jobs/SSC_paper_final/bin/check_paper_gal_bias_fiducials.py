import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('/home/davide/Documenti/Lavoro/Programmi/cl_v2/bin')
import wf_cl_lib

# this should be the file used in the paper
ngbtab_ep10 = np.genfromtxt('/home/davide/Documenti/Lavoro/Programmi/common_data/vincenzo/SPV3_07_2022/'
                            'Flagship_1_restored/InputNz/Lenses/Flagship/ngbTab-EP10.dat')
nitab_ep10 = np.genfromtxt('/home/davide/Documenti/Lavoro/Programmi/common_data/vincenzo/SPV3_07_2022/'
                            'Flagship_1_restored/InputNz/Lenses/Flagship/niTab-EP10.dat')

# where is this bias taken from?
bias_from_ngtab_file = ngbtab_ep10[1, :]

# wrong zeff
z_edges_low = ngbtab_ep10[3, :]
z_edges_high = ngbtab_ep10[4, :]
z_centers = (z_edges_low + z_edges_high) / 2


# right zeff
z_centers = wf_cl_lib.get_z_effective_isaac(nitab_ep10[:, 0], nitab_ep10[:, 1:])
print([f'{z_centers[i]:.3f}' for i in range(len(z_centers))])

# paper zeff - very similar to the ones above, I don't know the reason of the mismatch
z_centers = np.array([ 0.233, 0.373, 0.455, 0.571, 0.686,
0.796, 0.913, 1.070, 1.195, 1.628])

paper_bias_fiducials = wf_cl_lib.b_of_z_fs1_pocinofit(z_centers)
print([f'{paper_bias_fiducials[i]:.3f}' for i in range(len(paper_bias_fiducials))])

# plot pocino to check if they are the same 
z_plot = np.linspace(0, 2.5, 100)
bias_pocino_plot = wf_cl_lib.b_of_z_fs1_pocinofit(z_plot)
# bias_lepori_plot = wf_cl_lib.b_of_z_fs1_leporifit(z_plot)
# bias_istf_plot = wf_cl_lib.b_of_z(z_plot)

plt.figure()
plt.plot(z_plot, bias_pocino_plot, label='b_of_z_fs1_pocinofit')
# plt.plot(z_plot, bias_lepori_plot, label='b_of_z_fs1_leporifit')
# plt.plot(z_plot, bias_istf_plot, label='b_of_z_istf')
plt.scatter(z_centers, bias_from_ngtab_file, label='bias_from_ngtab_file', color='orange')
plt.legend()
