from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/cosmo/davide.sciotti/data/Spaceborne')
import spaceborne.ell_utils as ell_utils
import spaceborne.my_module as mm



def plot_dominant_array_element(arrays_dict: dict, tab_colors: list[str], elements_auto, elements_cross, elements_3x2pt):
    """
    Plot 2D arrays from a dictionary, highlighting the dominant component in each element.
    Colors are assigned based on the array with the dominant component at each position.
    If no component is dominant (all are zero), the color will be white.
    """


    # Stack arrays along a new dimension and calculate the absolute values
    stacked_abs_arrays = np.abs(np.stack(list(arrays_dict.values()), axis=-1))

    # Find indices of the dominant array at each position
    dominant_indices = np.argmax(stacked_abs_arrays, axis=-1)

    # Add an extra category for non-dominant cases (where all arrays are zero)
    non_dominant_value = -1  # Choose a value that doesn't conflict with existing indices
    dominant_indices[np.all(stacked_abs_arrays == 0, axis=-1)] = non_dominant_value

    # Prepare the colormap, including an extra color for non-dominant cases
    selected_colors = ['white'] + tab_colors[:len(arrays_dict)]  # 'white' is for non-dominant cases
    cmap = ListedColormap(selected_colors)

    # Plot the dominant indices with the custom colormap
    plt.figure(figsize=(10, 8))
    im = plt.imshow(dominant_indices, cmap=cmap, vmin=non_dominant_value, vmax=len(arrays_dict) - 1)

    # Create a colorbar with labels
    # Set the ticks so they are at the center of each color segment
    cbar_ticks = np.linspace(non_dominant_value, len(arrays_dict) - 1, len(selected_colors))
    cbar_labels = ['0'] + list(arrays_dict.keys())  # 'None' corresponds to the non-dominant case
    cbar = plt.colorbar(im, ticks=cbar_ticks)
    cbar.set_ticklabels(cbar_labels)

    # labels = ['WL', 'GGL', 'GCph']
    # centers = [elements_auto // 2, elements_auto + elements_cross //
    #            2, elements_auto + elements_cross + elements_auto // 2]
    # lw = 2
    # plt.axvline(elements_auto, c='k', lw=lw)
    # plt.axvline(elements_auto + elements_cross, c='k', lw=lw)
    # plt.axhline(elements_auto, c='k', lw=lw)
    # plt.axhline(elements_auto + elements_cross, c='k', lw=lw)
    # plt.xticks([])
    # plt.yticks([])

    # for idx, label in enumerate(labels):
    #     x = centers[idx]
    #     plt.text(x, -1.5, label, va='bottom', ha='center')
    #     plt.text(-1.5, x, label, va='center', ha='right', rotation='vertical')

    plt.show()


cov_folder = '/home/cosmo/davide.sciotti/data/Spaceborne'
cov_3x2pt_g = np.load(f'{cov_folder}/output_CCL_cNG/CovMat-3x2pt-Gauss-32Bins-13245deg2.npy')
cov_3x2pt_gcng = np.load(f'{cov_folder}/output_CCL_cNG/CovMat-3x2pt-GausscNGCCL-32Bins-13245deg2.npy')
cov_3x2pt_gssc = np.load(f'{cov_folder}/output_CCL_SSC/CovMat-3x2pt-GaussSSCCCL-32Bins-13245deg2.npy')

cov_3x2pt_cng = cov_3x2pt_gcng - cov_3x2pt_g
cov_3x2pt_ssc = cov_3x2pt_gssc - cov_3x2pt_g
cov_3x2pt_tot = cov_3x2pt_g + cov_3x2pt_cng + cov_3x2pt_ssc


zbins = 13
zpairs_auto = zbins * (zbins + 1) // 2
zpairs_cross = zbins ** 2
nbl_wl = 32
nbl_3x2pt = 29

ells_wl, _ = ell_utils.compute_ells(nbl_wl, ell_min=10, ell_max=5000, recipe='ISTF')
ells_3x2pt, _ = ell_utils.compute_ells(nbl_3x2pt, ell_min=10, ell_max=3000, recipe='ISTF')

elem_auto_wl = zpairs_auto * nbl_wl
elem_auto_gc = zpairs_auto * nbl_3x2pt
elem_cross_3x2pt = zpairs_cross * nbl_3x2pt

# WL
cov_block_g = cov_3x2pt_g[:elem_auto_wl, :elem_auto_wl]
cov_block_ssc = cov_3x2pt_ssc[:elem_auto_wl, :elem_auto_wl]
cov_block_cng = cov_3x2pt_cng[:elem_auto_wl, :elem_auto_wl]
cov_block_tot = cov_3x2pt_tot[:elem_auto_wl, :elem_auto_wl]

plt.loglog(ells_wl, np.diag(cov_block_g)[::zpairs_auto], c='tab:green', label='g')
plt.loglog(ells_wl, np.diag(cov_block_ssc)[::zpairs_auto], c='tab:red', label='ssc')
plt.loglog(ells_wl, np.diag(cov_block_cng)[::zpairs_auto], c='tab:blue', label='cng')
plt.loglog(ells_wl, np.diag(cov_block_tot)[::zpairs_auto], c='k', label='tot')
plt.legend()
plt.title('WL')

# GCph
cov_block_g = cov_3x2pt_g[-elem_auto_gc:, -elem_auto_gc:]
cov_block_ssc = cov_3x2pt_ssc[-elem_auto_gc:, -elem_auto_gc:]
cov_block_cng = cov_3x2pt_cng[-elem_auto_gc:, -elem_auto_gc:]
cov_block_tot = cov_3x2pt_tot[-elem_auto_gc:, -elem_auto_gc:]

plt.figure()
plt.loglog(ells_3x2pt, np.diag(cov_block_g)[::zpairs_auto], c='tab:green', label='g')
plt.loglog(ells_3x2pt, np.diag(cov_block_ssc)[::zpairs_auto], c='tab:red', label='ssc')
plt.loglog(ells_3x2pt, np.diag(cov_block_cng)[::zpairs_auto], c='tab:blue', label='cng')
plt.loglog(ells_3x2pt, np.diag(cov_block_tot)[::zpairs_auto], c='k', label='tot')
plt.legend()
plt.title('GCph')

mm.matshow(cov_3x2pt_cng)

arrays_dict = {'g': cov_3x2pt_g,
               'ssc': cov_3x2pt_ssc,
               'cng': cov_3x2pt_cng,
            #    'tot': cov_3x2pt_tot
               }
tab_colors = [ 'tab:green',
             'tab:red',
             'tab:blue',
            #  'k'
             ]
plot_dominant_array_element(arrays_dict, tab_colors, elements_auto=None, elements_cross=None, elements_3x2pt=None)

# test glll
cov_glll_g = cov_3x2pt_g[elem_auto_wl:elem_auto_wl + elem_cross_3x2pt, :elem_auto_wl]
cov_glll_ssc = cov_3x2pt_ssc[elem_auto_wl:elem_auto_wl + elem_cross_3x2pt, :elem_auto_wl]
cov_glll_cng = cov_3x2pt_cng[elem_auto_wl:elem_auto_wl + elem_cross_3x2pt, :elem_auto_wl]
arrays_dict = {'g': cov_glll_g,
               'ssc': cov_glll_ssc,
               'cng': cov_glll_cng,
            #    'tot': cov_glll_tot
               }

ratio = np.abs(cov_glll_g/cov_glll_cng)
ratio[ratio > 1] = 0
mm.matshow(ratio)

mm.matshow(cov_glll_g)
plot_dominant_array_element(arrays_dict, tab_colors, elements_auto=None, elements_cross=None, elements_3x2pt=None)