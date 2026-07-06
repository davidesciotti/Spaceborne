import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap

mpl_rcparams_dict = {
    'lines.linewidth': 1.5,
    'font.size': 17,
    'axes.labelsize': 'large',
    'axes.titlesize': 'large',
    'xtick.labelsize': 'large',
    'ytick.labelsize': 'large',
    #  'mathtext.fontset': 'stix',
    #  'font.family': 'STIXGeneral',
    'figure.figsize': (15, 10),
    'lines.markersize': 8,
    # 'axes.grid': True,
    # 'figure.constrained_layout.use': False,
    # 'axes.axisbelow': True
}

mpl_other_dict = {
    'cosmo_labels_TeX': [
        '$\\Omega_{{\\rm m},0}$',
        '$\\Omega_{{\\rm b},0}$',
        '$w_0$',
        '$w_a$',
        '$h$',
        '$n_{\\rm s}$',
        '$\\sigma_8$',
        '${\\rm log}_{10}(T_{\\rm AGN}/{\\rm K})$',
    ],
    'IA_labels_TeX': ['$A_{\\rm IA}$', '$\\eta_{\\rm IA}$', '$\\beta_{\\rm IA}$'],
    # 'galaxy_bias_labels_TeX': build_labels_TeX(zbins)[0],
    # 'shear_bias_labels_TeX': build_labels_TeX(zbins)[1],
    # 'zmean_shift_labels_TeX': build_labels_TeX(zbins)[2],
    'cosmo_labels': ['Om', 'Ob', 'wz', 'wa', 'h', 'ns', 's8', 'logT'],
    'IA_labels': ['AIA', 'etaIA', 'betaIA'],
    # 'galaxy_bias_labels': build_labels(zbins)[0],
    # 'shear_bias_labels': build_labels(zbins)[1],
    # 'zmean_shift_labels': build_labels(zbins)[2],
    'ylabel_perc_diff_wrt_mean': '$ \\bar{\\sigma}_\\alpha^i / \\bar{\\sigma}^{\\; m}_\\alpha -1 $ [%]',
    'ylabel_sigma_relative_fid': '$ \\sigma_\\alpha/ \\theta^{fid}_\\alpha $ [%]',
    'dpi': 500,
    'pic_format': 'pdf',
    'h_over_mpc_tex': '$h\\,{\\rm Mpc}^{-1}$',
    'kmax_tex': '$k_{\\rm max}$',
    'kmax_star_tex': '$k_{\\rm max}^\\star$',
}


# matplotlib.rcParams.update(mpl_cfg.mpl_rcParams_dict)

param_names_label = mpl_other_dict['cosmo_labels_TeX']
ylabel_perc_diff_wrt_mean = mpl_other_dict['ylabel_perc_diff_wrt_mean']
ylabel_sigma_relative_fid = mpl_other_dict['ylabel_sigma_relative_fid']
# plt.rcParams['axes.axisbelow'] = True
# markersize = mpl_cfg.mpl_rcParams_dict['lines.markersize']


def plot_dominant_array_element(
    arrays_dict, tab_colors, elements_auto, elements_cross, show_zero: bool = True
):
    """Plot 2D arrays from a dictionary, highlighting the dominant component in
    each element.

    Colors are assigned based on the array with the dominant component
    at each position. If no component is dominant (all are zero), the
    color will be white.
    """
    centers = [
        elements_auto // 2,
        elements_auto + elements_cross // 2,
        elements_auto + elements_cross + elements_auto // 2,
    ]
    labels = ['WL', 'GGL', 'GCph']

    # Stack arrays along a new dimension and calculate the absolute values
    stacked_abs_arrays = np.abs(np.stack(list(arrays_dict.values()), axis=-1))

    # Find indices of the dominant array at each position
    dominant_indices = np.argmax(stacked_abs_arrays, axis=-1)

    # Add an extra category for non-dominant cases (where all arrays are zero)
    non_dominant_value = (
        -1
    )  # Choose a value that doesn't conflict with existing indices
    dominant_indices[np.all(stacked_abs_arrays == 0, axis=-1)] = non_dominant_value

    # Prepare the colormap, including an extra color for non-dominant cases
    # 'white' is for non-dominant cases
    selected_colors = ['white'] + tab_colors[: len(arrays_dict)]
    cmap = ListedColormap(selected_colors)

    # Plot the dominant indices with the custom colormap
    plt.figure(figsize=(10, 8))
    im = plt.imshow(
        dominant_indices,
        cmap=cmap,
        vmin=non_dominant_value - 0.5,  # Shifted by -0.5
        vmax=len(arrays_dict)
        - 0.5,  # Shifted to +0.5 (which is len(arrays_dict) - 0.5)
    )

    # Create a colorbar with labels
    all_ticks = np.arange(non_dominant_value, len(arrays_dict))
    all_labels = ['0'] + list(arrays_dict.keys())

    if show_zero:
        # Show everything normally
        cbar = plt.colorbar(im, ticks=all_ticks)
        cbar.set_ticklabels(all_labels)
    else:
        # We need a new mappable just for the colorbar that excludes the -1 (white) class
        # Create a colormap with ONLY the valid colors (ignoring the first 'white' color)
        cmap_no_zero = ListedColormap(selected_colors[1:])

        # Create a dummy scalar mappable for the colorbar
        sm = plt.cm.ScalarMappable(
            cmap=cmap_no_zero,
            # We want the indices to be 0, 1, 2, ..., N.
            # So the boundaries of the colors must be at -0.5, 0.5, 1.5, ..., N+0.5
            norm=plt.Normalize(vmin=-0.5, vmax=len(arrays_dict) - 1 + 0.5),
        )
        sm.set_array([])

        # Draw the colorbar using the dummy mappable, explicitly attaching it to the image axes
        cbar = plt.colorbar(sm, ax=im.axes, ticks=np.arange(0, len(arrays_dict)))
        cbar.set_ticklabels(all_labels[1:])

    # plot lines to separate the probe blocks
    kw = {'lw': 2, 'color': 'k', 'ls': '--'}
    plt.axvline(elements_auto - 0.5, **kw)
    plt.axvline(elements_auto + elements_cross - 0.5, **kw)
    plt.axhline(elements_auto - 0.5, **kw)
    plt.axhline(elements_auto + elements_cross - 0.5, **kw)
    plt.xticks([])
    plt.yticks([])

    for idx, label in enumerate(labels):
        x = centers[idx]
        plt.text(x, -1.5, label, va='bottom', ha='center')
        plt.text(-1.5, x, label, va='center', ha='right', rotation='vertical')


def matshow_vcenter(matrix, vcenter=0):
    """Plots a matrix with a 0-centered, asymmetric colorbar."""
    from matplotlib.colors import TwoSlopeNorm

    plt.matshow(matrix, cmap='RdBu_r', norm=TwoSlopeNorm(vcenter=vcenter))
    plt.colorbar()
    plt.show()


def plot_correlation_matrix(correlation_matrix):
    plt.matshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar()


def plot_kernels(ccl_obj, z_grid: np.ndarray, zbins: int, clr: list):
    plt.figure()
    for zi in range(zbins):
        plt.plot(
            z_grid,
            ccl_obj.wf_delta_arr[:, zi],
            c=clr[zi],
            alpha=0.6,
            ls='-.',
            label='density' if zi == 0 else None,
        )
        plt.plot(
            z_grid,
            ccl_obj.wf_mu_arr[:, zi],
            c=clr[zi],
            alpha=0.6,
            ls='--',
            label='magnification' if zi == 0 else None,
        )
        plt.plot(
            z_grid,
            ccl_obj.wf_galaxy_arr[:, zi],
            c=clr[zi],
            alpha=0.6,
            label='total' if zi == 0 else None,
        )
    plt.xlabel('$z$')
    plt.ylabel(r'$W_i^{POS}(z)$')
    plt.suptitle('Galaxy kernels\n(w/o galaxy bias)')
    plt.tight_layout()
    plt.legend()

    wf_ia_contribution_arr = ccl_obj.wf_ia_contribution_arr

    plt.figure()
    for zi in range(zbins):
        plt.plot(
            z_grid,
            ccl_obj.wf_gamma_arr[:, zi],
            c=clr[zi],
            alpha=0.6,
            ls='-.',
            label='shear' if zi == 0 else None,
        )
        plt.plot(
            z_grid,
            wf_ia_contribution_arr[:, zi],
            c=clr[zi],
            alpha=0.6,
            ls='--',
            label='IA' if zi == 0 else None,
        )
        plt.plot(
            z_grid,
            ccl_obj.wf_lensing_arr[:, zi],
            c=clr[zi],
            alpha=0.6,
            label='total' if zi == 0 else None,
        )
    plt.xlabel('$z$')
    plt.legend()
    plt.ylabel(r'$W_i^{SHE}(z)$')
    plt.suptitle('Lensing kernels')
    plt.tight_layout()


def cls_triangle_plot(
    ells_dict: dict,
    cls_dict: dict,
    is_auto: bool,
    zbins: int,
    twoellplusone: bool,
    suptitle=None,
    cov_6d=None,
):
    fig, ax = plt.subplots(zbins, zbins, figsize=(7, 7), sharex=True, sharey=True)

    for zi in range(zbins):
        for zj in range(zbins):
            if is_auto and zj > zi:
                ax[zi, zj].axis('off')
                continue

            for label, (ells, cls) in zip(
                ells_dict.keys(),
                zip(ells_dict.values(), cls_dict.values(), strict=True),
                strict=True,
            ):
                prefac = 2 * ells + 1 if twoellplusone else 1.0

                if cov_6d is None:
                    ax[zi, zj].plot(
                        ells,
                        prefac * cls[:, zi, zj],
                        label=label if (zi == zbins - 1 and zj == zbins - 1) else None,
                        alpha=1,
                        ls='--' if label == 'input' else '-',
                        # zorder=2.0,
                        lw=1.5,
                    )
                else:
                    ax[zi, zj].errorbar(
                        ells,
                        prefac * cls[:, zi, zj],
                        yerr=np.sqrt(np.diag(cov_6d[:, :, zi, zj, zi, zj])),
                        label=label if (zi == zbins - 1 and zj == zbins - 1) else None,
                        alpha=1,
                        ls='--' if label == 'input' else '-',
                        # zorder=2.0,
                        lw=1.5,
                    )

            ax[zi, zj].axhline(0.0, c='k', lw=0.8, zorder=0)
            ax[zi, zj].tick_params(axis='both', which='both', direction='in')

            # rotate y ticks
            # if zj == 0:
            #     for tick_label in ax[zi, zj].get_yticklabels():
            #         tick_label.set_rotation(45)

    # Axes formatting
    ax[0, 0].set_xscale('log')
    ax[0, 0].xaxis.get_major_locator().set_params(numticks=99)
    ax[0, 0].xaxis.get_minor_locator().set_params(
        numticks=99, subs=np.arange(0.1, 1.0, 0.1)
    )
    ax[0, 0].set_yscale(
        'symlog', linthresh=1e-10, linscale=0.45, subs=np.arange(0.1, 1.0, 0.1)
    )

    # Axis limits
    # max_cl = np.max([cls_dict[key] for key in cls_dict])
    # min_cl = np.min([cls_dict[key] for key in cls_dict])
    # max_ell = np.max([ells_dict[key] for key in ells_dict])
    # min_ell = (
    #     5
    #     if np.min([ells_dict[key] for key in ells_dict]) > 5
    #     else np.min([ells_dict[key] for key in cls_dict])
    # )  # this is admittedly a bit arbitrary
    # ax[0, 0].set_ylim(min_cl - np.abs(5 * min_cl), 5 * max_cl)
    # ax[0, 0].set_xlim(min_ell, 2 * max_ell)

    all_plotted = []
    for label in ells_dict:
        ells = ells_dict[label]
        cls = cls_dict[label]
        prefac = 2 * ells + 1 if twoellplusone else 1.0
        # Loop over all zi, zj (including only visible panels if is_auto)
        for zi in range(zbins):
            for zj in range(zbins):
                if is_auto and zj > zi:
                    continue
                all_plotted.append(prefac * cls[:, zi, zj])
    all_plotted = np.concatenate([arr.flatten() for arr in all_plotted])
    max_cl = np.max(all_plotted)
    min_cl = np.min(all_plotted)
    ax[0, 0].set_ylim(
        min_cl - 0.1 * (max_cl - min_cl), max_cl + 0.1 * (max_cl - min_cl)
    )

    for zi in range(zbins):
        for zj in range(zbins):
            if not (is_auto and zj > zi):
                ax[zi, zj].autoscale(False)

    for zi in range(zbins):
        for zj in range(zbins):
            if not (is_auto and zj > zi):
                ax[zi, zj].set_yscale(
                    'symlog',
                    linthresh=1e-10,
                    linscale=0.45,
                    subs=np.arange(0.1, 1.0, 0.1),
                )

    fig.subplots_adjust(
        left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.0, hspace=0.0
    )

    fig.supxlabel('$\\ell$', y=-0.05, va='top')
    fig.supylabel('$C_\\ell$', x=-0.1, ha='right')

    # Add legend in bottom-right visible subplot
    ax[-1, -1].legend(loc='upper right', fontsize='small')

    if suptitle is not None:
        fig.suptitle(suptitle)

    return fig, ax


def cls_triangle_plot_v2(cl_dict, zbins, is_auto, weights=None):
    """
    Function adapted from https://heracles.readthedocs.io/stable/examples/example.html
    """

    # Find max y value across all cls for setting y-axis limits
    cl_list = [cl_dict[key]['cls'] for key in cl_dict]
    max_y = 1.5 * np.max([np.max(cl) for cl in cl_list])

    # Find min and max y values across all ells for setting x-axis limits
    ell_list = [cl_dict[key]['ells'] for key in cl_dict]
    max_x = 3 * np.max([np.max(ell) for ell in ell_list])
    min_x = 1 / 3 * np.min([np.min(ell) for ell in ell_list])

    fig, ax = plt.subplots(
        zbins, zbins, figsize=(zbins, zbins), sharex=True, sharey=True
    )

    for key in cl_dict:
        ells = cl_dict[key]['ells']
        cl_3d = cl_dict[key]['cls']
        ls = cl_dict[key]['ls']

        prefactor = 2 * ells + 1 if weights == 'twoellplusone' else 1.0

        if is_auto:
            for zi in range(zbins):
                for zj in range(zi):
                    ax[zj, zi].axis('off')
                for zj in range(zi, zbins):
                    ax[zj, zi].plot(
                        ells, prefactor * cl_3d[:, zi, zj], label=key, ls=ls
                    )
                    ax[zj, zi].axhline(0.0, c='k', zorder=-1)
                    ax[zj, zi].tick_params(axis='both', which='both', direction='in')
        else:
            for zi in range(zbins):
                for zj in range(zbins):
                    ax[zj, zi].plot(
                        ells, prefactor * cl_3d[:, zi, zj], label=key, ls=ls
                    )
                    ax[zj, zi].axhline(0.0, c='k', zorder=-1)
                    ax[zj, zi].tick_params(axis='both', which='both', direction='in')

    ax[0, 0].set_xscale('log')
    ax[0, 0].set_xlim(min_x, max_x)
    ax[0, 0].xaxis.get_major_locator().set_params(numticks=99)
    ax[0, 0].xaxis.get_minor_locator().set_params(
        numticks=99, subs=np.arange(0.1, 1.0, 0.1)
    )
    # ax[0, 0].set_yscale(
    # 'symlog', linthresh=1e-7, linscale=0.45, subs=np.arange(0.1, 1.0, 0.1)
    # )
    # ax[0, 0].set_ylim(-2e-7, max_y)
    # ax[0, 0].set_yscale('log')
    ax[-1, -1].legend(loc='lower right', bbox_to_anchor=(1, -1))

    fig.subplots_adjust(
        left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.0, hspace=0.0
    )
    fig.supxlabel(r'$\ell$', y=-0.05, va='top')
    fig.supylabel(r'$C_\ell$ diff [%]', x=-0.1, ha='right')


def bar_plot(
    data,
    title,
    label_list,
    divide_fom_by_10_plt,
    bar_width=0.18,
    nparams=7,
    param_names_label=None,
    second_axis=False,
    no_second_axis_bars=0,
    superimpose_bars=False,
    show_markers=False,
    ylabel=None,
    include_fom=False,
    figsize=None,
    grey_bars=False,
    alpha=1,
):
    """data: usually the percent uncertainties, but could also be the percent difference"""
    no_cases = data.shape[0]
    no_params = data.shape[1]

    markers = [
        '^',
        '*',
        'D',
        'v',
        'p',
        'P',
        'X',
        'h',
        'H',
        'd',
        '8',
        '1',
        '2',
        '3',
        '4',
        'x',
        '+',
    ]
    marker_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    markers = markers[:no_cases]
    marker_colors = marker_colors[:no_cases]
    # zorders = np.arange(no_cases)  # this is because I want to revert this in the case of superimposed bars
    zorders = np.arange(
        1, no_cases + 1
    )  # this is because I want to revert this in the case of superimposed bars

    # colors = cm.Paired(np.linspace(0, 1, data.shape[1]))

    # Set position of bar on x-axis
    bar_centers = np.zeros(data.shape)

    if data.ndim == 1:  # just one vector
        data = data[None, :]
        bar_centers = np.arange(no_params)
        bar_centers = bar_centers[None, :]
    elif data.ndim != 1 and not superimpose_bars:
        for bar_idx in range(no_cases):
            if bar_idx == 0:
                bar_centers[bar_idx, :] = np.arange(no_params) - bar_width
            else:
                bar_centers[bar_idx, :] = [
                    x + bar_idx * bar_width for x in bar_centers[0]
                ]

    # in this case, I simply define the bar centers to be the same
    elif data.ndim != 1 and superimpose_bars:
        zorders = zorders[::-1]
        bar_centers = np.arange(no_params)
        bar_centers = bar_centers[None, :]
        bar_centers = np.repeat(bar_centers, no_cases, axis=0)

    if param_names_label is None:
        param_names_label = mpl_other_dict['cosmo_labels_TeX']
        fom_div_10_str = '/10' if divide_fom_by_10_plt else ''
        if include_fom:
            param_names_label = mpl_other_dict['cosmo_labels_TeX'] + [
                f'FoM{fom_div_10_str}'
            ]

    if ylabel is None:
        ylabel = ylabel_sigma_relative_fid

    if figsize is None:
        figsize = (12, 8)

    bar_color = ['grey' for _ in range(no_cases)] if grey_bars else None

    if second_axis:
        # this check is quite obsolete...
        assert no_cases == 3, 'data must have 3 rows to display the second axis'

        fig, ax = plt.subplots(figsize=figsize)
        ax.grid()
        ax.set_axisbelow(True)

        for bar_idx in range(no_cases - no_second_axis_bars):
            ax.bar(
                bar_centers[bar_idx, :],
                data[bar_idx, :],
                width=bar_width,
                edgecolor='grey',
                label=label_list[bar_idx],
            )

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel(ylabel_sigma_relative_fid)
        ax.set_title(title)
        ax.set_xticks(range(nparams), param_names_label)

        # second axis
        ax2 = ax.twinx()
        # ax2.set_ylabel('(GS/GO - 1) $\\times$ 100', color='g')
        ax2.set_ylabel('% uncertainty increase')
        for bar_idx in range(1, no_second_axis_bars + 1):
            ax2.bar(
                bar_centers[-bar_idx, :],
                data[-bar_idx, :],
                width=bar_width,
                edgecolor='grey',
                label=label_list[-bar_idx],
                color='g',
                alpha=alpha,
                zorder=zorders[bar_idx],
            )
        ax2.tick_params(axis='y')

        fig.legend(
            loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes
        )
        return

    # elif not second_axis:
    plt.figure(figsize=figsize)
    plt.grid(zorder=0)
    plt.rcParams['axes.axisbelow'] = True

    for bar_idx in range(no_cases):
        label = label_list[bar_idx] if not superimpose_bars else None
        plt.bar(
            bar_centers[bar_idx, :],
            data[bar_idx, :],
            width=bar_width,
            edgecolor='grey',
            alpha=alpha,
            label=label,
            zorder=zorders[bar_idx],
            color=bar_color,
        )
        if show_markers:
            plt.scatter(
                bar_centers[bar_idx, :],
                data[bar_idx, :],
                color=marker_colors[bar_idx],
                marker=markers[bar_idx],
                label=label_list[bar_idx],
                zorder=zorders[bar_idx],
            )

    plt.ylabel(ylabel)
    plt.xticks(range(nparams), param_names_label)
    plt.title(title)
    plt.legend()
    plt.show()


def triangle_plot(
    fisher_matrices,
    fiducials,
    title,
    labels,
    param_names_labels,
    param_names_labels_toplot,
    param_names_labels_tex=None,
    rotate_param_labels=False,
    contour_colors=None,
    line_colors=None,
):
    from getdist import plots
    from getdist.gaussian_mixtures import GaussianND

    idxs_tokeep = [
        param_names_labels.index(param) for param in param_names_labels_toplot
    ]

    # Invert and slice the Fisher matrices, ensuring to keep only the desired parameters
    inv_fisher_matrices = [
        np.linalg.inv(fm)[np.ix_(idxs_tokeep, idxs_tokeep)] for fm in fisher_matrices
    ]

    fiducials = [fiducials[idx] for idx in idxs_tokeep]
    param_names_labels = [param_names_labels[idx] for idx in idxs_tokeep]

    if param_names_labels_tex is not None:
        warnings.warn(
            'Ensure that the order of param_names_labels_tex matches '
            'param_names_labels.',
            stacklevel=2,
        )
        param_names_labels_tex = [
            param_name.replace('$', '') for param_name in param_names_labels_tex
        ]

    # Prepare GaussianND contours for each Fisher matrix
    contours = [
        GaussianND(
            mean=fiducials,
            cov=fm_inv,
            names=param_names_labels,
            labels=param_names_labels_tex,
        )
        for fm_inv in inv_fisher_matrices
    ]

    g = plots.get_subplot_plotter(subplot_size=2.3)
    g.settings.subplot_size_ratio = 1
    g.settings.linewidth = 3
    g.settings.legend_fontsize = 20
    g.settings.linewidth_contour = 3
    g.settings.axes_fontsize = 20
    g.settings.axes_labelsize = 20
    g.settings.lab_fontsize = 25  # this is the x labels size
    g.settings.scaling = (
        True  # prevent scaling down font sizes even with small subplots
    )
    g.settings.tight_layout = True
    g.settings.axis_tick_x_rotation = 45
    g.settings.solid_colors = 'tab10'

    # Set default colors if not provided
    if contour_colors is None:
        contour_colors = [
            f'tab:{color}' for color in ['blue', 'orange', 'green', 'red']
        ]
    if line_colors is None:
        line_colors = contour_colors

    # Plot the triangle plot for all Fisher matrices
    g.triangle_plot(
        contours,
        filled=True,
        contour_lws=2,
        ls=['-'] * len(fisher_matrices),
        legend_labels=labels,
        legend_loc='upper right',
        contour_colors=contour_colors[: len(fisher_matrices)],
        line_colors=line_colors[: len(fisher_matrices)],
    )

    if rotate_param_labels:
        # Rotate x and y parameter name labels
        for ax in g.subplots[:, 0]:
            ax.yaxis.set_label_position('left')
            ax.set_ylabel(
                ax.get_ylabel(), rotation=45, labelpad=20, fontsize=30, ha='center'
            )

        for ax in g.subplots[-1, :]:
            ax.set_xlabel(
                ax.get_xlabel(),
                rotation=45,
                labelpad=20,
                fontsize=30,
                ha='center',
                va='center',
            )

    plt.suptitle(f'{title}', fontsize='x-large')
    plt.show()


def plot_nz_src_lns(zgrid_nz_src, nz_src, zgrid_nz_lns, nz_lns, colors):
    assert nz_src.shape[1] == nz_lns.shape[1], 'number of zbins is not the same'
    zbins = nz_src.shape[1]

    _, ax = plt.subplots(2, 1, sharex=True)
    colors = cm.rainbow(np.linspace(0, 1, zbins))
    for zi in range(zbins):
        ax[0].plot(zgrid_nz_src, nz_src[:, zi], c=colors[zi], label=f'$z_{zi + 1}$')
        # ax[0].axvline(zbin_centers_src[zi], c=colors[zi], ls='--', alpha=0.6, label=r'$z_{%d}^{eff}$' % (zi + 1))
        ax[0].fill_between(zgrid_nz_src, nz_src[:, zi], color=colors[zi], alpha=0.2)
        ax[0].set_xlabel('$z$')
        ax[0].set_ylabel(r'$n_i^{\rm SHE}(z)$')
    ax[0].legend(ncol=2)

    for zi in range(zbins):
        ax[1].plot(zgrid_nz_lns, nz_lns[:, zi], c=colors[zi], label=f'$z_{zi + 1}$')
        # ax[1].axvline(zbin_centers_lns[zi], c=colors[zi], ls='--', alpha=0.6, label=r'$z_{%d}^{eff}$' % (zi + 1))
        ax[1].fill_between(zgrid_nz_lns, nz_lns[:, zi], color=colors[zi], alpha=0.2)
        ax[1].set_xlabel('$z$')
        ax[1].set_ylabel(r'$n_i^{\rm POS}(z)$')
    ax[1].legend(ncol=2)
