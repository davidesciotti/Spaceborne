import warnings

import matplotlib.pyplot as plt
import numpy as np
from getdist import plots
from getdist.gaussian_mixtures import GaussianND
from matplotlib import gridspec

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


def cls_triangle_plot(ells_dict, cls_dict, is_auto, zbins, suptitle=None):
    fig, ax = plt.subplots(zbins, zbins, figsize=(7, 7), sharex=True, sharey=True)

    for zi in range(zbins):
        for zj in range(zbins):
            if is_auto and zj > zi:
                ax[zi, zj].axis('off')
                continue

            for label, (ells, cls) in zip(
                ells_dict.keys(), zip(ells_dict.values(), cls_dict.values())
            ):
                ax[zi, zj].plot(
                    ells,
                    cls[:, zi, zj],
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
    max_cl = np.max([cls_dict[key] for key in cls_dict])
    min_cl = np.min([cls_dict[key] for key in cls_dict])
    max_ell = np.max([ells_dict[key] for key in ells_dict])
    min_ell = (
        5
        if np.min([ells_dict[key] for key in ells_dict]) > 5
        else np.min([ells_dict[key] for key in cls_dict])
    )  # this is admittedly a bit arbitrary
    ax[0, 0].set_ylim(min_cl - np.abs(5 * min_cl), 5 * max_cl)
    ax[0, 0].set_xlim(min_ell, 2 * max_ell)

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


def plot_ell_cuts(
    ell_cuts_a,
    ell_cuts_b,
    ell_cuts_c,
    label_a,
    label_b,
    label_c,
    kmax_h_over_Mpc,
    zbins,
):
    # Get the global min and max values for the color scale
    vmin = min(ell_cuts_a.min(), ell_cuts_b.min(), ell_cuts_c.min())
    vmax = max(ell_cuts_a.max(), ell_cuts_b.max(), ell_cuts_c.min())

    # Create a gridspec layout
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.12])

    # Create axes based on the gridspec layout
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])
    cbar_ax = plt.subplot(gs[3])

    ticks = np.arange(1, zbins + 1)
    # Set x and y ticks for both subplots
    for ax in [ax0, ax1, ax2]:
        ax.set_xticks(np.arange(zbins))
        ax.set_yticks(np.arange(zbins))
        ax.set_xticklabels(ticks, fontsize=15)
        ax.set_yticklabels(ticks, fontsize=15)
        ax.set_xlabel('$z_{\\rm bin}$', fontsize=15)
        ax.set_ylabel('$z_{\\rm bin}$', fontsize=15)

    # Display the matrices with the shared color scale
    cax0 = ax0.matshow(ell_cuts_a, vmin=vmin, vmax=vmax)
    _cax1 = ax1.matshow(ell_cuts_b, vmin=vmin, vmax=vmax)
    _cax2 = ax2.matshow(ell_cuts_c, vmin=vmin, vmax=vmax)

    # Add titles to the plots
    ax0.set_title(label_a, fontsize=18)
    ax1.set_title(label_b, fontsize=18)
    ax2.set_title(label_c, fontsize=18)
    fig.suptitle(f'kmax = {kmax_h_over_Mpc:.2f} h_over_mpc_tex', fontsize=18, y=0.85)

    # Add a shared colorbar on the right
    cbar = fig.colorbar(cax0, cax=cbar_ax)
    cbar.set_label(
        '$\\ell^{\\rm max}_{ij}$',
        fontsize=15,
        loc='center',
    )
    cbar.ax.tick_params(labelsize=15)

    plt.tight_layout()
    plt.show()


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


def triangle_plot_old(
    fm_backround,
    fm_foreground,
    fiducials,
    title,
    label_background,
    label_foreground,
    param_names_labels,
    param_names_labels_toplot,
    param_names_labels_tex=None,
    rotate_param_labels=False,
):
    idxs_tokeep = [
        param_names_labels.index(param) for param in param_names_labels_toplot
    ]

    # parameters' covariance matrix - first invert, then slice! Otherwise, you're fixing the nuisance parameters
    fm_inv_bg = np.linalg.inv(fm_backround)[np.ix_(idxs_tokeep, idxs_tokeep)]
    fm_inv_fg = np.linalg.inv(fm_foreground)[np.ix_(idxs_tokeep, idxs_tokeep)]

    fiducials = [fiducials[idx] for idx in idxs_tokeep]
    param_names_labels = [param_names_labels[idx] for idx in idxs_tokeep]

    if param_names_labels_tex is not None:
        warnings.warn(
            'the user should make sure that the order of the param_names_labels_tex'
            ' list is the same as the order of the param_names_labels:',
            stacklevel=2,
        )
        print(param_names_labels_tex)
        print(param_names_labels)
        # remove all the "$" from param_names_labels_tex
        param_names_labels_tex = [
            param_name.replace('$', '') for param_name in param_names_labels_tex
        ]

    bg_contours = GaussianND(
        mean=fiducials,
        cov=fm_inv_bg,
        names=param_names_labels,
        labels=param_names_labels_tex,
    )
    fg_contours = GaussianND(
        mean=fiducials,
        cov=fm_inv_fg,
        names=param_names_labels,
        labels=param_names_labels_tex,
    )

    g = plots.get_subplot_plotter(subplot_size=2.3)
    g.settings.subplot_size_ratio = 1
    g.settings.linewidth = 3
    g.settings.legend_fontsize = 20
    g.settings.linewidth_contour = 3
    g.settings.axes_fontsize = 20
    g.settings.axes_labelsize = 20
    g.settings.lab_fontsize = 25  # this is the x labels size
    g.settings.scaling = (
        True  # prevent scaling down font sizes even though small subplots
    )
    g.settings.tight_layout = True
    g.settings.axis_tick_x_rotation = 45
    g.settings.solid_colors = 'tab10'

    g.triangle_plot(
        [bg_contours, fg_contours],
        # names=param_names_labels,
        filled=True,
        contour_lws=2,
        ls=['-', '-'],
        legend_labels=[label_background, label_foreground],
        legend_loc='upper right',
        contour_colors=['tab:blue', 'tab:orange'],
        line_colors=['tab:blue', 'tab:orange'],
    )

    if rotate_param_labels:
        # Rotate x and y parameter name labels.
        # * also useful if you want to simply align them, by setting rotation=0
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


def contour_plot_chainconsumer(cov, trimmed_fid_dict):
    """Example usage:
                # decide params to show in the triangle plot
                cosmo_param_names = list(fiducials_dict.keys())[:num_params_tokeep]
                shear_bias_param_names = [f'm{(zi + 1):02d}_photo' for zi in range(zbins)]
                params_tot_list = cosmo_param_names + shear_bias_param_names

                trimmed_fid_dict = {param: fiducials_dict[param] for param in params_tot_list}

                # get the covariance matrix (careful on how you cut the FM!!)
                fm_idxs_tokeep = [list(fiducials_dict.keys()).index(param) for param in params_tot_list]
                cov = np.linalg.inv(fm)[fm_idxs_tokeep, :][:, fm_idxs_tokeep]

                plot_utils.contour_plot_chainconsumer(cov, trimmed_fid_dict)
    :param cov:
    :param trimmed_fid_dict:
    :return:
    """
    from chainconsumer import ChainConsumer

    param_names = list(trimmed_fid_dict.keys())
    param_means = list(trimmed_fid_dict.values())

    c = ChainConsumer()
    c.add_covariance(param_means, cov, parameters=param_names, name='Cov')
    c.add_marker(
        param_means,
        parameters=param_names,
        name='fiducial',
        marker_style='.',
        marker_size=50,
        color='r',
    )
    c.configure(usetex=False, serif=True)
    fig = c.plotter.plot()
    return fig
