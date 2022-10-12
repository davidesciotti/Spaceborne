"""
module to reproduce vincenzo's plots in the paper, *after* the new specs update
"""
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib
from matplotlib.ticker import FormatStrFormatter
from scipy import interpolate
import numpy as np

matplotlib.use('Qt5Agg')

project_path = Path.cwd().parent.parent.parent.parent
job_path = Path.cwd().parent.parent

sys.path.append(str(project_path / 'lib'))
import my_module as mm

start_time = time.perf_counter()

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def fig_6_and_7(probe, probe_label, pedix, fmt='%.2f' , fig_number=6):

    params = {'lines.linewidth': 2,
              'font.size': 14,
              'axes.labelsize': 'small',
              'axes.titlesize': 'medium',
              'xtick.labelsize': 'small',
              'ytick.labelsize': 'small',
              'mathtext.fontset': 'stix',
              'font.family': 'STIXGeneral',
              }
    plt.rcParams.update(params)
    markersize = 4

    # plot 1: GS/GO vs nbl
    cases = ['Pes', 'Opt']
    linestyle = 'dashed'
    colors = ['tab:blue', 'tab:orange']
    params_latex = ["$\Omega_{{\\rm m},0}$", "$\Omega_{{\\rm b},0}$", "$w_0$", "$w_a$", "$h$", "$n_{\\rm s}$", "$\sigma_8$",
                    "${\\rm FoM}$"]
    params_plain = ["Om", "Ob", "w0", "wa", "h", "ns", "sigma8", "FoM"]

    fig, axs = plt.subplots(2, 4, sharex=True, subplot_kw=dict(box_aspect=0.75),
                            constrained_layout=False, figsize=(15, 6.5), tight_layout = {'pad': 0.7})

    # number each axs box: 0 for [0, 0], 1 for [0, 1] and so forth
    axs_idx = np.arange(0, 8, 1).reshape((2, 4))

    # loop over 7 cosmo params + FoM
    for param_idx in range(len(params_latex)):
        # loop over cases and probes
        for case, color in zip(cases, colors):

            print(case)

            tab = np.genfromtxt(job_path / f'input/vincenzo/replot_vincenzo/GSoverGOvsNbZed{case}{probe}.dat')
            NbZed = tab[:, 0].astype(int)

            # take i, j and their "counter" (the param index)
            i, j = np.where(axs_idx == param_idx)[0][0], np.where(axs_idx == param_idx)[1][0]

            # set y ticks with 0.5 step
            # start, end = axs[i, j].get_ylim()
            # axs[i, j].yaxis.set_ticks(np.arange(start, end, step=np.abs((start-end)/5)))
            # axs[i, j].yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))

            axs[i, j].plot(NbZed, tab[:, param_idx + 1], ls=linestyle, markersize=markersize, marker='o', color=color,
                           label=f'{probe_label} {case}')
            axs[i, j].yaxis.set_major_formatter(FormatStrFormatter(f'{fmt}'))

        axs[i, j].grid()
        axs[i, j].set_title(f'{params_latex[param_idx]}', pad=10.0, fontsize=panel_titles_fontsize)

    # legend in the bottom:
    # get labels and handles from one of the axis (the first)
    handles, labels = axs[0,0].get_legend_handles_labels()

    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.06), ncol=2)
    # fig.subplots_adjust(top=0.1)  # or whatever

    fig.supxlabel('${\\cal N}_%s$' %pedix)
    fig.supylabel('$\\sigma_{\\rm GS}(\\theta) \, / \, \\sigma_{\\rm GO}(\\theta)$', x=0.009)

    # fig.tight_layout()

    plt.savefig(job_path / f'output/plots/replot_vincenzo/figs/fig_{fig_number}_replot.{pic_format}', dpi=dpi, bbox_inches = 'tight')


# ! settings definition
which_fig = 'fig_9'
dpi = 500
pic_format = 'pdf'
panel_titles_fontsize = 17
# ! settings definition

if which_fig == 'fig_5':

    params = {'lines.linewidth': 2,
              'font.size': 14,
              'axes.labelsize': 'small',
              'axes.titlesize': 'medium',
              'xtick.labelsize': 'small',
              'ytick.labelsize': 'small',
              'mathtext.fontset': 'stix',
              'font.family': 'STIXGeneral',
              }
    plt.rcParams.update(params)
    markersize = 4

    # plot 1: GS/GO vs nbl
    probes = ['WLO', 'All']  # fig. 5 doesn't show GCph
    probe_labels = ['WL', '$3\\times$2pt']  # to uniform with the paper's notation
    cases = ['Pes', 'Opt']
    linestyles = ['dashed', 'dotted']
    colors = ['tab:blue', 'tab:orange']
    params_latex = ["$\Omega_{{\\rm m},0}$", "$\Omega_{{\\rm b},0}$", "$w_0$", "$w_a$", "$h$", "$n_s$", "$\sigma_8$",
                    "${\\rm FoM}$"]
    params_plain = ["Om", "Ob", "w0", "wa", "h", "ns", "sigma8", "FoM"]

    # import a random parameter's values, just to set nbl_values
    tab = np.genfromtxt(job_path / f'input/vincenzo/replot_vincenzo/GSoverGOvsNbEllOptWLO.dat')
    nbl_values = tab[:, 0].astype(int)

    fig, axs = plt.subplots(2, 4, sharex=True, subplot_kw=dict(box_aspect=0.75),
                            constrained_layout=False, figsize=(15, 6.5), tight_layout = {'pad': 0.7})

    # number each axs box: 0 for [0, 0], 1 for [0, 1] and so forth
    axs_idx = np.arange(0, 8, 1).reshape((2, 4))

    # loop over 7 cosmo params + FoM
    for param_idx in range(len(params_latex)):
        # loop over cases and probes
        for case, ls in zip(cases, linestyles):
            for probe, probe_label, color in zip(probes, probe_labels, colors):
                tab = np.genfromtxt(job_path / f'input/vincenzo/replot_vincenzo/GSoverGOvsNbEll{case}{probe}.dat')
                # take i, j and their "counter" (the param index)
                i, j = np.where(axs_idx == param_idx)[0][0], np.where(axs_idx == param_idx)[1][0]

                # set y ticks with 0.5 step
                # start, end = axs[i, j].get_ylim()
                # axs[i, j].yaxis.set_ticks(np.arange(start, end, step=np.abs((start-end)/5)))
                # axs[i, j].yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))

                axs[i, j].plot(nbl_values, tab[:, param_idx + 1], ls=ls, markersize=markersize, marker='o', color=color,
                               label=f'{probe_label} {case}')
                axs[i, j].yaxis.set_major_formatter(FormatStrFormatter(f'%.2f'))


        axs[i, j].grid()
        axs[i, j].set_title(f'{params_latex[param_idx]}', pad=10.0, fontsize=panel_titles_fontsize)

    # legend in the bottom:
    # get labels and handles from one of the axis (the first)
    handles, labels = axs[0,0].get_legend_handles_labels()

    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.06), ncol=4)
    # fig.subplots_adjust(top=0.1)  # or whatever

    fig.supxlabel('${\\cal N}_{\\ell}$')
    fig.supylabel('$\\sigma_{\\rm GS}(\\theta) \, / \, \\sigma_{\\rm GO}(\\theta)$', x=0.009)

    plt.savefig(job_path / f'output/plots/replot_vincenzo/figs/fig_5_replot.{pic_format}', dpi=dpi, bbox_inches = 'tight')


# ! fig. 6 and 7 are very similar: I defined a function
elif which_fig == 'fig_6':
    fig_6_and_7(probe='WLO', probe_label='WL', fmt='%.3f', pedix='b', fig_number=6)
elif which_fig == 'fig_7':
    fig_6_and_7(probe='All', probe_label='$3\\times 2$pt', fmt='%.2f', pedix='\ell', fig_number=7)


############ FIG 8, 9 ############

elif which_fig == 'fig_8':

    plt.rcParams['figure.figsize'] = (13, 6)

    Nz_values = [10, 12, 14]

    fig, axs = plt.subplots(2, 3, sharex=True, subplot_kw=dict(box_aspect=0.70), constrained_layout=True)
    # fig, axs = plt.subplots(2, 3, sharex=True)


    # number each axs box: 0 for [0, 0], 1 for [0, 1] and so forth
    axs_idx = np.arange(0, 6, 1).reshape((2, 3))

    cases = ['Pes', 'Opt']

    # loop over the different panels
    panel_idx = 0  # panel identifier
    for case in cases:
        for Nz in Nz_values:

            tab = np.genfromtxt(job_path / f'input/vincenzo/replot_vincenzo/RatioFoM-3x2pt-{case}-{Nz}.dat')

            # switch to linear scale
            tab[:, 0] = 10 ** tab[:, 0]
            tab[:, 1] = 10 ** tab[:, 1]

            # take the epsilon values
            epsb_values = np.unique(tab[:, 0])
            epsm_values = np.unique(tab[:, 1])
            n_points = epsm_values.size

            # XXX TODO understand the transpose better, but it works!
            # create a table from the 2nd column, which is a 1D vector.
            FoM_ratio_table = np.reshape(tab[:, 4], (n_points, n_points)).T
            # interpolate: pass x, y, f(x, y) (the latter must be a table, see above)
            f = interpolate.interp2d(epsb_values, epsm_values, FoM_ratio_table, kind='linear')
            # take the desired values
            eps_b = epsb_values

            i, j = np.where(axs_idx == panel_idx)[0][0], np.where(axs_idx == panel_idx)[1][0]

            # single panel
            for eps_m in [0.1, 1, 10, 100]:  # in linear scale (as in the descriprion of the plot)
                # intepolate
                z = f(epsb_values, eps_m)
                # plot - two %% signs to escape the special "%" character
                axs[i, j].plot(epsb_values, z, label='$\\epsilon_m$ = %.1f%%' % eps_m, zorder=1)
                # display only 2 decimal values
                axs[i, j].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

            axs[i, j].grid()

            panel_idx += 1

    # manually shift the y label by hand-picked value
    # fig.supxlabel('$\\epsilon_b \, (\%)$', y=0.04)

    fig.supxlabel('$\\epsilon_b \, (\%)$')
    # fig.supylabel('${\\rm FoM_{GS}} \, / \, {\\rm FoM_{GO}}$')
    fig.supylabel('${\\rm FoM_{GS}} \, / \, {\\rm FoM_{GO}}$', y=0.545)
    # fig.tight_layout()

    plt.xlim(0.1, 20)
    plt.xscale('log')
    axs.flatten()[-1].legend(loc='lower right', borderaxespad=0.3)


    plt.savefig(job_path / f'output/plots/replot_vincenzo/figs/fig_8_replot.{pic_format}', dpi=dpi)

    """############### "manual interpolation"
    # take the rows where eps_m (1st column) == [0.1, 1, 10, 100].
    # since I'm not actually interpolating, in this case just take the nearest value
    nearest_eps_m = find_nearest(tab[:, 1], eps_m)
    # isolate the rows where epsilon_m has the desired value
    rows = np.where(tab[:, 1] == nearest_eps_m)[0]
    
    # (tab[rows, 0]) is equal to epsb_values
    plt.plot(tab[rows, 0], tab[rows, 4], label='manual', zorder=0)"""


elif which_fig == 'fig_9':

    params = {'lines.linewidth': 2,
              'font.size': 15,
              'axes.labelsize': 'large',
              'axes.titlesize': 'large',
              'xtick.labelsize': 'large',
              'ytick.labelsize': 'large',
              'mathtext.fontset': 'stix',
              'font.family': 'STIXGeneral',
              # 'figure.figsize': (15, 8)
              }
    plt.rcParams.update(params)

    ###################### ! fig. 9 ######################

    Nz = 13
    lims = [[1, 8], [1, 15]]
    z_values = np.arange(0.8, 1.1, 0.05)

    fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(15, 7.5))

    panel_idx = 0
    for case, lim in zip(cases, lims):

        tab = np.genfromtxt(job_path / f'input/vincenzo/replot_vincenzo/RatioFoM-3x2pt-{case}-{Nz}.dat')
        tab[:, 0] = 10 ** tab[:, 0]
        tab[:, 1] = 10 ** tab[:, 1]

        # take the epsilon values
        epsb_values = np.unique(tab[:, 0])
        epsm_values = np.unique(tab[:, 1])
        n_points = epsm_values.size

        # produce grid and pass Z values
        X = epsb_values
        Y = epsm_values
        X, Y = np.meshgrid(X, Y)
        Z = np.reshape(tab[:, 3], (n_points, n_points)).T # XXX careful of the tab index!! it's GS/ref

        # levels of contour plot (set a high xorder to have line on top of legend)
        CS = axs[panel_idx].contour(X, Y, Z, levels=z_values, cmap='plasma', zorder=6)

        # plot adjustments
        axs[panel_idx].set_aspect('equal', 'box')
        axs[panel_idx].set_xlabel('$\\epsilon_b \, (\%)$')
        axs[panel_idx].set_ylabel('$\\epsilon_m \, (\%)$')
        axs[panel_idx].set_xlim(lim[0], lim[1])
        axs[panel_idx].set_ylim(lim[0], lim[1])
        axs[panel_idx].grid()


        # legend: from source (see the comment): https://stackoverflow.com/questions/64523051/legend-is-empty-for-contour-plot-is-this-by-design
        h, _ = CS.legend_elements()
        l = ['${\\rm FoM_{GS}} \, / \, {\\rm FoM}_{\\rm ref}}$ = ' + f'{a:.2f}' for a in CS.levels]
        axs[panel_idx].legend(h, l)

        panel_idx += 1


    plt.savefig(job_path / f'output/plots/replot_vincenzo/figs/fig_9_replot.{pic_format}', dpi=dpi)
