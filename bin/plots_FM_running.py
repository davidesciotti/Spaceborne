import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
from pathlib import Path
import pandas as pd
import array_to_latex as a2l
import plotly.graph_objects as go
import plotly.offline as pyo
import getdist
from getdist import plots
from getdist.gaussian_mixtures import GaussianND
from matplotlib import cm

project_path_here = Path.cwd().parent.parent.parent
sys.path.append(f'{project_path_here}/lib')
import my_module as mm

sys.path.append(f'{project_path_here}/config')
# import ISTF_fid_params
import mpl_cfg

matplotlib.rcParams.update(mpl_cfg.mpl_rcParams_dict)

param_names_label = mpl_cfg.general_dict['cosmo_labels_TeX']
ylabel_perc_diff_wrt_mean = mpl_cfg.general_dict['ylabel_perc_diff_wrt_mean']
ylabel_sigma_relative_fid = mpl_cfg.general_dict['ylabel_sigma_relative_fid']
# plt.rcParams['axes.axisbelow'] = True
markersize = mpl_cfg.mpl_rcParams_dict['lines.markersize']


###############################################################################
################## CODE TO PLOT THE FISHER CONSTRAINTS ########################
###############################################################################


def get_kv_pairs(folder):
    from pathlib import Path
    for path in Path(folder).glob("*.txt"):
        yield path.stem, np.genfromtxt(str(path))


def plot(array, style=".-"):
    name = mm.namestr(array, globals())
    plt.plot(range(7), array, style, label=name)


def bar_plot_old(uncert_gauss, uncert_SSC, difference):
    labels = ["$\Omega_m$", "$\Omega_b$", "$w_0$", "$w_a$", "$h$", "$n_s$", "$\sigma_8$"]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()

    ax.bar(x - width / 2, uncert_gauss, width, color="mediumseagreen", label='Gauss only')
    ax.bar(x + width / 2, uncert_SSC, width, color="tomato", label='Gauss + SSC')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('relative uncertainties $\sigma/ \\theta_{fid}$')
    ax.set_title(f'FM 1-$\sigma$ parameter constraints, {probe} - lower is better')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel("% uncertainty increase", color=color)  # we already handled the x-label with ax1
    ax2.plot(range(7), difference, "o--", color=color, markersize=markersize)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    ax.grid()
    plt.show()
    plt.savefig(fname=f'bar_plot_{probe}.png', dpi=300, figsize=[16, 9])


def bar_plot_bu(data, title, label_list, bar_width=0.18, nparams=7, param_names_label=param_names_label,
             second_axis=True, no_second_axis_bars=1):
    """
    data: usually the percent uncertainties, but could also be the percent difference
    """

    plt.rc('axes', axisbelow=True)  # grid behind the bars

    # colors = cm.Paired(np.linspace(0, 1, data.shape[1]))

    # Set position of bar on x-axis
    bar_centers = np.zeros(data.shape)

    if data.ndim == 1:  # just one vector
        data = np.expand_dims(data, 0)
        bar_centers = np.arange(data.shape[1])
        bar_centers = np.expand_dims(bar_centers, 0)

    else:
        for bar_idx in range(data.shape[0]):
            if bar_idx == 0:
                bar_centers[bar_idx, :] = np.arange(data.shape[1]) - bar_width
            else:
                bar_centers[bar_idx, :] = [x + bar_idx * bar_width for x in bar_centers[0]]

    # plt.grid()

    if second_axis:


        # assert data.shape[0] == 3, "data must have 3 rows to display the second axis"

        # plt.rcParams['axes.axisbelow'] = True

        fig, ax = plt.subplots(figsize=mpl_cfg.mpl_rcParams_dict['figure.figsize'])
        for bar_idx in range(data.shape[0] - no_second_axis_bars):
            ax.bar(bar_centers[bar_idx, :], data[bar_idx, :], width=bar_width, edgecolor='grey', label=label_list[bar_idx])

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.grid()
        ax.set_ylabel(ylabel_sigma_relative_fid)
        ax.set_title(title)
        ax.set_xticks(range(nparams), param_names_label)

        # second axis
        ax2 = ax.twinx()
        # ax2.set_ylabel('(GS/GO - 1) $\\times$ 100', color='g')
        ax2.set_ylabel('% uncertainty increase')
        for bar_idx in range(1, no_second_axis_bars + 1):
            ax2.bar(bar_centers[-bar_idx, :], data[-bar_idx, :], width=bar_width, edgecolor='grey',
                    label=label_list[-bar_idx], color='g')
        ax2.tick_params(axis='y')

        fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)

    else:
        plt.figure(figsize=mpl_cfg.mpl_rcParams_dict['figure.figsize'])
        plt.grid()

        # Make the plot
        for bar_idx in range(data.shape[0]):
            plt.bar(bar_centers[bar_idx, :], data[bar_idx, :], width=bar_width, edgecolor='grey', label=label_list[bar_idx])

        # Adding xticks
        plt.ylabel(ylabel_sigma_relative_fid)
        plt.xticks(range(nparams), param_names_label)

        plt.title(title)
        plt.legend()
        plt.show()


def bar_plot(data, title, label_list, bar_width=0.18, nparams=7, param_names_label=param_names_label,
             second_axis=False, no_second_axis_bars=0):
    """
    data: usually the percent uncertainties, but could also be the percent difference
    """

    plt.rc('axes', axisbelow=True)  # grid behind the bars

    # colors = cm.Paired(np.linspace(0, 1, data.shape[1]))

    # Set position of bar on x-axis
    bar_centers = np.zeros(data.shape)

    if data.ndim == 1:  # just one vector
        data = np.expand_dims(data, 0)
        bar_centers = np.arange(data.shape[1])
        bar_centers = np.expand_dims(bar_centers, 0)

    else:
        for bar_idx in range(data.shape[0]):
            if bar_idx == 0:
                bar_centers[bar_idx, :] = np.arange(data.shape[1]) - bar_width
            else:
                bar_centers[bar_idx, :] = [x + bar_idx * bar_width for x in bar_centers[0]]

    # plt.grid()

    if second_axis:


        # assert data.shape[0] == 3, "data must have 3 rows to display the second axis"

        # plt.rcParams['axes.axisbelow'] = True

        fig, ax = plt.subplots(figsize=mpl_cfg.mpl_rcParams_dict['figure.figsize'])
        for bar_idx in range(data.shape[0] - no_second_axis_bars):
            ax.bar(bar_centers[bar_idx, :], data[bar_idx, :], width=bar_width, edgecolor='grey', label=label_list[bar_idx])

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.grid()
        ax.set_ylabel(ylabel_sigma_relative_fid)
        ax.set_title(title)
        ax.set_xticks(range(nparams), param_names_label)

        # second axis
        ax2 = ax.twinx()
        # ax2.set_ylabel('(GS/GO - 1) $\\times$ 100', color='g')
        ax2.set_ylabel('% uncertainty increase')
        for bar_idx in range(1, no_second_axis_bars + 1):
            ax2.bar(bar_centers[-bar_idx, :], data[-bar_idx, :], width=bar_width, edgecolor='grey',
                    label=label_list[-bar_idx], color='g')
        ax2.tick_params(axis='y')

        fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)

    else:
        plt.figure(figsize=mpl_cfg.mpl_rcParams_dict['figure.figsize'])
        plt.grid()

        # Make the plot
        for bar_idx in range(data.shape[0]):
            plt.bar(bar_centers[bar_idx, :], data[bar_idx, :], width=bar_width, edgecolor='grey', label=label_list[bar_idx])

        # Adding xticks
        plt.ylabel(ylabel_sigma_relative_fid)
        plt.xticks(range(nparams), param_names_label)

        plt.title(title)
        plt.legend()
        plt.show()


def triangle_plot(FM_GO, FM_GS, fiducials, title, param_names_label):
    # should I do this?
    fiducials = np.where(fiducials == 0., 1,
                         fiducials)  # the fiducial for wa is 0, substitute with 1 to avoid division by zero
    fiducials = np.where(fiducials == -1, 1,
                         fiducials)  # the fiducial for wa is -1, substitute with 1 to avoid negative values

    nparams = len(param_names_label)

    # parameters' covariance matrix - first invert, then slice! Otherwise, you're fixing the nuisance parameters
    FM_inv_GO = np.linalg.inv(FM_GO)[:nparams, :nparams]
    FM_inv_GS = np.linalg.inv(FM_GS)[:nparams, :nparams]

    GO_gaussian = GaussianND(mean=fiducials, cov=FM_inv_GO, names=param_names_label)
    GS_gaussian = GaussianND(mean=fiducials, cov=FM_inv_GS, names=param_names_label)
    print(GS_gaussian, len(param_names_label), len(fiducials), FM_inv_GO.shape)
    g = plots.get_subplot_plotter()
    g.settings.linewidth = 2
    g.settings.legend_fontsize = 30
    g.settings.linewidth_contour = 2.5
    g.settings.axes_fontsize = 27
    g.settings.axes_labelsize = 30
    g.settings.subplot_size_ratio = 1
    g.settingstight_layout = True
    g.settings.solid_colors = 'tab10'
    g.triangle_plot([GS_gaussian, GO_gaussian], filled=True, contour_lws=1.4,
                    legend_labels=['Gauss + SSC', 'Gauss-only'], legend_loc='upper right')
    plt.suptitle(f'{title}', fontsize='xx-large')


# parametri fiduciali
fid = np.array((0.32, 0.05, 1, 1, 0.67, 0.96, 0.816, 0.55, 1, 1))
fidmn = np.array((0.32, 0.05, 1, 1, 0.67, 0.96, 0.816, 0.06, 0.55, 1))

IST = {}
IST['WL_pes'] = np.array((0.044, 0.47, 0.16, 0.59, 0.21, 0.038, 0.019))
IST['3x2pt_pes'] = np.array((0.011, 0.054, 0.042, 0.17, 0.029, 0.010, 0.0048))
IST['GC_pes'] = np.array((0, 0, 0, 0, 0, 0, 0))

IST['WL_opt'] = np.array((0.034, 0.42, 0.14, 0.48, 0.20, 0.030, 0.013))
IST['3x2pt_opt'] = np.array((0.0059, 0.046, 0.027, 0.10, 0.020, 0.0039, 0.0022))
IST['GC_opt'] = np.array((0, 0, 0, 0, 0, 0, 0))


# xxx attento, non stai normalizzando wa (come posso farlo?)
# params = [Om, Ob, wz, wa, h, ns, s8, mn, ga, 7, 8, 9, 10]
# xxx occhio a ga e mn

def plot_FM_constr(FM, label, uncert_kind='relative'):
    rel_uncert = mm.uncertainties_FM(FM)[:7]
    plt.plot(range(7), rel_uncert * 100, "--", marker='o', label=label, markersize=markersize)


def plot_FM(general_config, covariance_config, plot_config, FM_dict):
    # plot settings:
    params = plot_config['params']
    plt.rcParams.update(params)
    markersize = plot_config['markersize']
    dpi = plot_config['dpi']
    pic_format = plot_config['pic_format']
    plot_sylvain = plot_config['plot_sylvain']
    plot_ISTF = plot_config['plot_ISTF']
    custom_label = plot_config['custom_label']

    # import settings:
    nbl = general_config['nbl']
    ell_max_WL = general_config['ell_max_WL']
    ell_max_GC = general_config['ell_max_GC']
    zbins = general_config['zbins']
    Cij_folder = general_config['cl_folder']
    which_forecast = general_config['which_forecast']

    ind_ordering = covariance_config['ind_ordering']

    case = plot_config['case']
    probe = plot_config['probe']
    GO_or_GS = plot_config['GO_or_GS']
    covmat_dav_flag = plot_config['covmat_dav_flag']
    which_plot = plot_config['which_plot']

    param_names_label = plot_config['param_names_label']

    ell_max_XC = ell_max_GC
    ########################### import ######################################

    # output_folder = mm.get_output_folder(ind_ordering, which_forecast)

    ######################### OPTIONS
    # XXX
    # XXX
    # XXX
    """
    case = "opt"
    # case = "pes"
    
    probe = "3x2pt"
    probe = "GC"
    probe = "WL"
    
    GO_or_GS = "GS"
    # GO_or_GS = "GO"
    
    # covmat_dav_flag = "yes"                                                       # this is for GC FM computed by Sylvain using my covmat
    covmat_dav_flag = "no" 
    
    which_plot = "SSC_degradation_dav_vs_sylv"
    # which_plot = "SSC_degradation"
    which_plot = "dav_vs_sylv"
    which_plot = "bar_plot"
    # which_plot = "constraints_only"
    # which_plot = "none"
    
    """

    ############################### IMPORT FM DICTS ###############################

    # davide
    FM_dav_may = FM_dict

    # sylvain XXX renamed, careful!
    # XXX pay attention to the folder you're importing
    folder = project_path.parent / "common_data/sylvain/FM/common_ell_and_deltas/latest_downloads/renamed"
    FM_sylv_may = dict(mm.get_kv_pairs(folder, "txt"))

    # vincenzo may: to check whose GC is correct
    folder = project_path.parent / "common_data/vincenzo/14may/FishMat/EP10"
    FM_vinc_may = dict(mm.get_kv_pairs(folder, "dat"))

    ######################## COMPUTE UNCERTAINTIES ###############################
    probe_name = probe
    print(f'reminder: \nprobe = {probe} \nCij_folder = {Cij_folder}')

    if case == 'opt':
        if probe == "WL":
            ell_max = 5000
        else:
            ell_max = 3000
    elif case == 'pes':
        if probe == "WL":
            ell_max = 1500
        else:
            ell_max = 750

    # set ell_max name to "XC" in 3x2pt case (just naming)
    if probe != '3x2pt': probe_ell_max = probe
    if probe == '3x2pt': probe_ell_max = 'XC'

    ######### davide
    # XXX recheck
    # FM_dav_G   = FM_dav_may[f"FM_{probe}_G_lmax{probe_ell_max}{ell_max}_nbl{nbl}"]
    # FM_dav_SSC = FM_dav_may[f"FM_{probe}_G+SSC_lmax{probe_ell_max}{ell_max}_nbl{nbl}"]

    FM_dav_G = FM_dav_may[f"FM_{probe}_GO"]
    FM_dav_SSC = FM_dav_may[f"FM_{probe}_GS"]

    if probe == "GC" and covmat_dav_flag == "yes":  # sylvain's with my covmats
        FM_dav_G = FM_sylv_may[f"FM_{probe}_G_lmax{probe_ell_max}{ell_max}_nbl{nbl}_ellDavide_DavCovmat"]
        FM_dav_SSC = FM_sylv_may[f"FM_{probe}_G+SSC_lmax{probe_ell_max}{ell_max}_nbl{nbl}_ellDavide_DavCovmat"]

    ######### sylvain
    nbl = 30  # because in ISTF we used 30 ell bins
    FM_sylv_G = FM_sylv_may[f"FM_{probe}_G_lmax{probe_ell_max}{ell_max}_nbl{nbl}_ellDavide"]
    FM_sylv_SSC = FM_sylv_may[f"FM_{probe}_G+SSC_lmax{probe_ell_max}{ell_max}_nbl{nbl}_ellDavide"]
    nbl = general_config['nbl']  # back to correct number of bins

    ######### uncertainties
    uncert_dav_G = mm.uncertainties_FM(FM_dav_G)[:7]
    uncert_dav_SSC = mm.uncertainties_FM(FM_dav_SSC)[:7]

    uncert_sylv_G = mm.uncertainties_FM(FM_sylv_G)[:7]
    uncert_sylv_SSC = mm.uncertainties_FM(FM_sylv_SSC)[:7]

    # if which_forecast != 'sylvain':
    ######### SEYFERT
    # if probe == '3x2pt':
    # FM_SEYF_G = FM_dav_may[f"FM_{probe}_G_lmax{probe_ell_max}{ell_max}_nbl{nbl}_SEYFERT"]
    # FM_SEYF_SSC = FM_dav_may[f"FM_{probe}_G+SSC_lmax{probe_ell_max}{ell_max}_nbl{nbl}_SEYFERT"]
    # uncertainties
    # uncert_SEYF_G   = mm.uncertainties_FM(FM_SEYF_G)[:7]
    # uncert_SEYF_SSC = mm.uncertainties_FM(FM_SEYF_SSC)[:7]
    # else:
    #     uncert_SEYF_G = np.zeros(7)
    #     uncert_SEYF_SSC = np.zeros(7)

    ######### from 2DCLOE:
    # FM_2DCLOE_G = np.genfromtxt(path / f'output/FM/{output_folder}/{Cij_folder}/FM_2DCLOE_G.txt')
    # FM_2DCLOE_GpSSC = np.genfromtxt(path / f'output/FM/{output_folder}/{Cij_folder}/FM_2DCLOE_G+SSC.txt')
    # # uncertainties
    # uncert_2DCLOE_G =  mm.uncertainties_FM(FM_2DCLOE_G)[:7]
    # uncert_2DCLOE_GpSSC =  mm.uncertainties_FM(FM_2DCLOE_GpSSC)[:7]

    ################################## COMPUTE AND PLOT ###########################

    if GO_or_GS == "GO":
        dav = uncert_dav_G
        sylv = uncert_sylv_G
        # if which_forecast != 'sylvain': 
        # CLOE2D = uncert_2DCLOE_G
        # SEYF = uncert_SEYF_G

    elif GO_or_GS == "GS":
        dav = uncert_dav_SSC
        sylv = uncert_sylv_SSC
        # if which_forecast != 'sylvain': 
        # CLOE2D = uncert_2DCLOE_GpSSC
        # SEYF = uncert_SEYF_SSC

    ##################################  PLOT ######################################

    if which_plot == "dav_vs_sylv":

        mean = (sylv + dav) / 2
        # mean = (SEYF + dav)/2 ; print('attention, using SEYF instead of sylvain')
        diff_dav = mm.percent_diff(dav, mean)
        diff_sylv = mm.percent_diff(sylv, mean)

        # plt.plot(range(7), diff_dav, "o--", label = f"davide {GO_or_GS}") 
        # plt.plot(range(7), diff_sylv, "o--", label = f"sylvain {GO_or_GS}") 
        # plt.plot(range(7), diff_SEYF, "o--", label = f"SEYF. {GO_or_GS}") 

        plt.plot(range(7), diff_dav, "o", label=f"group A, {GO_or_GS}",
                 markersize=markersize)  # XXX delete, just different label
        plt.plot(range(7), diff_sylv, "o", label=f"group B, {GO_or_GS}",
                 markersize=markersize)  # XXX delete, just different label
        plt.fill_between(range(7), diff_dav, diff_sylv, color='gray', alpha=0.2)

        plt.ylabel("$ \\bar{\sigma}_\\alpha^i / \\bar{\sigma}^m_\\alpha -1$ [%]")
        # plt.title(f"FM forec., % diff w.r.t. mean, {probe}, {GO_or_GS}") # XXX ripristina
        plt.title(f"FM forec., % diff w.r.t. mean, {probe}, {case}.")  # XXX delete

    elif which_plot == "constraints_only":  # plot only constraints

        plt.plot(range(7), dav * 100, "o--", label=f"davide {custom_label} {GO_or_GS}")
        if plot_sylvain:
            plt.plot(range(7), sylv * 100, "o--", label=f"sylvain {custom_label} {GO_or_GS}")
        if plot_ISTF:
            plt.plot(range(7), IST[f'{probe}_{case}'] * 100, "o--", label=f"IST_{probe}_{case}")

        # if which_forecast != 'sylvain': 
        # if probe == '3x2pt': plt.plot(range(7), SEYF, "o--", label = f"SEYF {GO_or_GS}")
        # if probe == '3x2pt': plt.plot(range(7), CLOE2D, "o--", label = f"uncert_2DCLOE {GO_or_GS}")

        plt.ylabel("$ \\sigma_\\alpha/ \\theta_{fid} [\\%]$")
        plt.title(f"FM forec., {probe}, {case}.")




    elif which_plot == "SSC_degradation_dav_vs_sylv":  # plot only constraints

        ########## noSSC vs SSC, dav vs sylv

        diff_dav = mm.percent_diff(uncert_dav_SSC, uncert_dav_G)
        diff_sylv = mm.percent_diff(uncert_sylv_SSC, uncert_sylv_G)

        # mean = (diff_dav + diff_sylv)/2

        # diff_dav  = mm.percent_diff(diff_dav, mean) # XXX careful
        # diff_sylv = mm.percent_diff(diff_sylv, mean)    

        plt.plot(range(7), diff_dav, "o-", label=f"davide {probe}")
        plt.plot(range(7), diff_sylv, "o--", label=f"sylvain {probe}")

        plt.ylabel("$ \\sigma_{G+SSC}/ \\sigma_{G} -1$  [%]")
        plt.title(f"FM forec. SSC uncertainty increase, {probe}")

        ######################

    elif which_plot == "bar_plot":
        # davide
        diff = mm.percent_diff(uncert_dav_SSC, uncert_dav_G)
        bar_plot(uncert_dav_G, uncert_dav_SSC, diff)
        # sylvain, just to check - GC is different, remember! we agree on the
        # relative errors in the G and G + SSC cases, not on the % uncertainty increase!
        # diff = mm.percent_diff(uncert_sylv_SSC, uncert_sylv_G)
        # bar_plot(uncert_sylv_G, uncert_sylv_SSC, diff)


    elif which_plot == "SSC_degradation":
        ########## noSSC vs SSC
        diff_dav = mm.percent_diff(uncert_dav_SSC, uncert_dav_G)
        diff_sylv = mm.percent_diff(uncert_sylv_SSC, uncert_sylv_G)

        plt.plot(range(7), diff_dav, "o-", label=f"{probe}")
        # plt.plot(range(7), diff_sylv, "o--", label = f"sylvain {probe}")

        plt.ylabel("$ \\sigma_{G+SSC}/ \\sigma_{G} -1$  [%]")
        plt.title(f"FM forec. SSC uncertainty increase")

    elif which_plot == 'radar_plot':

        categories = param_names_label
        categories = [*categories, categories[0]]

        restaurant_1 = dav * 100
        restaurant_2 = sylv * 100
        restaurant_1 = [*restaurant_1, restaurant_1[0]]
        restaurant_2 = [*restaurant_2, restaurant_2[0]]

        fig = go.Figure(
            data=[
                go.Scatterpolar(r=restaurant_1, theta=categories, fill='toself',
                                name=f'davide {custom_label} {GO_or_GS}'),
                go.Scatterpolar(r=restaurant_2, theta=categories, fill='toself',
                                name=f'sylvain {custom_label} {GO_or_GS}'),
                # go.Scatterpolar(r=restaurant_3, theta=categories, fill='toself', name='Restaurant 3')
            ],
            layout=go.Layout(
                title=go.layout.Title(text='Restaurant comparison'),
                polar={'radialaxis': {'visible': True}},
                showlegend=True
            )
        )

        pyo.plot(fig)

    # cases
    # plot(uncert_mt, style = ".-")
    # plot(uncert_sylvain, style = "o-")
    # if SSC == "no" and probe != "GC":
    #     plot(IST)
    # plot(uncert_unif, style = "o--")
    # plot(uncert_GCph_3000, style = "o--")

    ############## FOM 
    FoM = mm.compute_FoM(FM_dav_G)
    print(f'FoM davide G {probe}:\t{FoM:.2f}')
    FoM = mm.compute_FoM(FM_dav_SSC)
    print(f'FoM davide GpSSC {probe}:\t{FoM:.2f}')

    FoM = mm.compute_FoM(FM_sylv_G)
    print(f'FoM sylvain G {probe}:\t{FoM:.2f}')
    FoM = mm.compute_FoM(FM_sylv_SSC)
    print(f'FoM sylvain GpSSC {probe}:\t{FoM:.2f}')

    """
    values_array = np.zeros((3,3))
    
    probes = ['WL', 'GC', '3x2pt']
    
    
    # XXX BROKEN, DAVIDE GC HAS NOT THE CORRECT FM!!
    i = 0
    for (probe, ell_max) in zip(probes, (ell_max_WL, ell_max_GC, ell_max_XC)):
        
        print(probe, ell_max)
        fom = mm.compute_FoM(FM_dav_may[f"FM_{probe}_lmax{probe}{ell_max}_nbl{nbl}"])
        fom = mm.compute_FoM(FM_sylv_may["FM_WL_lmax5000_ellDavide_gauss"])
    
    
        print(f'fom G {probe}, nbl = {nbl}, l_max = {ell_max}, dav:', fom)
        fom_SSC = mm.compute_FoM(FM_dav_may[f"FM_{probe}_SSC_lmax{probe}{ell_max}_nbl{nbl}"])
        print(f'fom G+SSC {probe}, nbl = {nbl}, l_max = {ell_max}, dav:', fom_SSC)
        perc_decrease = (fom/fom_SSC-1)*100
        print(perc_decrease)
        values_array[i,:] = np.asarray((fom, fom_SSC, perc_decrease))
        i += 1
    
    a2l.to_ltx(values_array, frmt = '{:6.1f}', arraytype = 'array')
    """

    # for probe
    # a2l.to_ltx(A, frmt = '{:6.2f}', arraytype = 'array')

    if which_plot != 'bar_plot':
        plt.legend()
        plt.tight_layout()
        plt.grid()
        plt.xticks(range(7), param_names_label)

        plt.show()
        # plt.yscale("log")

    # plt.savefig(path / f'output/plots/{which_plot}_{probe}_{GO_or_GS}.png', dpi = 500) #, figsize=[16,9])

    #################### ALTERNATIVE PLOT ##############

    ########## PRINT ################
    # print("sylvain WL: ", uncertainties(FM_WL_sylvain)[:7])
    # print("FM_XC_pess_master : ",  uncertainties(FM_XC_pess_master)[:7])
    # print("IST_XC_pess: ", IST_XC_pess)
    # print("perc_incr_sylvain: ", percent_diff(uncertainties(FM_WL_sylvain)[:7], SSC)[:7])

    # save on a file 
    ###############################################################################
    """
    header = 'params: [$\Omega_m$, $\Omega_b$, $w_0$, $w_a$, $h$, $n_s$, $\sigma_8$]'
    uncert = {
        'params_order': '[$\Omega_m$, $\Omega_b$, $w_0$, $w_a$, $h$, $n_s$, $\sigma_8$]',
        'uncert_dav_G': list(uncert_dav_G),
        'uncert_dav_SSC': list(uncert_dav_SSC),
        'uncert_SEYF_G': list(uncert_SEYF_G),
        'uncert_SEYF_SSC': list(uncert_SEYF_SSC)
        }
    
    # import json
    # json.dump(
    #     uncert,
    #     open(f"{path}/output/uncertainties.json", "w"))
    
    import pickle
    a_file = open(f"{path}/output/uncertainties.pkl", "wb")
    pickle.dump(uncert, a_file)
    a_file.close()
    a_file = open(f"{path}/output/uncertainties.pkl", "rb")
    output = pickle.load(a_file)
    print(output)
    
    # TODO do it with pd when you have time
    # d = {'col1': [1, 2], 'col2': [3, 4]}
    # df = pd.DataFrame(data=d)
    """
    ###############################################################################

    ############################# new bit: vincenzo's 14 may data
    """
    FM_vinc = FM_vinc_may[f"fm{probe.lower()}o-GR-Flat-eNLA-NA-GO-Opt-EP10"]
    if probe == "GC":
        FM_vinc = np.delete(FM_vinc, (7,8,9), 0)
        FM_vinc = np.delete(FM_vinc, (7,8,9), 1)
    
    uncert_vinc_G_may = uncertainties(FM_vinc)[:7]
    
    # plt.plot(range(7), uncert_vinc_G_may, "o--", label = "uncert_vinc_G_may new")
    # plt.plot(range(7), uncert_dav_G_may, "o--", label = "uncert_dav_G_may new")
    # plt.plot(range(7), uncert_sylv_G_may, "o--", label = "uncert_sylv_G_may new")
    
    array1 = uncert_vinc_G_may
    array2 = uncert_sylv_G_may
    
    mean = (array1 + array2)/2
    diff_array1 = mm.percent_diff(array1, mean)
    diff_array2 = mm.percent_diff(array2, mean)
    plt.plot(range(7), diff_array1, "o--", label = mm.namestr(array1, globals()))
    plt.plot(range(7), diff_array2, "o--", label = mm.namestr(array2, globals()))
    
    plt.legend()
    
    sys.exit()
    """
