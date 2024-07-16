"""
module to reproduce vincenzo's plots in the paper, *after* the AA referee comments (july 2024)
"""
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter
from scipy import interpolate
import numpy as np
import pandas as pd


import common_cfg.mpl_cfg as mpl_cfg
import bin.my_module as mm


def remove_row_col(F, start, stop):
    return np.delete(np.delete(F, slice(start, stop), 0), slice(start, stop), 1)


def select_FM(F, probe, curvature=False, shear_bias=False):
    if probe == 'WL':
        if shear_bias:
            F = remove_row_col(F, 11, 21)
            # np.delete(np.delete(F,slice(11,21),0),slice(11,21),1)
            F = F[:21, :21]
        else:
            F = F[:11, :11]

    elif probe == 'GC':
        F = remove_row_col(F, 8, 11)
        F = F[:18, :18]

    elif probe == 'XC':
        if shear_bias:
            F = F[:31, :31]
        else:
            F = F[:21, :21]

    if not curvature:
        F = np.delete(np.delete(F, 1, 0), 1, 1)

    return F


fm_folder = '/home/davide/Documenti/Lavoro/Programmi/common_data/Spaceborne/jobs/SSC_paper_final/output/FM/ell_cuts_False'
job_path = '/home/davide/Documenti/Lavoro/Programmi/Spaceborne/jobs/SSC_paper_final'
uncert_df = pd.read_pickle(f'{fm_folder}/uncert_df_variable_zbins.pkl')
params_latex = mpl_cfg.general_dict['cosmo_labels_TeX'] + ['FoM']
params_latex_noFoM = mpl_cfg.general_dict['cosmo_labels_TeX']
params_plain = ["Om", "Ob", "w0", "wa", "h", "ns", "sigma8", 'logT', "FoM"]
probe_tex_dict = {
    'WL': '${\\rm WL}$',
    'GC': '${\\rm GCph}$',
    '3x2pt': '${\\rm 3\\times 2pt}$'
}
panel_titles_fontsize = 17
pic_format = 'pdf'
fmt = '%.2f'
dpi = 500


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

# ! =========================================== Fig. 9 ================================================================

# Probes and colors
probes = ['WL', '3x2pt']
opt_pes_list = ['Pes', 'Opt']
colors = ['tab:blue', 'tab:blue', 'tab:green', 'tab:green']

# Create subplots
# fig, axs = plt.subplots(3, 3, sharex=True, subplot_kw=dict(box_aspect=0.6),
# constrained_layout=True, figsize=(15, 6.5),)# tight_layout={'pad': 0.4})
# plt.subplots_adjust(wspace=0.01)
fig, axs = plt.subplots(3, 3, sharex=True, subplot_kw=dict(box_aspect=0.6),
                        figsize=(10.5, 6.5),  # layout='constrained',
                        gridspec_kw={'wspace': 0.0001, 'hspace': 0.0002})

# number each axs box: 0 for [0, 0], 1 for [0, 1] and so forth
axs_idx = np.arange(0, 9, 1).reshape((3, 3))

# loop over 9 parameters
for param_idx, param in enumerate(params_plain):
    # loop over probes and optimization scenarios
    for probe_idx, probe in enumerate(probes):
        for opt_idx, (_opt_pes, color) in enumerate(zip(opt_pes_list, colors[2 * probe_idx:2 * probe_idx + 2])):

            # Filter DataFrame
            _uncert_df = uncert_df[
                (uncert_df['which_cov'] == f'perc_diff_{probe}_G') &
                (uncert_df['opt_pes'] == _opt_pes)
            ]

            if _opt_pes == 'Opt':
                alpha = 1
                ls = '--'
            else:
                alpha = 0.6
                ls = ':'

            NbZed = _uncert_df['zbins'].values
            if param == 'FoM':
                param_values = np.abs((_uncert_df[param].values / 100) - 1)
            else:
                param_values = (_uncert_df[param].values / 100) + 1

            # Get subplot indices
            i, j = np.where(axs_idx == param_idx)[0][0], np.where(axs_idx == param_idx)[1][0]

            # Plot data
            axs[i, j].plot(NbZed, param_values, markersize=markersize, marker='o', color=color,
                           label=f'{probe} {_opt_pes}', alpha=alpha, ls=ls)
            axs[i, j].yaxis.set_major_formatter(FormatStrFormatter(f'{fmt}'))
            axs[i, j].xaxis.set_major_formatter(FormatStrFormatter('%d'))
            axs[i, j].set_xticks(NbZed)

    axs[i, j].grid()
    axs[i, j].set_title(f'{params_latex[param_idx]}', pad=10.0, fontsize=panel_titles_fontsize)

# Legend in the bottom
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)

fig.supxlabel('${\\cal N}_\\mathrm{b}$')
fig.supylabel('${\\cal R}(x) = \\sigma_{\\rm GS}(x) \\, / \\, \\sigma_{\\rm G}(x)$', x=-0.02)

# fig.tight_layout()

plt.savefig(f'{fm_folder}/plots/GS_G_ratio_vs_zbins_jul24.{pic_format}', dpi=dpi, bbox_inches='tight')
plt.show()

# ! =========================================== Fig. 10 ================================================================

uncert_df = pd.read_pickle(f'{fm_folder}/uncert_df_opt_ep_or_ed_vs_zbins.pkl')
# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)

# Probes and colors
ep_ed_values = ['EP', 'ED']
colors = ['tab:blue', 'tab:orange']
linestyle = 'dashed'
opt_pes = 'Opt'

# Filter DataFrame and plot for each subplot
for probe_idx, probe in enumerate(probes):
    for ep_ed_idx, ep_ed in enumerate(ep_ed_values):

        # Filter DataFrame
        _uncert_df = uncert_df[
            (uncert_df['EP_or_ED'] == ep_ed) &
            (uncert_df['opt_pes'] == opt_pes) &
            (uncert_df['which_cov'] == f'perc_diff_{probe}_G')
        ]

        # Plot data for each "Nbins"
        NbZed = _uncert_df['zbins'].unique()

        # Loop over "Nbins" values
        # for zbin in NbZed:
        # _data = _uncert_df[_uncert_df['zbins'] == zbin]

        # Extract FoM values
        fom_values = np.abs((_uncert_df['FoM'].values / 100) - 1)

        # Plot data
        axs[probe_idx].plot(NbZed, fom_values, ls=linestyle, markersize=markersize, marker='o', color=colors[ep_ed_idx],
                            label=f'{ep_ed}')

        axs[probe_idx].yaxis.set_major_formatter(FormatStrFormatter(f'{fmt}'))
        axs[probe_idx].xaxis.set_major_formatter(FormatStrFormatter('%d'))
        axs[probe_idx].set_xticks(NbZed)
        axs[probe_idx].legend()
        axs[probe_idx].set_xlabel('${\\cal N}_\\mathrm{b}$')
        axs[probe_idx].set_ylabel('$\\mathcal{R}(\\mathrm{FoM}) \\, , \\; {\\rm %s}$' %
                                  probe_tex_dict[probe].replace('$', ''))
        axs[probe_idx].grid(True)


# fig.supylabel('$\\mathcal{R}(\\mathrm{FoM})$')

# Legend
# handles, labels = axs[0].get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2)

plt.savefig(f'{fm_folder}/plots/FoM_vs_EP-ED_zbins_v2.{pic_format}', dpi=dpi, bbox_inches='tight')
plt.show()

# ! =========================================== Fig(s). 7 ================================================================
uncert_df = pd.read_pickle(f'{fm_folder}/uncert_df_Opt_marg_vs_cond_for_sylv_barplots.pkl')
uncert_df.drop_duplicates(inplace=True)
npar = 8
fix_shear_bias = False

xlim_dict = {
    'WL': [0, 60],
    'GC': [0, 23],
    '3x2pt': [0, 5],
}

for probe in ['WL', 'GC', '3x2pt']:

    Marg_err_G = uncert_df[
        (uncert_df['probe'] == probe) &
        (uncert_df['which_cov'] == f'FM_{probe}_G') &
        (uncert_df['which_uncertainty'] == 'marginal') &
        (uncert_df['fix_shear_bias'] == fix_shear_bias)
    ].loc[:, 'Om':'logT'].values[0]

    UnMarg_err_G = uncert_df[
        (uncert_df['probe'] == probe) &
        (uncert_df['which_cov'] == f'FM_{probe}_G') &
        (uncert_df['which_uncertainty'] == 'conditional') &
        (uncert_df['fix_shear_bias'] == fix_shear_bias)
    ].loc[:, 'Om':'logT'].values[0]

    Marg_err_GS = uncert_df[
        (uncert_df['probe'] == probe) &
        (uncert_df['which_cov'] == f'FM_{probe}_GSSC') &
        (uncert_df['which_uncertainty'] == 'marginal') &
        (uncert_df['fix_shear_bias'] == fix_shear_bias)
    ].loc[:, 'Om':'logT'].values[0]

    UnMarg_err_GS = uncert_df[
        (uncert_df['probe'] == probe) &
        (uncert_df['which_cov'] == f'FM_{probe}_GSSC') &
        (uncert_df['which_uncertainty'] == 'conditional') &
        (uncert_df['fix_shear_bias'] == fix_shear_bias)
    ].loc[:, 'Om':'logT'].values[0]

    FoM = uncert_df[
        (uncert_df['probe'] == probe) &
        (uncert_df['which_cov'] != f'perc_diff_{probe}_G') &
        (uncert_df['which_uncertainty'] == 'conditional') &
        (uncert_df['fix_shear_bias'] == fix_shear_bias)
    ].loc[:, 'FoM'].values
    
    x = (np.arange(npar) + 1) * 2
    wid = 0.7

    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(15, 8))
    ax.set_title(probe_tex_dict[probe], fontsize=34, pad=15, loc='left')

    ax.barh(x + wid / 2, Marg_err_G, height=wid, color='lightcoral', label='G, marginal', edgecolor='k')
    ax.barh(x + wid / 2, UnMarg_err_G, height=wid, color='firebrick', label='G, conditional', edgecolor='k')
    ax.barh(x - wid / 2, Marg_err_GS, height=wid, color='skyblue', label='GS, marginal', edgecolor='k')
    ax.barh(x - wid / 2, UnMarg_err_GS, height=wid, color='dodgerblue', label='GS, conditional', edgecolor='k')
    # ax.barh(x[-1]+2+wid/2, FoM)

    ax.set_yticks(x)
    ax.set_yticklabels(params_latex_noFoM)

    # ax.set_xscale('log')
    ax.set_xlabel('$\\bar{\sigma} \\; [\%]$', fontsize=46)
    
        
    ax.set_xscale('log')

    # ax.set_xlim(xlim_dict[probe])

    ax.tick_params(direction='in', which='both', labelsize=34)
    ax.xaxis.set_ticks_position('both')
    ax.xaxis.set_minor_locator(AutoMinorLocator())

    ax.set_axisbelow(True)
    plt.grid(axis='x', which='both')
    if probe in ['WL']:
        plt.legend(fontsize=28, ncol=2, bbox_to_anchor=(1.02, 1.27))









    
"""
# ! sylavin's code for Fig. 6
curv = False
if curv:
    npar_cos = 8
else:
    npar_cos = 7

cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
ref='/renoir/gouyou/SSC_project/code_comparison/davide_FM/update_october22/davide_newFM_non-flat_'

Fisher_XC_noSSC_opt_Dav = select_FM(np.loadtxt(ref+'3x2pt_opt_GO.dat'), 'XC', curv, True)
Fisher_XC_SSC_opt_Dav = select_FM(np.loadtxt(ref+'3x2pt_opt_GS.dat'), 'XC', curv, True)
print(np.shape(Fisher_XC_SSC_opt_Dav))
#print(Fisher_XC_SSC_opt_Dav)
#print("")

Fisher_WL_noSSC_opt_Dav = select_FM(np.loadtxt(ref+'WLO_opt_GO.dat'), 'WL', curv, True)
Fisher_WL_SSC_opt_Dav = select_FM(np.loadtxt(ref+'WLO_opt_GS.dat'), 'WL', curv, True)
print(np.shape(Fisher_WL_SSC_opt_Dav))
#print(Fisher_WL_SSC_opt_Dav)
#print("")

Fisher_GC_noSSC_opt_Dav = select_FM(np.loadtxt(ref+'GCO_opt_GO.dat'), 'GC', curv, True)
Fisher_GC_SSC_opt_Dav = select_FM(np.loadtxt(ref+'GCO_opt_GS.dat'), 'GC', curv, True)
print(np.shape(Fisher_GC_SSC_opt_Dav))
#print(Fisher_GC_SSC_opt_Dav)
#print("")

FG = [Fisher_WL_noSSC_opt_Dav, Fisher_GC_noSSC_opt_Dav, Fisher_XC_noSSC_opt_Dav]
FSSC = [Fisher_WL_SSC_opt_Dav, Fisher_GC_SSC_opt_Dav, Fisher_XC_SSC_opt_Dav]

param_names=["$\mathcal{A}_\mathrm{IA}$", "$\eta_\mathrm{IA}$", "$\\beta_\mathrm{IA}$", 
        "$b_1$", "$b_2$", "$b_3$", "$b_4$", "$b_5$", "$b_6$", "$b_7$", "$b_8$", "$b_9$", "$b_{10}$",
        "$m_1$", "$m_2$", "$m_3$", "$m_4$", "$m_5$", "$m_6$", "$m_7$", "$m_8$", "$m_9$", "$m_{10}$"]
npar = [13, 10, 23]
x = []
x.append((np.arange(npar[0])+1)*2)
x.append((np.arange(npar[1])+4)*2)
x.append((np.arange(npar[2])+1)*2)

width = 0.35

probes=['WL', 'GCph', '$3\\times 2$pt']
nprobes=len(probes)
#Get marginalised errors
stdev = np.zeros((2, nprobes, 23))
for pro in range(nprobes):
    covG = np.linalg.inv(FG[pro])
    covSSC = np.linalg.inv(FSSC[pro])
    #print(np.diag(covG))
    #print(np.shape(np.diag(covG[7:7+npar[pro]])))

    stdev[0, pro, :npar[pro]] = np.sqrt(np.diag(covG)[npar_cos:npar_cos+npar[pro]])
    stdev[1, pro, :npar[pro]] = np.sqrt(np.diag(covSSC)[npar_cos:npar_cos+npar[pro]])

#Plot
fig, ax = plt.subplots(1,1, sharex=True, figsize=(24, 6))
print(x[2][13:])
print(stdev[1,0,3:])

ax.bar(x[0][:3]-width, (stdev[1,0,:3]/stdev[0,0,:3] - 1)*100, label=probes[0], width=width, color=cycle[0])
ax.bar(x[2][13:]-width, (stdev[1,0,3:13]/stdev[0,0,3:13] - 1)*100, width=width, color=cycle[0])
ax.bar(x[1], (stdev[1,1,:npar[1]]/stdev[0,1,:npar[1]] - 1)*100, label=probes[1], width=width, color=cycle[1])
ax.bar(x[2]+width, (stdev[1,2,:npar[2]]/stdev[0,2,:npar[2]] - 1)*100, label=probes[2], width=width, color=cycle[2])

print(stdev[1,0,:npar[0]], stdev[0,0,:npar[0]])
print(stdev[1,2,:npar[2]], stdev[0,2,:npar[2]])

ax.tick_params(direction='in', which='both')
ax.yaxis.set_ticks_position('both')
ax.yaxis.set_minor_locator(AutoMinorLocator())

ax.set_ylabel('$[\cal{R}(\\theta) -1]\\times 100 $ ', fontsize=40)
#ax.set_title("Nuisance parameters, Optimistic case", fontsize=22, pad=10)
ax.set_xticks(x[2])
ax.set_xticklabels(param_names)
ax.tick_params(labelsize=42)
#ax.set_xlim([1,27])
ax.set_ylim([0,18.5])
#ax.set_yscale('log')
#ax.set_title('Flat')
ax.legend(ncol=1, fontsize=30)
plt.grid(axis='y', which='both')
plt.rc('axes', axisbelow=True) 
plt.savefig("plots/davide_paper_update_shearbias_histo_nuisance_nbl20_Opt.pdf")

# ! sylavin's code for Fig. 7 and 8
x       = (np.arange(npar)+1)*2
wid     = 0.5

fig, ax = plt.subplots(1,1, sharex=True, figsize=(15, 8))

ax.set_title('$3\\times 2$pt', fontsize=34, pad=15, loc='left')

ax.barh(x+wid/2, Marg_err_3x2_G/param_values, height=wid, color='lightcoral', label='G, Marginalised')
ax.barh(x+wid/2, UnMarg_err_3x2_G/param_values, height=wid, color='firebrick', label='G, Unmarginalised')
ax.barh(x-wid/2, Marg_err_3x2_GS/param_values, height=wid, color='skyblue', label='GS, Marginalised')
ax.barh(x-wid/2, UnMarg_err_3x2_GS/param_values, height=wid, color='dodgerblue', label='GS, Unmarginalised')
ax.barh(x[-1]+2+wid/2, FoM())

ax.set_yticks(x)
ax.set_yticklabels(param_names)

#ax.set_xscale('log')
ax.set_xlabel('$\\bar{\sigma}$', fontsize=46)
ax.set_xlim(0, 0.17)

ax.tick_params(direction='in', which='both', labelsize=34)
ax.xaxis.set_ticks_position('both')
ax.xaxis.set_minor_locator(AutoMinorLocator())

ax.set_axisbelow(True)
plt.grid(axis='x', which='both')
plt.legend(fontsize=28, ncol=2, bbox_to_anchor=(1.02,1.27))
plt.savefig('plots/barplot_3x2pt.pdf', bbox_inches='tight')
"""
