"""
module to reproduce vincenzo's plots in the paper, *after* the AA referee comments (july 2024)
"""
import sys
import time
from pathlib import Path

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter
from scipy import interpolate
import numpy as np
import warnings
import pandas as pd

warnings.filterwarnings('ignore', category=SyntaxWarning, message=r'invalid escape sequence')


sys.path.append('/home/davide/Documenti/Lavoro/Programmi/Spaceborne')


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

save_figs = True

# ! =========================================== Fig(s). 7 ================================================================
uncert_df = pd.read_pickle(f'{fm_folder}/uncert_df_Opt_marg_vs_cond_for_sylv_barplots.pkl')
uncert_df.drop_duplicates(inplace=True)
npar = 8
fix_shear_bias = False
probes = ['WL', 'GC', '3x2pt']
for probe in probes:

    if fix_shear_bias:
        probe = 'WL'

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
        (uncert_df['which_cov'] != f'FM_{probe}_G') &
        (uncert_df['which_uncertainty'] == 'marginal') &
        (uncert_df['fix_shear_bias'] == fix_shear_bias)
    ].loc[:, 'FoM'].values

    x = (np.arange(npar) + 1) * 2
    wid = 0.6

    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(15, 10))

    if fix_shear_bias:
        ax.set_title(f'{probe} w/ fixed shear bias', fontsize=34, pad=15, loc='left')
    else:
        ax.set_title(probe_tex_dict[probe], fontsize=34, pad=15, loc='left')

    ax.barh(x + wid / 2, Marg_err_G, height=wid, color='lightcoral', label='G, marginal', edgecolor='k')
    ax.barh(x + wid / 2, UnMarg_err_G, height=wid, color='firebrick', label='G, conditional', edgecolor='k')
    ax.barh(x - wid / 2, Marg_err_GS, height=wid, color='skyblue', label='GS, marginal', edgecolor='k')
    ax.barh(x - wid / 2, UnMarg_err_GS, height=wid, color='dodgerblue', label='GS, conditional', edgecolor='k')
    # ax.barh(x[-1]+2+wid/2, FoM)

    ax.set_yticks(x)
    ax.set_yticklabels(params_latex_noFoM, rotation=45)

    ax.set_xlabel('$\\bar{\\sigma} \\; [\%]$', fontsize=46)
    ax.set_xscale('log')

    ax.tick_params(direction='in', which='both', labelsize=38, pad=10)
    ax.xaxis.set_ticks_position('both')
    # ax.xaxis.set_minor_locator(AutoMinorLocator())

    ax.set_axisbelow(True)
    plt.grid(axis='x', which='both')
    if probe in ['WL'] and not fix_shear_bias:
        plt.legend(fontsize=36, ncol=2, bbox_to_anchor=(1.02, 1.27))

    if save_figs:
        plt.savefig(f'{fm_folder}/plots/barplot_{probe}_v2.{pic_format}', dpi=dpi, bbox_inches='tight')


# * same plot but for the FoM
fom_dict = {}
for probe in probes:
    for which_cov in ['G', 'GSSC']:

        fom_dict[f'fom_{which_cov}'] = uncert_df[
            (uncert_df['which_cov'].str.startswith('FM_') & uncert_df['which_cov'].str.endswith(f'_{which_cov}')) &
            (uncert_df['which_uncertainty'] == 'marginal') &
            (uncert_df['fix_shear_bias'] == fix_shear_bias)
        ].loc[:, 'FoM'].values

x = (np.arange(len(probes)) + 1) * 2
wid = 0.6

fig, ax = plt.subplots(1, 1, sharex=True, figsize=(13, 8))

ax.barh(x + wid / 2, fom_dict['fom_G'], height=wid, color='lightcoral', label='G, marginal', edgecolor='k')
ax.barh(x - wid / 2, fom_dict['fom_GSSC'], height=wid, color='skyblue', label='GS, marginal', edgecolor='k')

ax.set_yticks(x)
ax.set_yticklabels(list(probe_tex_dict.values()), rotation=45)

ax.set_xlabel('FoM', fontsize=46)
ax.set_xscale('log')
ax.legend(fontsize=28, ncol=2)

ax.tick_params(direction='in', which='both', labelsize=35, pad=10)
ax.xaxis.set_ticks_position('both')
# ax.xaxis.set_minor_locator(AutoMinorLocator())

ax.set_title(f' ', fontsize=34, pad=15, loc='left')
ax.set_axisbelow(True)
plt.grid(axis='x', which='both')
if probe in ['WL'] and not fix_shear_bias:
    plt.legend(fontsize=40, ncol=2, bbox_to_anchor=(1.02, 1.27))

if save_figs:
    plt.savefig(f'{fm_folder}/plots/barplot_FoM_v2.{pic_format}', dpi=dpi, bbox_inches='tight')


# ! =========================================== Fig. 9 ================================================================

# Probes and colors
probes = ['WL', '3x2pt']
opt_pes_list = ['Pes', 'Opt']
colors = ['tab:blue', 'tab:blue', 'tab:green', 'tab:green']
uncert_df = pd.read_pickle(f'{fm_folder}/uncert_df_variable_zbins.pkl')

fom_opt_3x2pt_g_ref = uncert_df[
    (uncert_df['zbins'] == 10) &
    (uncert_df['opt_pes'] == 'Opt') &
    (uncert_df['which_cov'] == 'FM_3x2pt_G')
]['FoM'].values[0]
print(f'Reference FoM for EP10 flat optimistic G 3x2pt case: {int(fom_opt_3x2pt_g_ref)}')

# Create subplots
fig, axs = plt.subplots(3, 3, sharex=True, subplot_kw=dict(box_aspect=0.6),
                        constrained_layout=True, figsize=(10.5, 6.5),)  # tight_layout={'pad': 0.4})

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

if save_figs:
    plt.savefig(f'{fm_folder}/plots/GS_G_ratio_vs_zbins_jul24.{pic_format}', dpi=dpi, bbox_inches='tight')
plt.show()

# ! =========================================== Fig. 10 ================================================================

uncert_df = pd.read_pickle(f'{fm_folder}/uncert_df_opt_ep_or_ed_vs_zbins.pkl')

# Create subplots
common_figsize = (13, 5)  # for plots 10 and 11
fig, axs = plt.subplots(1, 2, figsize=common_figsize, constrained_layout=False, tight_layout={'pad': 2})
points_per_inch = 72 / fig.dpi
labelsize = 23 * points_per_inch
legendsize = 20 * points_per_inch
ticks_size = 20 * points_per_inch

# Probes and colors
ep_ed_values = ['EP', 'ED']
colors = ['tab:blue', 'tab:orange']
linestyle = 'dashed'
opt_pes = 'Opt'
fom_values_dict = {}

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

        # Extract FoM values
        fom_values_dict[f'{probe}_{ep_ed}'] = np.abs((_uncert_df['FoM'].values / 100) - 1)

        # Plot data
        axs[probe_idx].plot(NbZed, fom_values_dict[f'{probe}_{ep_ed}'], ls=linestyle, markersize=7, marker='o', color=colors[ep_ed_idx],
                            label=f'{ep_ed}')

        axs[probe_idx].yaxis.set_major_formatter(FormatStrFormatter(f'{fmt}'))
        axs[probe_idx].xaxis.set_major_formatter(FormatStrFormatter('%d'))
        axs[probe_idx].set_xticks(NbZed)
        axs[probe_idx].legend(fontsize=legendsize)
        axs[probe_idx].set_xlabel('${\\cal N}_\\mathrm{b}$', fontsize=labelsize)
        axs[probe_idx].set_ylabel('$\\mathcal{R}(\\mathrm{FoM}) \\; \\; {\\rm %s}$' %
                                  probe_tex_dict[probe].replace('$', ''), fontsize=labelsize)
        axs[probe_idx].grid(True)
        axs[probe_idx].tick_params(labelsize=ticks_size)  # Convert 30 points to inches

# compute EP vs ED variation
print('WL EP vs ED variation:\n', (fom_values_dict['WL_EP'] / fom_values_dict['WL_ED'] - 1) * 100)
print('3x2pt EP vs ED variation:\n', (fom_values_dict['3x2pt_EP'] / fom_values_dict['3x2pt_ED'] - 1) * 100)

# compute EP and ED vs zbins variation
for probe in ['WL', '3x2pt']:
    for ep_ed in ep_ed_values:
        print(f'{probe} {ep_ed} variation:\n', np.diff(
            fom_values_dict[f'{probe}_{ep_ed}']) / fom_values_dict[f'{probe}_{ep_ed}'][:-1] * 100)

if save_figs:
    plt.savefig(f'{fm_folder}/plots/FoM_vs_EP-ED_zbins_v2.{pic_format}', dpi=dpi, bbox_inches='tight')
plt.show()


# ! =========================================== Fig. 11 ================================================================

uncert_df = pd.read_pickle(f'{fm_folder}/uncert_df_zbinsEP10_fom_vs_epsilonb.pkl')
eps_b_list = np.unique(uncert_df['epsilon_b'].values) * 100  # it's in percent units
shear_bias_prior_list = np.unique(uncert_df['shear_bias_prior'].values)
ls_list = ['-', '--', ':']
colors = ['tab:blue', 'tab:orange']

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=common_figsize, constrained_layout=True, tight_layout={'pad': 2})
points_per_inch = 72 / fig.dpi

# Plot FoM for both G and GS in the left subplot
for idx, shear_bias_prior in enumerate(shear_bias_prior_list):
    fom_g = uncert_df[
        (uncert_df['which_cov'] == 'FM_3x2pt_G') &
        (uncert_df['probe'] == '3x2pt') &
        (uncert_df['shear_bias_prior'] == shear_bias_prior)
    ]['FoM'].values
    fom_gs = uncert_df[
        (uncert_df['which_cov'] == 'FM_3x2pt_GSSC') &
        (uncert_df['probe'] == '3x2pt') &
        (uncert_df['shear_bias_prior'] == shear_bias_prior)
    ]['FoM'].values

    axs[0].semilogx(eps_b_list, fom_g, ls=ls_list[idx], c=colors[0], label=f'G $\sigma_m={shear_bias_prior}$')
    axs[0].semilogx(eps_b_list, fom_gs, ls=ls_list[idx], c=colors[1], label=f'GS $\sigma_m={shear_bias_prior}$')
    axs[1].semilogx(eps_b_list, fom_gs / fom_g, ls=ls_list[idx], c='k', label=f'$\sigma_m={shear_bias_prior}$')

# Set labels and legend for the left subplot
axs[0].set_xlabel('$\\epsilon_b \\, [\%]$', fontsize=labelsize)
axs[1].set_xlabel('$\\epsilon_b \\, [\%]$', fontsize=labelsize)
axs[0].set_ylabel('${\\rm FoM}  \\; \\; {\\rm 3}\\times 2 {\\rm pt}$', fontsize=labelsize)
axs[1].set_ylabel('$\\mathcal{R}(\\mathrm{FoM}) \\; \\; {\\rm 3}\\times 2 {\\rm pt}$', fontsize=labelsize)
axs[0].grid(True)
axs[1].grid(True)
axs[0].tick_params(labelsize=ticks_size)
axs[1].tick_params(labelsize=ticks_size)

# legends
line_styles = [Line2D([0], [0], color='k', lw=2, linestyle=ls) for ls in ls_list]
line_colors = [Line2D([0], [0], color=color, lw=2) for color in colors]

combined_legend = line_styles + line_colors
combined_labels = [r'$\sigma_m=5 \times 10^{-4}$',
                   r'$\sigma_m=50 \times 10^{-4}$', r'$\sigma_m=500 \times 10^{-4}$', 'G', 'GS']

legend = axs[0].legend(combined_legend, combined_labels, loc='upper right')

# line_styles = [Line2D([0], [0], color='k', lw=2, linestyle=ls) for ls in ls_list]
# line_colors = [Line2D([0], [0], color=color, lw=2) for color in colors]
# legend1 = axs[0].legend(line_styles, [r'$\sigma_m=5 \times 10^{-4}$', r'$\sigma_m=50 \times 10^{-4}$', r'$\sigma_m=500 \times 10^{-4}$'], loc='upper right')
# legend2 = axs[0].legend(line_colors, ['G', 'GS'], loc='lower right')
# axs[0].add_artist(legend1)
legend1 = axs[1].legend(line_styles, [
                        r'$\sigma_m=5 \times 10^{-4}$', r'$\sigma_m=50 \times 10^{-4}$', r'$\sigma_m=500 \times 10^{-4}$'], loc='upper right')
axs[1].add_artist(legend1)

if save_figs:
    plt.savefig(f'{fm_folder}/plots/FoM_vs_epsb_and_ratio_v2.pdf', dpi=500, bbox_inches='tight')
plt.show()


# ! =========================================== Fig. 12 ================================================================

uncert_df = pd.read_pickle(f'{fm_folder}/uncert_df_zbinsEP10_fom_vs_epsilonbANDsigmam_isocontour.pkl')
# clean df
uncert_df = uncert_df.drop(columns=['gal_bias_prior'])
uncert_df = uncert_df[
    (uncert_df['which_cov'] == 'FM_3x2pt_GSSC') &
    (uncert_df['probe'] == '3x2pt')
]
uncert_df = uncert_df[['shear_bias_prior', 'epsilon_b', 'FoM']]

# Generate grid data for contour plot
eps_b_values = np.unique(uncert_df['epsilon_b'].values)
sigma_m_values = np.unique(uncert_df['shear_bias_prior'].values)
eps_b_grid, sigma_m_grid = np.meshgrid(eps_b_values, sigma_m_values)

# Pivoting to create grid values for FoM
fom_gs_grid = uncert_df.pivot(index='shear_bias_prior', columns='epsilon_b', values='FoM').values
levels = [0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10]
xlim = (0, 0.1)


plt.scatter(eps_b_grid * 100, sigma_m_grid * 1e4, c=fom_gs_grid / fom_opt_3x2pt_g_ref,
            cmap='plasma', s=30)
plt.xlim(xlim)
plt.grid()
norm = plt.Normalize(vmin=levels[0], vmax=levels[-1])
sm = plt.cm.ScalarMappable(cmap='plasma', norm=norm)
sm.set_array([])
plt.colorbar(sm, ax=ax, label='FoM$_{\\rm GS}$/FoM$_{\\rm ref}$')
plt.show()

# Create contour plot
fig, ax = plt.subplots(figsize=(10, 10))

contour = ax.contour(eps_b_grid * 100, sigma_m_grid * 1e4, fom_gs_grid / fom_opt_3x2pt_g_ref,
                     levels=levels, cmap='plasma')

# Set labels
ax.set_xlabel('$\\epsilon_b \\; [\\%]$', fontsize=15)
ax.set_ylabel('$\\sigma_m \\times 10^{4}$', fontsize=15)
ax.set_xlim(xlim)
# ax.set_ylim(0, 10)

legend_elements = [plt.Line2D([0], [0], color=contour.cmap(contour.norm(level)), lw=2,
                              label=f'FoM$_{{\\rm GS}}$/FoM$_{{\\rm ref}}$ = {level:.2f}')
                   for level in levels]
ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
ax.grid()

# plt.savefig('isocontour_plot.pdf', dpi=300, bbox_inches='tight')
plt.show()

print(f'sigma_m_values.min(): \t {sigma_m_values.min()}')
print(f'sigma_m_values.max(): \t {sigma_m_values.max()}')
print(f'eps_b_values.min(): \t {eps_b_values.min()}')
print(f'eps_b_values.max(): \t {eps_b_values.max()}')
eps_b_triplet = np.array((0.01, 0.05, 1)) / 100  # not in percent units
sigma_m_triplet = (0.5e-4, 5e-4, 10e-4)
from pynverse import inversefunc


# with RegularGridInterpolator
f = interpolate.RegularGridInterpolator((sigma_m_values, eps_b_values),
                                        fom_gs_grid / fom_opt_3x2pt_g_ref, 
                                        method='linear')
eps_b_xx, sigma_m_yy = np.meshgrid(eps_b_triplet, sigma_m_triplet)
# the rows of the result correspond to different fixed values of sigma_m
fom_gs_over_fom_ref = f((sigma_m_yy, eps_b_xx))
print(f'FoM_GS/FoM for eps_b = {eps_b_triplet[0]*100} %: \t',fom_gs_over_fom_ref.T[0])
print(f'FoM_GS/FoM for eps_b = {eps_b_triplet[1]*100} %: \t',fom_gs_over_fom_ref.T[1])
print(f'FoM_GS/FoM for eps_b = {eps_b_triplet[2]*100} %: \t',fom_gs_over_fom_ref.T[2])
print(f'\t\t for sigma_m: \t ', sigma_m_triplet)


#  ! redo eps_b = {... table
for sigma_m_tofix in sigma_m_triplet:
    z_values = (0.8, 0.9, 1)
    # this is a function of eps_b only, because pyinverse works in 1d
    def f_fixed_sigmam(epsb): return f((sigma_m_tofix, epsb))
    # without specifying the domani it gives interpolation issues
    eps_b_vals = inversefunc(f_fixed_sigmam, y_values=z_values, domain=[eps_b_values.min(), eps_b_values.max()])
    print(f'eps_b_vals for sigma_m = {sigma_m_tofix}: {eps_b_vals*100} [%]')


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
