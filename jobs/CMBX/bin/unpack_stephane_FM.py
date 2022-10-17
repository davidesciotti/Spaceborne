import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
from pathlib import Path

job_path = '/Users/davide/Documents/Lavoro/Programmi/SSC_restructured_v2/jobs/CMBX'
project_path = '/Users/davide/Documents/Lavoro/Programmi/SSC_restructured_v2'

sys.path.append(f'{project_path}/lib')
import my_module as mm

sys.path.append(f'{project_path}/config')
import ISTF_fid_params

sys.path.append(f'{project_path}/bin')
import plots_FM_running as plot_utils

matplotlib.use('Qt5Agg')

# mode:
# * p ("plus")  == CMB + Euclid as independent probes
# * a ("add")   == proper CMBxEuclid covariance but no cross-probes in data
# * c ("cross") == full CMBxEucld covariance and data

# for case in ['pess', 'opti']:
#     for Rl_str in ['_Rlconst', '_Rlvar']:
#         for cosmo_model in ['LCDM', 'w0waCDM']:

case = 'opti'
CMB_probe = 'CMB'  # == all CMB probes (T, E, phi) or just phi
CMB_obs = 'planck'  # "planck" or "SO" or "S4"
mode = 'p'
cosmo_model = 'w0waCDM'  # w0waCDM or LCDM
probe = '3x2pt'
Rl_str = '_Rlconst'  # _Rlvar or _Rlconst

if case == 'opti':
    lmax = 3000
elif case == 'pess':
    lmax = 750
else:
    raise ValueError('case not recognized')

FM_steph_CMBO = np.load(
    f'/Users/davide/Documents/Lavoro/Programmi/SSC_restructured_v2/jobs/CMBX/input/FM_stephane/'
    f'fish_{CMB_probe}-{CMB_obs}_flat_{cosmo_model}_max-bins_super-prec_21point.npz')
FM_steph_CMBX = np.load(
    f'/Users/davide/Documents/Lavoro/Programmi/SSC_restructured_v2/jobs/CMBX/input/FM_stephane/'
    f'fish_Euclid-{case}_{CMB_probe}-{CMB_obs}_mode-{mode}_flat_{cosmo_model}_max-bins_super-prec_21point.npz')
FM_steph_eucl = np.load(
    f'/Users/davide/Documents/Lavoro/Programmi/SSC_restructured_v2/jobs/CMBX/input/FM_stephane/'
    f'fish_Euclid-{case}_flat_{cosmo_model}_max-bins_super-prec_21point.npz')

FM_dav_dict = dict(
    mm.get_kv_pairs('/Users/davide/Documents/Lavoro/Programmi/SSC_restructured_v2/jobs/IST_forecast/output/FM', 'txt'))
FM_dav_GO = FM_dav_dict[f'FM_{probe}_GO_lmaxXC{lmax}_nbl30']
FM_dav_GS = FM_dav_dict[f'FM_{probe}_GS_lmaxXC{lmax}_nbl30{Rl_str}']

FM_sylv_GO = np.genfromtxt(
    f'/Users/davide/Documents/Lavoro/Programmi/common_data/sylvain/FM/common_ell_and_deltas/latest_downloads/renamed/FM_{probe}_G_lmaxXC{lmax}_nbl30_ellDavide.txt')

print('shape:', FM_steph_CMBO['fish'].shape, '\nparams:', FM_steph_CMBO['all_params'], '\nfid:', FM_steph_CMBO['me'])

if cosmo_model == 'LCDM':
    nparams = 5
    FM_dav_GO = np.delete(FM_dav_GO, (2, 3), axis=0)
    FM_dav_GO = np.delete(FM_dav_GO, (2, 3), axis=1)
    FM_dav_GS = np.delete(FM_dav_GS, (2, 3), axis=0)
    FM_dav_GS = np.delete(FM_dav_GS, (2, 3), axis=1)
    FM_sylv_GO = np.delete(FM_sylv_GO, (2, 3), axis=0)
    FM_sylv_GO = np.delete(FM_sylv_GO, (2, 3), axis=1)
    param_names_label = ["$\Omega_{{\\rm m},0}$", "$\Omega_{{\\rm b},0}$", "$h$", "$n_{\\rm s}$",
                         "$\sigma_8$"]
elif cosmo_model == 'w0waCDM':
    nparams = 7
    param_names_label = ["$\Omega_{{\\rm m},0}$", "$\Omega_{{\\rm b},0}$", "$w_0$", "$w_a$", "$h$", "$n_{\\rm s}$",
                         "$\sigma_8$"]
else:
    raise ValueError('cosmo_model not recognized')

# a test: bias params
"""
plt.figure()
unc_dav_GO = mm.uncertainties_FM(FM_dav_GO, 15, FM_steph_eucl['me'][:15])
unc_dav_GS = mm.uncertainties_FM(FM_dav_GS, 15, FM_steph_eucl['me'][:15])
plt.plot(mm.percent_diff(unc_dav_GS, unc_dav_GO))
"""
# assert 1 == 2

# reorganize FM! the param order is not the same
mm.matshow(FM_dav_GO, log=True, abs_val=True, title='orig')
FM_dav_resh = np.zeros(FM_dav_GO.shape)
FM_dav_resh[:7, :7] = FM_dav_GO.copy()[:7, :7]
FM_dav_resh[nparams:nparams + 10, nparams:nparams + 10] = FM_dav_GO.copy()[-10:, -10:]
FM_dav_resh[-3:, -3:] = FM_dav_GO.copy()[7:10, 7:10:]
FM_dav_resh[7:17, :7] = FM_dav_GO.copy()[10:, :7]
FM_dav_resh[:7, 7:17] = FM_dav_GO.copy()[:7, 10:]
FM_dav_resh[17:, :7] = FM_dav_GO.copy()[7:10, :7]
FM_dav_resh[:7, 17:] = FM_dav_GO.copy()[:7, 7:10]
FM_dav_resh[17:, 7:17] = FM_dav_GO.copy()[7:10, 10:]
FM_dav_resh[7:17, 17:] = FM_dav_GO.copy()[10:, 7:10]
mm.matshow(FM_dav_resh, log=True, abs_val=True, title='end')

# alternative way:
ia_pars_idx = [7, 8, 9]
FM_dav_resh_2 = FM_dav_GO.copy()
FM_dav_resh_2 = FM_dav_resh_2[:, ia_pars_idx][:, [17, 18, 19]]
mm.matshow(FM_dav_resh, log=True, abs_val=True, title='end')





FM_steph_CMBO_arr = FM_steph_CMBO['fish']
# delete tau
# if CMB_probe == 'CMB':
#     # ! not quite sure about this, but otherwise the FM dimensions don't match
#     FM_steph_CMBO_arr = np.delete(FM_steph_CMBO_arr, 7, axis=0)
#     FM_steph_CMBO_arr = np.delete(FM_steph_CMBO_arr, 7, axis=1)

# my Euclid-only + stephane's CMB-only, but only in the rows and columns of the cosmo params!!
FM_CMBX_GO = np.copy(FM_dav_GO)
FM_CMBX_GS = np.copy(FM_dav_GS)
for i in range(nparams):
    for j in range(nparams):
        FM_CMBX_GO[i, j] += FM_steph_CMBO_arr[i, j]
        FM_CMBX_GS[i, j] += FM_steph_CMBO_arr[i, j]

# fiducials
fid_steph_CMBX = FM_steph_CMBX['me'][:nparams]
fid_steph_euc = FM_steph_eucl['me'][:nparams]
assert np.array_equal(fid_steph_euc, fid_steph_CMBX), 'fiducial parameters do not match!'
fid_steph = fid_steph_CMBX

# marginal uncertainties
unc_dav_GO = mm.uncertainties_FM(FM_dav_GO, nparams, fid_steph)
unc_dav_GS = mm.uncertainties_FM(FM_dav_GS, nparams, fid_steph)
unc_CMBX_GO = mm.uncertainties_FM(FM_CMBX_GO, nparams, fid_steph)
unc_CMBX_GS = mm.uncertainties_FM(FM_CMBX_GS, nparams, fid_steph)
unc_sylv_GO = mm.uncertainties_FM(FM_sylv_GO, nparams, fid_steph)
unc_steph_eucl = FM_steph_eucl['prec'][:nparams] * 100
unc_steph_CMBX = FM_steph_CMBX['prec'][:nparams] * 100

# conditional uncertainties
cond_unc_dav_GO = mm.conditional_uncert_FM(FM_dav_GO, nparams, fid_steph)
cond_unc_steph_eucl = mm.conditional_uncert_FM(FM_steph_eucl['fish'], nparams, fid_steph)

# just a check that the uncertainties computed by me and stephane on his FM are indeed the same
assert np.allclose(mm.uncertainties_FM(FM_steph_CMBX['fish'], nparams, fid_steph), unc_steph_CMBX, rtol=1e-3), \
    'unc_dav != unc_steph - note that the fiducial value for s8 is slightly different'
assert np.allclose(mm.uncertainties_FM(FM_steph_eucl['fish'], nparams, fid_steph), unc_steph_eucl, rtol=1e-3), \
    'unc_dav != unc_steph - note that the fiducial value for s8 is slightly different'

diff = mm.percent_diff_mean(cond_unc_dav_GO, cond_unc_steph_eucl)
unc_list = [cond_unc_dav_GO, cond_unc_steph_eucl, diff]
label_list = ['cond_unc_dav GO', 'cond_unc_steph eucl', 'diff wrt mean']
unc_list = np.asarray(unc_list)

# set cmap to plasma
plt.figure()
plt.imshow(np.log10(np.abs(FM_steph_eucl['fish'] / FM_dav_resh - 1)), cmap='plasma')
plt.colorbar()
plt.title('log10(|FM_steph_eucl/FM_dav_GO - 1|)')

title = f'case-{case}_model-{cosmo_model}_CMB_probe-{CMB_probe}_response-{Rl_str[3:]}'
plot_utils.bar_plot_v2(unc_list, title=title, label_list=label_list, bar_width=0.18,
                       nparams=nparams, param_names_label=param_names_label)

plt.savefig(f'/Users/davide/Documents/Lavoro/Programmi/SSC_restructured_v2/jobs/CMBX/output/plots/{title}.png', dpi=300)
