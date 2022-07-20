import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('.../lib')
import my_module as mm

sys.path.append('.../bin')
import plots_FM_running as plot_utils

case = 'pess'
CMB_probe = 'CMB'  # CMB or _CMBphionly
CMB_obs = 'planck'  # "-planck" or "-SO" or "-S4"
mode = 'p'  # * p ("plus")  == CMB + Euclid as independent probes
# * a ("add")   == proper CMBxEuclid covariance but no cross-probes in data
# * c ("cross") == full CMBxEucld covariance and data
cosmo_model = 'w0waCDM'
probe = '3x2pt'
GO_or_GS = 'GO'
steph_with_CMB = False

if case == 'opti':
    lmax = 3000
elif case == 'pess':
    lmax = 750
else:
    raise ValueError('case not recognized')

if GO_or_GS == 'GO':
    Rl_str = ''
else:
    Rl_str = '_Rlconst'

if steph_with_CMB:
    fish = np.load(
        f'/Users/davide/Documents/Lavoro/Programmi/SSC_restructured_v2/jobs/CMBX/input/FM_stephane/'
        f'fish_Euclid-{case}_{CMB_probe}-planck_mode-{mode}_flat_{cosmo_model}_max-bins_super-prec_21point.npz')
else:
    fish = np.load(
        f'/Users/davide/Documents/Lavoro/Programmi/SSC_restructured_v2/jobs/CMBX/input/FM_stephane/'
        f'fish_Euclid-{case}_flat_{cosmo_model}_max-bins_super-prec_21point.npz')


fish_dav_dict = dict(mm.get_kv_pairs('/Users/davide/Documents/Lavoro/Programmi/SSC_restructured_v2/jobs/IST_forecast/output/FM', 'txt'))


print('shape:', fish['fish'].shape, '\nparams:', fish['all_params'], '\nfid:', fish['me'])

unc_dav = mm.uncertainties_FM(fish_dav_dict[f'FM_{probe}_{GO_or_GS}_lmaxXC{lmax}_nbl30{Rl_str}'])[:7]
unc_steph = fish['prec'][:7]* 100

# just a check that the uncertainties computed by me and stephane on his FM are indeed the same
assert np.allclose(mm.uncertainties_FM(fish['fish'])[:7], unc_steph[:7], rtol=1e-3), \
    'unc_dav != unc_steph - note that the fiducial value for s8 is slightly different'

diff = mm.percent_diff_mean(unc_dav, unc_steph)

unc_list = [unc_dav, unc_steph, diff]
unc_list = np.asarray(unc_list)
label_list = ['davide GO', 'stéphane', '%diff']

plot_utils.bar_plot_v2(unc_list, title=f'case: {case}, model: {cosmo_model}', label_list=label_list, bar_width=0.25)
