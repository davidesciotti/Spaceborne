import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('.../lib')
import my_module as mm

sys.path.append('.../bin')
import plots_FM_running as plot_utils

# mode:
# * p ("plus")  == CMB + Euclid as independent probes
# * a ("add")   == proper CMBxEuclid covariance but no cross-probes in data
# * c ("cross") == full CMBxEucld covariance and data

for case in ['pess', 'opti']:
    for Rl_str in ['_Rlconst', '_Rlvar']:
        for cosmo_model in ['LCDM', 'w0waCDM']:

            case = 'pess'
            CMB_probe = 'CMB'  # CMB or CMBphionly
            CMB_obs = 'planck'  # "planck" or "SO" or "S4"
            mode = 'p'
            cosmo_model = 'w0waCDM'  # w0waCDM or LCDM
            probe = '3x2pt'
            GO_or_GS = 'GO'
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

            print('shape:', FM_steph_CMBO['fish'].shape, '\nparams:', FM_steph_CMBO['all_params'], '\nfid:', FM_steph_CMBO['me'])


            if cosmo_model == 'LCDM':
                nparams = 5
                FM_dav_GO = np.delete(FM_dav_GO, (2, 3), axis=0)
                FM_dav_GO = np.delete(FM_dav_GO, (2, 3), axis=1)
                FM_dav_GS = np.delete(FM_dav_GS, (2, 3), axis=0)
                FM_dav_GS = np.delete(FM_dav_GS, (2, 3), axis=1)
                param_names_label = ["$\Omega_{{\\rm m},0}$", "$\Omega_{{\\rm b},0}$", "$h$", "$n_{\\rm s}$",
                                      "$\sigma_8$"]
            elif cosmo_model == 'w0waCDM':
                nparams = 7
                param_names_label = ["$\Omega_{{\\rm m},0}$", "$\Omega_{{\\rm b},0}$", "$w_0$", "$w_a$", "$h$", "$n_{\\rm s}$",
                                      "$\sigma_8$"]
            else:
                raise ValueError('cosmo_model not recognized')

            FM_steph_CMBO_arr = FM_steph_CMBO['fish']
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

            # compute uncertainties
            unc_dav_GO = mm.uncertainties_FM(FM_dav_GO, nparams, fid_steph)
            unc_dav_GS = mm.uncertainties_FM(FM_dav_GS, nparams, fid_steph)
            unc_CMBX_GO = mm.uncertainties_FM(FM_CMBX_GO, nparams, fid_steph)
            unc_CMBX_GS = mm.uncertainties_FM(FM_CMBX_GS, nparams, fid_steph)
            unc_steph_eucl = FM_steph_eucl['prec'][:nparams] * 100
            unc_steph_CMBX = FM_steph_CMBX['prec'][:nparams] * 100

            # just a check that the uncertainties computed by me and stephane on his FM are indeed the same
            assert np.allclose(mm.uncertainties_FM(FM_steph_CMBX['fish'], nparams, fid_steph), unc_steph_CMBX, rtol=1e-3), \
                'unc_dav != unc_steph - note that the fiducial value for s8 is slightly different'
            assert np.allclose(mm.uncertainties_FM(FM_steph_eucl['fish'], nparams, fid_steph), unc_steph_eucl, rtol=1e-3), \
                'unc_dav != unc_steph - note that the fiducial value for s8 is slightly different'


            # diff = mm.percent_diff_mean(unc_steph_eucl, unc_steph_CMBX)
            # unc_list = [unc_steph_eucl, unc_steph_CMBX, diff]
            unc_list = [unc_dav_GO, unc_dav_GS, unc_CMBX_GO, unc_CMBX_GS]
            unc_list = np.asarray(unc_list)
            label_list = ['dav GO', 'dav GS', 'dav GO + steph CMB', 'dav GS + steph CMB']

            print(unc_CMBX_GO)

            title = f'case: {case}, model: {cosmo_model}, CMB_probe: {CMB_probe}, response: {Rl_str[3:]}'
            plot_utils.bar_plot_v2(unc_list, title=title, label_list=label_list, bar_width=0.18,
                                   nparams=nparams, param_names_label=param_names_label)

            plt.savefig(f'/Users/davide/Documents/Lavoro/Programmi/SSC_restructured_v2/jobs/CMBX/output/plots/{title}.png', dpi=300)


