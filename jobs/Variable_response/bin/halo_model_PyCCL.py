import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pyccl as ccl
import numpy as np

project_path = Path.cwd().parent.parent.parent
job_path = Path.cwd().parent

sys.path.append(f'{project_path}/lib')
import my_module as mm
import cosmo_lib as csmlib

sys.path.append(f'{project_path}/common_config')
import ISTF_fid_params as ISTfid
import mpl_cfg

matplotlib.use('Qt5Agg')

plt.rcParams.update(mpl_cfg.mpl_rcParams_dict)

start_time = time.perf_counter()

########################################################################################################################

Omega_c = ISTfid.primary['Om_m0'] - ISTfid.primary['Om_b0']
cosmo = ccl.Cosmology(Omega_c=Omega_c, Omega_b=ISTfid.primary['Om_b0'], w0=ISTfid.primary['w_0'],
                        wa=ISTfid.primary['w_a'], h=ISTfid.primary['h_0'], sigma8=ISTfid.primary['sigma_8'],
                        n_s=ISTfid.primary['n_s'], m_nu=ISTfid.extensions['m_nu'],
                        Omega_k=1 - (Omega_c + ISTfid.primary['Om_b0']) - ISTfid.extensions['Om_Lambda0'])

hm_recipe = 'KiDS1000'

z_arr = np.linspace(0.0, 4.0, num=300)
a_arr = csmlib.z_to_a(z_arr)



# mass definition
if hm_recipe == 'KiDS1000':  # arXiv:2007.01844
    c_m = 'Duffy08'  # ! NOT SURE ABOUT THIS
    mass_def = ccl.halos.MassDef200m(c_m='Duffy08')
    c_M_relation = ccl.halos.concentration.ConcentrationDuffy08(mdef=mass_def)
elif hm_recipe == 'Krause2017':  # arXiv:1601.05779
    c_m = 'Bhattacharya13'  # see paper, after Eq. 1
    mass_def = ccl.halos.MassDef200m(c_m=c_m)
    c_M_relation = ccl.halos.concentration.ConcentrationBhattacharya13(mdef=mass_def)  # above Eq. 12
else:
    raise ValueError('Wrong choice of hm_recipe: it must be either "KiDS1000" or "Krause2017".')

# TODO pass mass_def object? plus, understand what exactly is mass_def_strict

# mass function
massfunc = ccl.halos.hmfunc.MassFuncTinker10(cosmo, mass_def=mass_def, mass_def_strict=True)

# ! test, delete
mf_test = [massfunc.get_mass_function(cosmo, M=1e11, a=a, mdef_other=None) for a in a_arr]
plt.plot(a_arr, mf_test)
# ! finish test

# halo bias
hbias = ccl.halos.hbias.HaloBiasTinker10(cosmo, mass_def=mass_def, mass_def_strict=True)


# TODO understand better this object. We're calling the abstract class, is this ok?
# HMCalculator
hmc = ccl.halos.halo_model.HMCalculator(cosmo, massfunc, hbias, mass_def=mass_def,
                                        log10M_min=8.0, log10M_max=16.0, nlog10M=128,
                                        integration_method_M='simpson', k_min=1e-05)

# halo profile
halo_profile = ccl.halos.profiles.HaloProfileNFW(c_M_relation=c_M_relation,
                                                 fourier_analytic=True, projected_analytic=False,
                                                 cumul2d_analytic=False, truncated=True)

b = [halo_profile.projected(cosmo=cosmo, r_t=1, M=1e11, a=a, mass_def=mass_def) for a in a_arr]
plt.plot(a_arr, b)