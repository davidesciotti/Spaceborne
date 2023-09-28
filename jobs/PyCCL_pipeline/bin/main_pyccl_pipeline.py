import sys
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('/Users/davide/Documents/Lavoro/Programmi/common_lib_and_cfg/common_lib')
import my_module as mm
import cosmo_lib

sys.path.append(f'/Users/davide/Documents/Lavoro/Programmi/common_lib_and_cfg/common_config')
import ISTF_fid_params as ISTFfid
import mpl_cfg

sys.path.append('/Users/davide/Documents/Lavoro/Programmi/cl_v2/bin')
import wf_cl_lib

matplotlib.use('Qt5Agg')
plt.rcParams.update(mpl_cfg.mpl_rcParams_dict)

cfg = mm.read_yaml('../cfg/cfg_pyccl_pipeline.yml')
fiducial_pars_dict_nested = mm.read_yaml(
    '/Users/davide/Documents/Lavoro/Programmi/common_lib_and_cfg/common_config/ISTF_fiducial_params.yml')
fiducial_pars_dict = mm.flatten_dict(fiducial_pars_dict_nested)

zbins = cfg['zbins']
ell_min = cfg['ell_min']
ell_max = cfg['ell_max']
nbl = cfg['nbl']

cosmo_ccl = cosmo_lib.instantiate_cosmo_ccl_obj(fiducial_pars_dict)
n_of_z = np.genfromtxt('/Users/davide/Documents/Lavoro/Programmi/CLOE_validation/data/n_of_z/nzTabISTF.dat')
z_grid_nz = n_of_z[:, 0]
n_of_z = n_of_z[:, 1:]
dndz = (z_grid_nz, n_of_z)

# TODO ellmax 5000 for WL
ell_LL = np.logspace(np.log10(ell_min), np.log10(ell_max), nbl + 1)
ell_GL = np.logspace(np.log10(ell_min), np.log10(ell_max), nbl + 1)
ell_GG = np.logspace(np.log10(ell_min), np.log10(ell_max), nbl + 1)

list_params_to_vary = list(fiducial_pars_dict_nested['FM_ordered_params'].keys())
list_params_to_vary.remove('Om_Lambda0')

dz_param_list = [f'dz{zi:02d}_photo' for zi in range(1, zbins + 1)]
mag_param_list = [f'm{zi:02d}_photo' for zi in range(1, zbins + 1)]
elements_to_remove = ['Om_Lambda0'] + dz_param_list + mag_param_list
list_params_to_vary = [x for x in list_params_to_vary if x not in elements_to_remove]

cl_LL, cl_GL, cl_GG, dcl_LL, dcl_GL, dcl_GG = wf_cl_lib.cls_and_derivatives(
    fiducial_pars_dict, list_params_to_vary, zbins, dndz, ell_LL, ell_GG,
    'step-wise', pk=None, use_only_flat_models=True)

# ells = np.logspace(np.log10(ell_min), np.log10(ell_max), nbl + 1)
# cl_LL_3d = wf_cl_lib.cl_PyCCL(wl_kernel, wl_kernel, ells, zbins, p_of_k_a=None, cosmo=cosmo_ccl)
# cl_GL_3d = wf_cl_lib.cl_PyCCL(gc_kernel, wl_kernel, ells, zbins, p_of_k_a=None, cosmo=cosmo_ccl)
# cl_GG_3d = wf_cl_lib.cl_PyCCL(gc_kernel, gc_kernel, ells, zbins, p_of_k_a=None, cosmo=cosmo_ccl)

# for zi in range(zbins):
#     plt.plot(ells, cl_GG_3d[:, zi, zi], label=f'zbin {zi}')

assert False, 'stop here'

recipe = cl_cases_df[condition]['n(z)'].values[0]
bias_model = cl_cases_df[condition]['bias'].values[0]
ia_model = cl_cases_df[condition]['IA'].values[0]
zbins = cfg['zbins']

print('****************************************************')
print(f'computing cls for case_id Id {case_id}:\ncosmology {cosmology}, '
      f'recipe {recipe}, bias_model {bias_model}, ia_model {ia_model}')

# colormap
cmap = matplotlib.cm.get_cmap("rainbow")
colors = [cmap(i) for i in np.linspace(0, 1, zbins)]

zpairs_auto, zpairs_cross, zpairs_3x2pt = mm.get_zpairs(zbins)

if case_id == 'C08':
    warnings.warn('C08: z grid arrives only to 3, redefining z_grid_all; this is probably not ideal...')
n_of_z = np.genfromtxt(project_path / f'data/n_of_z/nzTab{recipe}.dat')
z_grid_nz = n_of_z[:, 0]
z_grid_all = z_grid_nz[1:]

if use_same_z_grid:
    # save z and k grid
    np.savetxt(f'{project_path}/output/intermediate_quantities/{case_id}/z_grid_CLOE.dat', z_grid_all,
               header='z (redshift)')
np.savetxt(f'{project_path}/output/intermediate_quantities/{case_id}/k_grid_pk_CLOE.dat',
           np.log10(k_grid_pk),
           header='k [1/Mpc]')

# interpolate the power spectrum to the same z grid used for the other quantities
pk_2d_func = interp1d(z_grid_pk, pk_2d, axis=0, kind='cubic')
pk_2d = pk_2d_func(z_grid_all)
z_grid_pk = z_grid_all

# add the parameters needed by PyCCL to the dataframe
cosmo_params_df['omnuh2'] = m_nu / neutrino_mass_fac * g_factor ** 0.75
cosmo_params_df['Omega_nu'] = cosmo_params_df['omnuh2'] / (cosmo_params_df['h'] ** 2)
cosmo_params_df['Omega_c'] = cosmo_params_df['Omega_M'] - (
        cosmo_params_df['Omega_b'] + cosmo_params_df['Omega_nu'])
cosmo_params_df['Omega_k'] = 1 - cosmo_params_df['Omega_M'] - cosmo_params_df['Omega_DE']
# where abs(omk) < 1e-8, set it to zero
condition = np.abs(cosmo_params_df['Omega_k'] < 1e-8)
cosmo_params_df.loc[condition, 'Omega_k'] = 0

# only the non-default parameters are shown
id_condition = (cosmo_params_df['Id'] == cosmology)
cosmo = ccl.core.Cosmology(Omega_c=cosmo_params_df.loc[id_condition]['Omega_c'].values[0],
                           Omega_b=cosmo_params_df.loc[id_condition]['Omega_b'].values[0],
                           h=cosmo_params_df.loc[id_condition]['h'].values[0],
                           n_s=cosmo_params_df.loc[id_condition]['n_s'].values[0],
                           sigma8=cosmo_params_df.loc[id_condition]['sigma_8'].values[0],
                           Omega_k=cosmo_params_df.loc[id_condition]['Omega_k'].values[0],
                           m_nu=m_nu,
                           w0=cosmo_params_df.loc[id_condition]['w_0'].values[0],
                           wa=cosmo_params_df.loc[id_condition]['w_a'].values[0],
                           extra_parameters={"camb": {"dark_energy_model": "ppf", }},
                           matter_power_spectrum='halo_model',
                           )

# this is needed because the scale_factor_grid_pk argument in the call to cosmo must me monothonically increased
# (so it's flipped, look at the [::-1]); as a consequence, the values of pk must be flipped along the z axis,
# so that the first rows corresponds to the highest redshift and the last rows to the lowest redshift
# scale_factor_grid_pk = 1 / (1 + z_grid_pk[::-1])  # alternative way, ALREADY FLIPS IT
pk_flipped_in_z = np.flip(pk_2d, axis=0)
scale_factor_grid_pk = cosmo_lib.z_to_a(z_grid_pk)[::-1]  # flip it
pk2d_pyccl = ccl.pk2d.Pk2D(pkfunc=None, a_arr=scale_factor_grid_pk, lk_arr=np.log(k_grid_pk),
                           pk_arr=pk_flipped_in_z, is_logp=False, cosmo=cosmo)
plot_pk_func()

# ! load n(z) and nuisance
n_of_z = np.genfromtxt(project_path / f'data/n_of_z/nzTab{recipe}.dat')
z_grid_nz = n_of_z[:, 0]
n_of_z = n_of_z[:, 1:]
# n_samples_wf = len(z_grid_nz)

nuisance_tab = np.genfromtxt(f'{project_path}/data/nuisance/nuiTab{recipe}.dat')
zbin_center_values = nuisance_tab[:, 0]
galaxy_bias_values = nuisance_tab[:, 2]
z_shifts = nuisance_tab[:, 4]

# IA
condition = intrinsic_alignment_df['Id'] == ia_model
a_ia = intrinsic_alignment_df.loc[condition]['A_IA'].values[0]
eta_ia = intrinsic_alignment_df.loc[condition]['eta_IA'].values[0]
beta_ia = intrinsic_alignment_df.loc[condition]['beta_IA'].values[0]

# ! shift n(z) for SPV3
# not-very-pythonic implementation: create an interpolator for each bin
_n_of_z_func = interp1d(z_grid_nz, n_of_z[:, zbin_idx], kind='linear')

z_grid_nz_shifted = z_grid_nz - z_shifts[zbin_idx]
z_grid_nz_shifted = np.clip(z_grid_nz_shifted, 0, 3)  # where < 0, set to 0; where > 3, set to 3
n_of_z[:, zbin_idx] = _n_of_z_func(z_grid_nz_shifted)

if use_same_z_grid:
    # re-interpolate the input z-dependent quantities on the same z grid (as opposed to their own z grid)

    # L(z)
    lumin_ratio_func = interp1d(z_grid_lumin_ratio, lumin_ratio, kind='linear')
lumin_ratio = lumin_ratio_func(z_grid_all)
z_grid_lumin_ratio = z_grid_all

# n(z)
n_of_z_func = interp1d(z_grid_nz, n_of_z, kind='linear', axis=0)
n_of_z = n_of_z_func(z_grid_all)
z_grid_nz = z_grid_all

# load growth factor and/or compute it on the same z grid as the luminosity ratio
if which_growth_factor == 'PyCCL':
    growth_factor = ccl.growth_factor(cosmo, a=1 / (1 + z_grid_lumin_ratio))  # validated against mine
elif which_growth_factor == 'CLOE':
    # load growth factor from CLOE
    growth_factor = np.genfromtxt(f'{project_path}/data/growth_factor/Dz_id{cosmology}_NLflag2_zpoints4000.txt')
z_grid_growth_factor = growth_factor[:, 0]
growth_factor = growth_factor[:, 1]

# interpolate on the same z grid as the luminosity ratio
growth_factor_func = interp1d(z_grid_growth_factor, growth_factor, kind='linear')
# removing the last element, max growth factor is at z=3.99 intead of 4
z_grid_lumin_ratio = z_grid_lumin_ratio[:-1]
lumin_ratio = lumin_ratio[:-1]  # removing the last element
growth_factor = growth_factor_func(z_grid_lumin_ratio)
raise ValueError('which_growth_factor must be either "PyCCL" or "CLOE"')

# ! IA bias
ia_bias = wf_cl_lib.build_IA_bias_1d_arr(z_grid_out=z_grid_lumin_ratio, input_z_grid_lumin_ratio=z_grid_lumin_ratio,
                                         input_lumin_ratio=lumin_ratio, cosmo=cosmo, A_IA=a_ia, eta_IA=eta_ia,
                                         beta_IA=beta_ia, C_IA=None, growth_factor=growth_factor,
                                         Omega_m=cosmo.cosmo.params.Omega_m)

# ! galaxy bias
# I just picked one of the possible grids; I could have created a new one as well. Anyway, if use_same_z_grid is
# True, then z_grid_galaxy_bias = z_grid_all = z_grid_lumin_ratio
z_grid_galaxy_bias = z_grid_lumin_ratio
galaxy_bias_2d_array = wf_cl_lib.build_galaxy_bias_2d_arr(galaxy_bias_values, zbin_center_values, zbins,
                                                          z_grid_galaxy_bias, bias_model, plot_bias)

# ! compute the tracer objects
wf_lensing = [ccl.tracers.WeakLensingTracer(cosmo, dndz=(z_grid_nz, n_of_z[:, zbin_idx]),
                                            ia_bias=(z_grid_lumin_ratio, ia_bias), use_A_ia=False,
                                            n_samples=n_samples_wf)
              for zbin_idx in range(zbins)]

wf_gamma = [ccl.tracers.WeakLensingTracer(cosmo, dndz=(z_grid_nz, n_of_z[:, zbin_idx]),
                                          ia_bias=None, use_A_ia=False,
                                          n_samples=n_samples_wf)
            for zbin_idx in range(zbins)]

wf_IA = [ccl.tracers.WeakLensingTracer(cosmo, dndz=(z_grid_nz, n_of_z[:, zbin_idx]),
                                       ia_bias=(z_grid_lumin_ratio, ia_bias), use_A_ia=False,
                                       n_samples=n_samples_wf, has_shear=False)
         for zbin_idx in range(zbins)]

wf_galaxy = [ccl.tracers.NumberCountsTracer(cosmo, has_rsd=False, dndz=(z_grid_nz, n_of_z[:, zbin_idx]),
                                            bias=(z_grid_galaxy_bias, galaxy_bias_2d_array[:, zbin_idx]),
                                            mag_bias=None,
                                            n_samples=n_samples_wf)
             for zbin_idx in range(zbins)]

# ! compute cls
cl_LL_3D = wf_cl_lib.cl_PyCCL(wf_lensing, wf_lensing, ell_grid, zbins, p_of_k_a=pk2d_pyccl, cosmo=cosmo)
cl_GL_3D = wf_cl_lib.cl_PyCCL(wf_galaxy, wf_lensing, ell_grid, zbins, p_of_k_a=pk2d_pyccl, cosmo=cosmo)
cl_GG_3D = wf_cl_lib.cl_PyCCL(wf_galaxy, wf_galaxy, ell_grid, zbins, p_of_k_a=pk2d_pyccl, cosmo=cosmo)
cl_gammaIA_3D = wf_cl_lib.cl_PyCCL(wf_gamma, wf_IA, ell_grid, zbins, p_of_k_a=pk2d_pyccl, cosmo=cosmo)
cl_IAIA_3D = wf_cl_lib.cl_PyCCL(wf_IA, wf_IA, ell_grid, zbins, p_of_k_a=pk2d_pyccl, cosmo=cosmo)
cl_gammagamma_3D = wf_cl_lib.cl_PyCCL(wf_gamma, wf_gamma, ell_grid, zbins, p_of_k_a=pk2d_pyccl, cosmo=cosmo)
