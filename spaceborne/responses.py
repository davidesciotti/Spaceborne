import sys
import time
import warnings
import matplotlib.pyplot as plt
from scipy import integrate
import yaml
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import pyccl as ccl
import matplotlib as mpl
from matplotlib.lines import Line2D


ROOT = '/home/davide/Documenti/Lavoro/Programmi'
sys.path.append(f'{ROOT}/Spaceborne')
import spaceborne.my_module as mm
import spaceborne.cosmo_lib as csmlib
import spaceborne.ell_utils as ell_utils
import spaceborne.sigma2_SSC as sigma2_SSC
import spaceborne.wf_cl_lib as wf_cl_lib
import spaceborne.cosmo_lib as cosmo_lib

import common_cfg.mpl_cfg as mpl_cfg


plt.rcParams.update(mpl_cfg.mpl_rcParams_dict)
start_time = time.perf_counter()


class SpaceborneResponses():

    def __init__(self, cfg, k_grid, z_grid, cosmo_ccl, b1_func):
        
        ccl.spline_params['A_SPLINE_NA_PK'] = 240  # gives CAMB error if too high
        ccl.spline_params['K_MAX_SPLINE'] = 300 


        # grids over which to compute the responses
        self.k_grid = k_grid
        self.k_min = k_grid.min()
        self.k_max = k_grid.max()
        self.k_steps = len(k_grid)

        self.z_grid = z_grid
        self.z_min = z_grid.min()
        self.z_max = z_grid.max()
        self.z_steps = len(z_grid)

        self.zbins = cfg['general_cfg']['zbins']

        self.cosmo_ccl = cosmo_ccl
        self.use_h_units = cfg['general_cfg']['use_h_units']
        self.h = cfg['cosmology']['FM_ordered_params']['h']
        assert self.use_h_units is False, 'case True should be fine but for now stick to False'
        self.b1_func = b1_func

        # ! get growth only values - DIMENSIONLESS
        g1_table = np.genfromtxt(f'{ROOT}/Spaceborne/input/Resp_G1_fromsims.dat')

        # take k and z values (the latter from the header), k is in [h/Mpc]
        self.k_grid_g1 = g1_table[:, 0]
        self.g1_table = g1_table[:, 1:]

        # convert k_G1 to [1/Mpc] if needed
        if not self.use_h_units:
            self.k_grid_g1 *= self.h
        assert np.all(np.diff(self.k_grid_g1) > 0), 'k_grid_g1 is not sorted!'

        # self.k_fund = 0.012 # [h/Mpc], this value is from the paper
        self.k_fund_g1 = self.k_grid_g1.min()  # to avoid interpolation issues (it's 0.0125664 insted of 0.012, no big deal)
        self.k_max_g1 = self.k_grid_g1.max()

        self.z_grid_g1 = np.array((0.00, 0.50, 1.00, 2.00, 3.00))
        self.b1 = -0.75
        self.g1_linear_value = 26 / 21

        # which_linear_bias = cfg['which_linear_bias']
        # which_wf_gc = cfg['which_wf_gc']

    def g1_extrap_func_original(self, k, z, g1_interp):
        # extrapolate according to Eq. (2.7) in Alex's paper
        result = self.b1 + (g1_interp((self.k_max_g1, z)).reshape(z.size, 1) -
                            self.b1) * (k / self.k_max_g1) ** (- 1 / 2)
        # result = self.b1 + (g1_interp((self.k_max_g1, z)) - self.b1) * (k / self.k_max_g1) ** (- 1 / 2)
        return result

    def g1_extrap_func(self, k_array, z_array, g1_interp):
        result = np.zeros((k_array.size, z_array.size))
        for zi, z in enumerate(z_array):
            result[:, zi] = self.b1 + (g1_interp((self.k_max_g1, z)) - self.b1) * (k_array / self.k_max_g1) ** (- 1 / 2)
        # result = self.b1 + (g1_interp((self.k_max_g1, z)) - self.b1) * (k / self.k_max_g1) ** (- 1 / 2)
        return result

    def g1_tot_func(self, k_array, z, g1_interp, g1_extrap):
        """
        G1 is equal to:
        * 26/21 for k < k_fund
        * G1 from Alex's table for k_fund < k < k_max
        * G1 from Eq. (2.7) for k > k_max
        """

        # find indices for the various thresholds
        k_low_indices = np.where(k_array <= self.k_fund_g1)[0]
        k_mid_indices = np.where((self.k_fund_g1 < k_array) & (k_array <= self.k_max_g1))[0]
        k_high_indices = np.where(k_array > self.k_max_g1)[0]

        # fill the 3 arrays
        low = np.zeros((k_low_indices.size, z.size))
        low.fill(self.g1_linear_value)

        kk, zz = np.meshgrid(k_array[k_mid_indices], z)
        mid = g1_interp((kk, zz)).T

        high = g1_extrap(k_array=k_array[k_high_indices], z_array=z, g1_interp=g1_interp)

        # concatenate the 3 arrays over the rows, i.e. the k values
        return np.concatenate((low, mid, high), axis=0)

    def b2_func(self, z):
        """ second-order galaxy bias from fit in Lazeyras et al. 2016 - note that this formula is actually
        for b2_halo(b1_halo), I need to test this better"""
        return 0.412 - 2.143 * self.b1_func(z) + 0.929 * (self.b1_func(z) ** 2) + 0.008 * (self.b1_func(z) ** 3)

    def compute_r1_mm(self):

        # interpolate G1; attention: the function is g1_interp(z, k), while the array is G1[k, z]
        g1_interp = RegularGridInterpolator((self.k_grid_g1, self.z_grid_g1), self.g1_table, method='linear')

        # ! nonlinear pk and its derivative
        # TODO extract direcly from cosmo object
        self.k_grid, self.pk_mm = csmlib.pk_from_ccl(k_array=self.k_grid, z_array=self.z_grid,
                                                     use_h_units=self.use_h_units, cosmo_ccl=self.cosmo_ccl,
                                                     pk_kind='nonlinear')
        
        dpkmm_dk = np.gradient(self.pk_mm, self.k_grid, axis=0)
        # I broadcast k_grid as k_grid[:, None] here and below to have the correct shape (k_points, 1)
        dlogpkmm_dlogk = self.k_grid[:, None] / self.pk_mm * dpkmm_dk

        # ! response coefficient
        self.r1_mm = 1 - 1 / 3 * dlogpkmm_dlogk + \
            self.g1_tot_func(k_array=self.k_grid, z=self.z_grid, g1_interp=g1_interp, g1_extrap=self.g1_extrap_func)

        # if self.plot_r1mm_func:
        # self.plot_r1mm_func()

        return self.r1_mm

    def get_b2_with_ccl(self, z, cosmo_ccl):
        # TODO implement this!
        
        # ! new section: compute second-order galaxy bias (I basically didn't even begin this test)
        mass_def = ccl.halos.MassDef200m
        c_M_relation = ccl.halos.ConcentrationDuffy08(mass_def=mass_def)
        hmf = ccl.halos.MassFuncTinker10(mass_def=mass_def)
        hbf = ccl.halos.HaloBiasTinker10(mass_def=mass_def)
        hmc = ccl.halos.HMCalculator(mass_function=hmf, halo_bias=hbf, mass_def=mass_def)
        halo_profile_nfw = ccl.halos.HaloProfileNFW(mass_def=mass_def, concentration=c_M_relation)
        halo_profile_hod = ccl.halos.HaloProfileHOD(mass_def=mass_def, concentration=c_M_relation)
        
        
        a = cosmo_lib.z_to_a(z)
        
        # Define the integrand
        def integrand(M, a):
            # Compute Φ_MF(M, a)
            phi_mf = hmc.mass_function(cosmo_ccl, M, a)
            
            # Compute b_1^h(M, a)
            b1h = hmc.halo_bias(cosmo_ccl, M, a)
            
            # Compute N(M)
            n_given_m = halo_profile_hod.get_number(M)
            
            return phi_mf * b1h * n_given_m

        # Define n_gal(z)
        def n_gal(a):
            return integrate.quad(lambda M: integrand(M, a), 1e10, 1e16)[0]

        # Compute b₂(z)
        numerator = integrate.quad(lambda M: integrand(M, a) * self.hbf(M, a), 1e10, 1e16)[0]
        denominator = n_gal(a)
            
        
        return numerator / denominator


    def get_rab_and_dpab_ddeltab(self):
        # galaxy bias (I broadcast it to be able to multiply/sum it with r1_mm and pk_mm)
        # I loop to check the impact (and the correctness) of b2
        self.b1_arr = self.b1_func(self.z_grid)[None, :]
        self.b2_arr = self.b2_func(self.z_grid)[None, :]
        self.b2_arr_null = np.zeros(self.b2_arr.shape)

        # ! compute dPk/ddelta_b (not the projected ones!)
        term1 = 1 / self.b1_arr
        term2 = self.b2_arr - self.b1_arr ** 2
        term2_nob2 = self.b2_arr_null - self.b1_arr ** 2

        self.dPmm_ddeltab = self.r1_mm * self.pk_mm
        self.dPgm_ddeltab = (self.r1_mm + term1 * term2) * self.pk_mm
        self.dPgg_ddeltab = (self.r1_mm + 2 * term1 * term2) * self.pk_mm

        self.dPgm_ddeltab_nob2 = (self.r1_mm + term1 * term2_nob2) * self.pk_mm
        self.dPgg_ddeltab_nob2 = (self.r1_mm + 2 * term1 * term2_nob2) * self.pk_mm

        # ! compute r1_AB (again, not the projected ones)
        self.pk_gg = self.pk_mm * self.b1_arr ** 2
        self.pk_gm = self.pk_mm * self.b1_arr
        self.r1_gm = self.dPgm_ddeltab / self.pk_gm
        self.r1_gg = self.dPgg_ddeltab / self.pk_gg

        self.r1_gm_nob2 = self.dPgm_ddeltab_nob2 / self.pk_gm
        self.r1_gg_nob2 = self.dPgg_ddeltab_nob2 / self.pk_gg

    def plot_r1mm_func(self):

        # increase font and legend size
        plt.rcParams.update({'font.size': 24})
        plt.rcParams.update({'legend.fontsize': 28})

        # reproduce Alex's plot
        z_max_plot = 1.8  # from the figure in the paper
        z_max_idx = np.argmin(np.abs(z_grid - z_max_plot))
        z_reduced = z_grid[:z_max_idx + 1]

        # from https://stackoverflow.com/questions/26545897/drawing-a-colorbar-aside-a-line-plot-using-matplotlib
        # norm is a class that, when called, can normalize data into the [0.0, 1.0] interval.
        norm = mpl.colors.Normalize(
            vmin=np.min(z_reduced),
            vmax=np.max(z_reduced))

        # choose a colormap and a line width
        cmap = mpl.cm.jet
        lw = 1

        # create a ScalarMappable and initialize a data structure
        s_m = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        s_m.set_array([])

        plt.figure()
        # the colors are chosen by calling the ScalarMappable that was initialized with c_m and norm
        for z_idx, z_val in enumerate(z_reduced):
            plt.plot(k_grid, self.r1_mm[:, z_idx], color=cmap(norm(z_val)), lw=lw)

        cbar = plt.colorbar(s_m)
        cbar.set_label('$z$', rotation=0, labelpad=20)

        plt.xscale('log')
        plt.xlabel(k_TeX_label)
        plt.ylabel('$R_1^{\\rm mm}(k, z)$')
        plt.axvline(x=self.k_max_g1, ls='--', lw=3, label='$k_{\\rm max}^{\\rm G1}$')
        plt.axvline(x=self.k_fund_g1, ls='--', lw=3, label='$k_{\\rm fund}$')
        plt.xlim(1e-4, 1e1)
        plt.ylim(0.5, 4)
        plt.grid()
        plt.show()
        plt.legend()

        plt.savefig('../output/plots/r1mm_rainbow.pdf', bbox_inches='tight', dpi=500)

    def plot_responses(self):
        if not plot_responses:
            return
        z_idx = 0

        plt.figure()
        plt.plot(k_grid, dPmm_ddeltab[:, z_idx], label=f'dPmm_ddeltab, z={z_grid[z_idx]}', alpha=0.8)
        plt.plot(k_grid, dPgm_ddeltab[:, z_idx], label=f'dPgm_ddeltab, z={z_grid[z_idx]}', alpha=0.8)
        plt.plot(k_grid, dPgg_ddeltab[:, z_idx], label=f'dPgg_ddeltab, z={z_grid[z_idx]}', alpha=0.8)
        plt.legend()
        plt.xscale('log')
        plt.xlabel(f'k {k_TeX_label}')
        plt.ylabel(f'dPAB/ddelta_b')

        # Response coefficients, or dlogPAB/ddelta_b
        plt.figure()
        plt.plot(k_grid, r1_mm[:, z_idx], label='r1_mm', alpha=0.8)
        plt.plot(k_grid, r1_gm[:, z_idx], label='r1_gm', alpha=0.8)
        plt.plot(k_grid, r1_gg[:, z_idx], label='r1_gg', alpha=0.8)
        plt.plot(k_grid, r1_gm_nob2[:, z_idx], label='r1_gm_nob2', alpha=0.8, ls='--')
        plt.plot(k_grid, r1_gg_nob2[:, z_idx], label='r1_gg_nob2', alpha=0.8, ls='--')
        plt.xscale('log')
        plt.legend()
        plt.xlabel(f'k {k_TeX_label}')
        plt.ylabel(f'r1_AB')

        plt.figure()
        plt.title(which_linear_bias)
        plt.plot(z_grid, b1_arr.flatten(), label='b1')
        plt.plot(z_grid, b2_arr.flatten(), label='b2')
        plt.legend()
        plt.xlabel('z')
        plt.ylabel('bias')

        # matshow r1_mm
        plt.figure()
        plt.matshow(r1_mm)
        plt.colorbar()
        plt.title(r'$R_1^{mm}$')
        plt.xlabel('$z$')
        plt.ylabel(k_TeX_label)

        plt.figure()
        plt.matshow(np.log10(dPmm_ddeltab))
        plt.colorbar()
        plt.title(r'$dP^{mm}/d\delta_b$')
        plt.xlabel('$z$')
        plt.ylabel(k_TeX_label)

        # set k and z values as tick labels
        step = 100
        y_positions = np.arange(0, len(k_grid), step)
        x_positions = np.arange(0, len(z_grid), step)
        y_labels = k_grid[::step]
        x_labels = z_grid[::step]
        plt.xticks(x_positions, x_labels)
        plt.yticks(y_positions, y_labels)

        # Set the desired number of decimal places for tick labels
        decimal_places = 2
        plt.gca().set_xticklabels([f'{val:.{decimal_places}f}' for val in x_labels])
        plt.gca().set_yticklabels([f'{val:.{decimal_places}f}' for val in y_labels])

    # assert which_wf_gc == 'without_galaxy_bias', 'the galaxy bias included in the Pk, not in the kernels!!'


"""
# ! ============================================== project responses ===================================================
# compute ell values
ell_min = cfg['ell_min']
ell_max = cfg['ell_max']
nbl = cfg['nbl']
ell_grid, _ = ell_utils.compute_ells(nbl, ell_min, ell_max, cfg['ell_grid_recipe'])

# find k_limber max do decide the minimum z for which to compute cls in the projeciton
kmax_limber = csmlib.get_kmax_limber(ell_grid, z_grid, use_h_units, cosmo_ccl)
assert kmax_limber < k_max, f'kmax_limber > k_max ({kmax_limber} > {k_max}): increase k_max or increase z_min. k is in' \
    f'{k_TeX_label}'

# at low redshift and high ell_grid, k_limber explodes: cut the z range
# z_grid = z_grid[5:]  # ! this has been found by hand, fix this!

# option 1, the easy way: interpolate and fill
r1mm_interp_func = RegularGridInterpolator((k_grid, z_grid), r1_mm, method='linear')
# 2. TODO option 2, the hard(er) way: construct r1_mm function and evaluate it in kl, z

# ! ccl kernels
n_of_z = np.genfromtxt(cfg['dndz_filename'].format(ROOT=ROOT))
z_grid_nz = n_of_z[:, 0]
n_of_z = n_of_z[:, 1:]
dndz_tuple = (z_grid_nz, n_of_z)

z_means = np.array([flat_fid_pars_dict[f'zmean{zi:02d}_photo'] for zi in range(1, zbins + 1)])
z_edges = np.array([flat_fid_pars_dict[f'zedge{zi:02d}_photo'] for zi in range(1, zbins + 2)])
gal_bias_vs_zmeans = np.asarray([b1_func(z_mean) for z_mean in z_means])
gal_bias_2d = wf_cl_lib.build_galaxy_bias_2d_arr(
    gal_bias_vs_zmean=gal_bias_vs_zmeans, zmeans=z_means, z_edges=z_edges, zbins=zbins, z_grid=z_grid, bias_model=bias_model, plot_bias=False)
gal_bias_tuple = (z_grid, gal_bias_2d)

kernel_wl = wf_cl_lib.wf_ccl(z_grid, probe='lensing', which_wf='with_IA', flat_fid_pars_dict=flat_fid_pars_dict,
                             cosmo_ccl=cosmo_ccl, dndz_tuple=dndz_tuple, ia_bias_tuple=None,
                             gal_bias_tuple=None, mag_bias_tuple=None,
                             return_ccl_obj=False, n_samples=1000)
kernel_wl_ccl_obj = wf_cl_lib.wf_ccl(z_grid, probe='lensing', which_wf='with_IA', flat_fid_pars_dict=flat_fid_pars_dict,
                                     cosmo_ccl=cosmo_ccl, dndz_tuple=dndz_tuple, ia_bias_tuple=None,
                                     gal_bias_tuple=None, mag_bias_tuple=None,
                                     return_ccl_obj=True, n_samples=1000)
kernel_gc = wf_cl_lib.wf_ccl(z_grid, probe='galaxy', which_wf=which_wf_gc, flat_fid_pars_dict=flat_fid_pars_dict,
                             cosmo_ccl=cosmo_ccl, dndz_tuple=dndz_tuple, ia_bias_tuple=None,
                             gal_bias_tuple=gal_bias_tuple, mag_bias_tuple=None,
                             return_ccl_obj=False, n_samples=1000)
kernel_gc_ccl_obj = wf_cl_lib.wf_ccl(z_grid, probe='galaxy', which_wf=which_wf_gc, flat_fid_pars_dict=flat_fid_pars_dict,
                                     cosmo_ccl=cosmo_ccl, dndz_tuple=dndz_tuple, ia_bias_tuple=None,
                                     gal_bias_tuple=gal_bias_tuple, mag_bias_tuple=None,
                                     return_ccl_obj=True, n_samples=1000)


# convert kernels to h/Mpc if h_units is True, else leave it as it is (1/Mpc)
if use_h_units:
    kernel_wl /= h
    kernel_gc /= h

# ! normalize them if integral convention is PySSC
# TODO this factor could probably be moved to the volume element, to be able to skip this passage...
r_of_z = csmlib.ccl_comoving_distance(z=z_grid, use_h_units=use_h_units, cosmo_ccl=cosmo_ccl)
if cl_integral_convention == 'PySSC':
    kernel_wl /= np.repeat(r_of_z[:, None], zbins, axis=1) ** 2
    kernel_gc /= np.repeat(r_of_z[:, None], zbins, axis=1) ** 2

# ! compute the terms d2CAB_dVddeltab
dPmm_ddeltab_interp_func = RegularGridInterpolator((k_grid, z_grid), dPmm_ddeltab, method='linear')
dPgm_ddeltab_interp_func = RegularGridInterpolator((k_grid, z_grid), dPgm_ddeltab, method='linear')
dPgg_ddeltab_interp_func = RegularGridInterpolator((k_grid, z_grid), dPgg_ddeltab, method='linear')
pk_mm_interp_func = RegularGridInterpolator((k_grid, z_grid), pk_mm, method='linear')
pk_gm_interp_func = RegularGridInterpolator((k_grid, z_grid), pk_gm, method='linear')
pk_gg_interp_func = RegularGridInterpolator((k_grid, z_grid), pk_gg, method='linear')

# note that I don't need to project the responses if I use
# covSSC = \iint dV1 dV2 dClAB/d\delta_bdV dClCD/d\delta_bdV sigma2(z1, z2)
# this is because the projection is done by the integral in dV itself
dPmm_ddeltab_klimb = np.array(
    [dPmm_ddeltab_interp_func((csmlib.k_limber(ell_val, z_grid, use_h_units, cosmo_ccl), z_grid)) for ell_val in ell_grid])
dPgm_ddeltab_klimb = np.array(
    [dPgm_ddeltab_interp_func((csmlib.k_limber(ell_val, z_grid, use_h_units, cosmo_ccl), z_grid)) for ell_val in ell_grid])
dPgg_ddeltab_klimb = np.array(
    [dPgg_ddeltab_interp_func((csmlib.k_limber(ell_val, z_grid, use_h_units, cosmo_ccl), z_grid)) for ell_val in ell_grid])

# L = ell index; i, j = z indices; o = z step index
d2CLL_dVddeltab = np.einsum('oi, oj, Lo -> Lijo', kernel_wl, kernel_wl, dPmm_ddeltab_klimb)
d2CGL_dVddeltab = np.einsum('oi, oj, Lo -> Lijo', kernel_gc, kernel_wl, dPgm_ddeltab_klimb)
d2CGG_dVddeltab = np.einsum('oi, oj, Lo -> Lijo', kernel_gc, kernel_gc, dPgg_ddeltab_klimb)

# ! perform the actual projection
other_project_pk_args = (z_grid, ell_grid, cl_integral_convention, use_h_units, cosmo_ccl)
cl_ll_3d = csmlib.project_pk(pk_mm_interp_func, kernel_wl, kernel_wl, *other_project_pk_args)
cl_gl_3d = csmlib.project_pk(pk_gm_interp_func, kernel_gc, kernel_wl, *other_project_pk_args)
cl_gg_3d = csmlib.project_pk(pk_gg_interp_func, kernel_gc, kernel_gc, *other_project_pk_args)
dcl_ll_ddeltab_3d = csmlib.project_pk(dPmm_ddeltab_interp_func, kernel_wl, kernel_wl, *other_project_pk_args)
dcl_gl_ddeltab_3d = csmlib.project_pk(dPgm_ddeltab_interp_func, kernel_gc, kernel_wl, *other_project_pk_args)
dcl_gg_ddeltab_3d = csmlib.project_pk(dPgg_ddeltab_interp_func, kernel_gc, kernel_gc, *other_project_pk_args)

# finally, divide by Cl to get the (projected) response coefficient
rl_ll_3d = dcl_ll_ddeltab_3d / cl_ll_3d
rl_gl_3d = dcl_gl_ddeltab_3d / cl_gl_3d
rl_gg_3d = dcl_gg_ddeltab_3d / cl_gg_3d


# ! save everything
output_path = cfg['output_path'].format(ROOT=ROOT)
benchmark_path = '../output/responses/benchmarks'
np.savetxt(f'{output_path}/other_stuff/k_grid_responses_{k_txt_label}.txt', k_grid)

# these are the actual ingredients used in the latest version of the integral
general_suffix = f'nbl{nbl}_ellmax{ell_max}_zbins{ep_or_ed}{zbins}_zsteps{
    z_steps}_k{k_txt_label}_convention{cl_integral_convention}'
cl_integral_prefactor_arr = csmlib.cl_integral_prefactor(z_grid, cl_integral_convention, use_h_units=use_h_units,
                                                         cosmo_ccl=cosmo_ccl)
np.save(f'{output_path}/d2ClAB_dVddeltab/z_grid_responses_zsteps{z_steps}.npy', z_grid)
np.save(f'{output_path}/d2ClAB_dVddeltab/d2CLL_dVddeltab_{general_suffix}.npy', d2CLL_dVddeltab)
np.save(f'{output_path}/d2ClAB_dVddeltab/d2CGL_dVddeltab_{general_suffix}.npy', d2CGL_dVddeltab)
np.save(f'{output_path}/d2ClAB_dVddeltab/d2CGG_dVddeltab_{general_suffix}.npy', d2CGG_dVddeltab)
np.save(f'{output_path}/d2ClAB_dVddeltab/cl_integral_prefactor_{general_suffix}.npy', cl_integral_prefactor_arr)

# pk responses dPAB_ddeltab = R1_AB * Pk_AB
np.save(f'{output_path}/other_stuff/dPmm_ddeltab_{pk_txt_label}.npy', dPmm_ddeltab)
np.save(f'{output_path}/other_stuff/dPgm_ddeltab_{pk_txt_label}.npy', dPgm_ddeltab)
np.save(f'{output_path}/other_stuff/dPgg_ddeltab_{pk_txt_label}.npy', dPgg_ddeltab)
# np.save(f'{output_path}/other_stuff/dPgm_ddeltab_nob2_{pk_txt_label}.npy', dPgm_ddeltab_nob2)
# np.save(f'{output_path}/other_stuff/dPgg_ddeltab_nob2_{pk_txt_label}.npy', dPgg_ddeltab_nob2)

# pk response coefficients R1_AB(k) = dlogPAB/ddelta_b = dPAB/ddelta_b / PAB
np.save(f'{output_path}/other_stuff/r1_mm_k{k_txt_label}.npy', r1_mm)
np.save(f'{output_path}/other_stuff/r1_gm_k{k_txt_label}.npy', r1_gm)
np.save(f'{output_path}/other_stuff/r1_gg_k{k_txt_label}.npy', r1_gg)

# projected responses R1_AB(ell) = dlogClAB/delta_b = dClAB/delta_b / ClAB
np.save(f'{output_path}/other_stuff/projected_responses/rl_ll_3d_k{k_txt_label}.npy', rl_ll_3d)
np.save(f'{output_path}/other_stuff/projected_responses/rl_gl_3d_k{k_txt_label}.npy', rl_gl_3d)
np.save(f'{output_path}/other_stuff/projected_responses/rl_gg_3d_k{k_txt_label}.npy', rl_gg_3d)
# associated cls, for completeness
np.save(f'{output_path}/other_stuff/projected_responses/cl_ll_3d_k{k_txt_label}.npy', cl_ll_3d)
np.save(f'{output_path}/other_stuff/projected_responses/cl_gl_3d_k{k_txt_label}.npy', cl_gl_3d)
np.save(f'{output_path}/other_stuff/projected_responses/cl_gg_3d_k{k_txt_label}.npy', cl_gg_3d)
np.savetxt(f'{output_path}/other_stuff/projected_responses/ell_grid_nbl{nbl}.txt', ell_grid)

if cfg['test_against_benchmarks']:
    mm.test_folder_content(output_path, output_path + '/benchmarks', 'npy')

# ! test: compare against Vincenzo's responses and cls
probe_to_test = 'll'
assert probe_to_test in ['ll', 'gg'], 'update the reshaping below for gl'

if probe_to_test == 'll':
    rl_dav_3d = rl_ll_3d
    cl_dav_3d = cl_ll_3d
elif probe_to_test == 'gl':
    rl_dav_3d = rl_gl_3d
    cl_dav_3d = cl_gl_3d
elif probe_to_test == 'gg':
    rl_dav_3d = rl_gg_3d
    cl_dav_3d = cl_gg_3d
else:
    raise ValueError('probe_to_test not recognized')

common_data_path = f'{ROOT}/common_data/vincenzo'
rl_ll_vinc_2d = np.loadtxt(f'{common_data_path}/Pk_responses_2D/EP10/rijllcorr-istf-alex.dat')
rl_gl_vinc_2d = np.loadtxt(f'{common_data_path}/Pk_responses_2D/EP10/rijglcorr-istf-alex.dat')
rl_gg_vinc_2d = np.loadtxt(f'{common_data_path}/Pk_responses_2D/EP10/rijggcorr-istf-alex.dat')

# some example cls, not necessarily the ones used for these responses!!
cl_ll_vinc_2d = np.genfromtxt(f'{common_data_path}/Cij-NonLin-eNLA_15gen/CijLL-LCDM-NonLin-eNLA.dat')
cl_lg_vinc_2d = np.genfromtxt(f'{common_data_path}/Cij-NonLin-eNLA_15gen/CijLG-LCDM-NonLin-eNLA.dat')  # may be GL!!!
cl_gg_vinc_2d = np.genfromtxt(f'{common_data_path}/Cij-NonLin-eNLA_15gen/CijGG-LCDM-NonLin-eNLA.dat')

# extract ell values
ell_rl_ll_vinc = 10 ** rl_ll_vinc_2d[:, 0]
ell_rl_gl_vinc = 10 ** rl_gl_vinc_2d[:, 0]
ell_rl_gg_vinc = 10 ** rl_gg_vinc_2d[:, 0]
ell_cl_ll_vinc = cl_ll_vinc_2d[:, 0]
ell_cl_lg_vinc = cl_lg_vinc_2d[:, 0]
ell_cl_gg_vinc = cl_gg_vinc_2d[:, 0]

# they are the same for all probes:
assert np.all(ell_rl_ll_vinc == ell_rl_gl_vinc) and np.all(ell_rl_gl_vinc == ell_rl_gg_vinc), 'ell values do not match'
assert np.all(ell_cl_ll_vinc == ell_cl_lg_vinc) and np.all(ell_cl_lg_vinc == ell_cl_gg_vinc), 'ell values do not match'
ell_rl_vinc = ell_rl_ll_vinc
ell_cl_vinc = ell_cl_ll_vinc

# slice arrays to remove ell values
rl_ll_vinc_2d = rl_ll_vinc_2d[:, 1:]
rl_gl_vinc_2d = rl_gl_vinc_2d[:, 1:]
rl_gg_vinc_2d = rl_gg_vinc_2d[:, 1:]
cl_ll_vinc_2d = cl_ll_vinc_2d[:, 1:]
cl_lg_vinc_2d = cl_lg_vinc_2d[:, 1:]
cl_gg_vinc_2d = cl_gg_vinc_2d[:, 1:]

# reshape in 3d
zpairs_auto, zpairs_cross, zpairs_3x2pt = mm.get_zpairs(zbins)
rl_ll_vinc_3d = mm.cl_2D_to_3D_symmetric(rl_ll_vinc_2d, len(ell_rl_ll_vinc), zpairs_auto, zbins)
rl_gl_vinc_3d = mm.cl_2D_to_3D_asymmetric(rl_gl_vinc_2d, len(ell_rl_gl_vinc), zbins, order='C')
rl_gg_vinc_3d = mm.cl_2D_to_3D_symmetric(rl_gg_vinc_2d, len(ell_rl_gg_vinc), zpairs_auto, zbins)

cl_ll_vinc_3d = mm.cl_2D_to_3D_symmetric(cl_ll_vinc_2d, len(ell_cl_ll_vinc), zpairs_auto, zbins)
# apparently the line below is correct. This means either that the file is actually GL, or that the order should be 'F'
# (and the output should then be transposed)
cl_gl_vinc_3d = mm.cl_2D_to_3D_asymmetric(cl_lg_vinc_2d, len(ell_cl_lg_vinc), zbins, order='C')
cl_gg_vinc_3d = mm.cl_2D_to_3D_symmetric(cl_gg_vinc_2d, len(ell_cl_gg_vinc), zpairs_auto, zbins)

# ! last benchmark: pyccl cls

# z_mean = np.array((0.2095, 0.489, 0.619, 0.7335, 0.8445, 0.9595, 1.087, 1.2395, 1.45, 2.038))
# bias_values = np.asarray([b1_func(z) for z in z_mean])
# gal_bias_2d_array = wf_cl_lib.build_galaxy_bias_2d_arr(bias_values, z_mean, zbins, z_grid, bias_model, plot_bias=False)
# kernel_gc = wf_cl_lib.wf_galaxy_ccl(z_grid, 'without_galaxy_bias', cosmo=cosmo_ccl, gal_bias_2d_array=1,
#                                     bias_model=bias_model, return_PyCCL_object=True)

# # new version of the function
# kernel_wl = wf_cl_lib.wf_ccl(z_grid, probe='lensing', which_wf='with_IA', fid_pars_dict=fid_pars_dict,
#                              cosmo_ccl=cosmo_ccl, dndz_tuple=dndz, ia_bias_tuple=None,
#                              gal_bias_tuple=None,
#                              bias_model=bias_model,
#                              return_ccl_obj=False, n_samples=1000)
# warnings.warn('recheck galaxy bias pliiiz')
# kernel_gc = wf_cl_lib.wf_ccl(z_grid, probe='galaxy', which_wf='without_galaxy_bias', fid_pars_dict=fid_pars_dict,
#                              cosmo_ccl=cosmo_ccl, dndz_tuple=dndz, ia_bias_tuple=None,
#                              gal_bias_tuple=(z_grid, gal_bias_2d_array),
#                              bias_model=bias_model,
#                              return_ccl_obj=False, n_samples=1000)


# ! these don't work! understand why
cl_ll_ccl = wf_cl_lib.cl_PyCCL(kernel_wl_ccl_obj, kernel_wl_ccl_obj, ell_cl_vinc,
                               zbins, 'delta_matter:delta_matter', cosmo_ccl, limber_integration_method='spline')
cl_gl_ccl = wf_cl_lib.cl_PyCCL(kernel_gc_ccl_obj, kernel_wl_ccl_obj, ell_cl_vinc,
                               zbins, 'delta_matter:delta_matter', cosmo_ccl, limber_integration_method='spline')
cl_gg_ccl = wf_cl_lib.cl_PyCCL(kernel_gc_ccl_obj, kernel_gc_ccl_obj, ell_cl_vinc,
                               zbins, 'delta_matter:delta_matter', cosmo_ccl, limber_integration_method='spline')

# cl_LL are ok, responses as well (better with use_h_units = True)

# interpolate cl_gg_vinc_3d on ell_grid usint numpy interpolate on axis=0
custom_lines = [Line2D([0], [0], color='black', lw=2, ls='-'),
                Line2D([0], [0], color='black', lw=2, ls='--'),
                Line2D([0], [0], color='black', lw=2, ls=':')]

mpl.rcParams['font.size'] = 13
figsize = (25, 6)
fig, ax = plt.subplots(1, 3, figsize=figsize)  # 1 row, 3 columns
for i in range(zbins):
    j = 5
    ax[0].loglog(ell_grid, cl_ll_3d[:, i, j], label=f'cl_ll_3d {i, j}', c=colors[i], ls='-', alpha=0.8)
    ax[0].loglog(ell_cl_vinc, cl_ll_vinc_3d[:, i, j], label=f'cl_ll_vinc_3d {i, j}', c=colors[i], ls='--', alpha=0.8)
    ax[0].loglog(ell_cl_vinc, cl_ll_ccl[:, i, j], label=f'cl_ll_ccl {i, j}', c=colors[i], ls=':', alpha=0.8)

    ax[1].loglog(ell_grid, cl_gl_3d[:, i, j], label=f'cl_gl_3d {i, j}', c=colors[i], ls='-', alpha=0.8)
    ax[1].loglog(ell_cl_vinc, cl_gl_vinc_3d[:, i, j], label=f'cl_gl_vinc_3d {i, j}', c=colors[i], ls='--', alpha=0.8)
    ax[1].loglog(ell_cl_vinc, cl_gl_ccl[:, i, j], label=f'cl_gl_ccl {i, j}', c=colors[i], ls=':', alpha=0.8)

    ax[2].loglog(ell_grid, cl_gg_3d[:, i, j], label=f'cl_gg_3d {i, j}', c=colors[i], ls='-', alpha=0.8)
    ax[2].loglog(ell_cl_vinc, cl_gg_vinc_3d[:, i, j], label=f'cl_gg_vinc_3d {i, j}', c=colors[i], ls='--', alpha=0.8)
    ax[2].loglog(ell_cl_vinc, cl_gg_ccl[:, i, j], label=f'cl_gg_ccl {i, j}', c=colors[i], ls=':', alpha=0.8)

    ax[0].set_ylabel('C_ell')

    fig.text(0.5, 0.04, 'ell', ha='center')

    # Set titles for each subplot
    ax[0].set_title('LL')
    ax[1].set_title('GL')
    ax[2].set_title('GG')

    # plt.tight_layout()  # Adjusts the layout to prevent overlap
plt.legend(custom_lines, ['dav', 'vinc', 'ccl'])

fig, ax = plt.subplots(1, 3, figsize=figsize)  # 1 row, 3 columns
for i in range(zbins):
    j = 5
    ax[0].plot(ell_grid, rl_ll_3d[:, i, j], label=f'rl_ll_3d {i, j}', c=colors[i], ls='-', alpha=0.8)
    ax[0].plot(ell_rl_vinc, rl_ll_vinc_3d[:, i, j], label=f'rl_ll_vinc_3d {i, j}', c=colors[i], ls='--', alpha=0.8)

    ax[1].plot(ell_grid, rl_gl_3d[:, i, j], label=f'rl_gl_3d {i, j}', c=colors[i], ls='-', alpha=0.8)
    ax[1].plot(ell_rl_vinc, rl_gl_vinc_3d[:, i, j], label=f'rl_gl_vinc_3d {i, j}', c=colors[i], ls='--', alpha=0.8)

    ax[2].plot(ell_grid, rl_gg_3d[:, i, j], label=f'rl_gg_3d {i, j}', c=colors[i], ls='-', alpha=0.8)
    ax[2].plot(ell_rl_vinc, rl_gg_vinc_3d[:, i, j], label=f'rl_gg_vinc_3d {i, j}', c=colors[i], ls='--', alpha=0.8)

    ax[0].set_xscale('log')
    ax[1].set_xscale('log')
    ax[2].set_xscale('log')
    ax[0].set_ylabel('R_ell')

    fig.text(0.5, 0.04, 'ell', ha='center')

    # Set titles for each subplot
    ax[0].set_title('LL')
    ax[1].set_title('GL')
    ax[2].set_title('GG')

    plt.tight_layout()  # Adjusts the layout to prevent overlap
plt.legend(custom_lines, ['dav', 'vinc'])

if cfg['plot_responses_allprobes']:
    # this is just for the paper/thesis!
    mpl.rcParams['font.size'] = 18

    plt.figure()
    plt.plot(ell_rl_vinc, rl_ll_vinc_3d[:, 5, 5], label='LL')
    plt.plot(ell_rl_vinc, rl_gl_vinc_3d[:, 5, 5], label='GL')
    plt.plot(ell_rl_vinc, rl_gg_vinc_3d[:, 5, 5], label='GG')

    plt.legend()
    plt.xscale('log')
    plt.xlabel('$\ell$')
    plt.ylabel('$\\mathcal{R}_{5, 5}^{AB}(\ell)$')
    plt.savefig(f'../output/plots/responses_vinc_allprobes.pdf', bbox_inches='tight', dpi=500)

print('*********** done ***********')
"""