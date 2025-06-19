from copy import deepcopy

import numpy as np
from cloelib.cosmology.camb_cosmology import (
    CAMBBackground,
    CAMBNonLinearPerturbations,
)
from cloelib.observables.photo import PositionsTracer, ShearTracer
from cloelib.summary_statistics.angular_two_point import AngularTwoPoint
from scipy.interpolate import CubicSpline

from spaceborne import cosmo_lib

# TODO problem: is there a way to use z_grid in the rest of the code without manually
# TODO interpolate pk?


class CloelibInterface:
    def __init__(self, cfg, pvt_cfg, z_grid, k_grid):
        self.cfg = cfg
        self.pvt_cfg = pvt_cfg
        self.zbins = self.pvt_cfg['zbins']

        # camb accepts at most 256 redshifts
        self.z_grid_pk = np.linspace(z_grid.min(), z_grid.max(), 256)
        self.z_grid = z_grid
        self.k_grid = k_grid

        self.poly_order = 4
        assert self.poly_order == len(self.cfg['C_ell']['galaxy_bias_fit_coeff']), (
            f'Poly order {self.poly_order} does not match the number of galaxy bias '
            f'coefficients {len(self.cfg["C_ell"]["galaxy_bias_fit_coeff"])}.'
        )

    def cfg_sanity_checks(self):
        assert self.cfg['intrinsic_alignment']['z_pivot_IA'] == 0, (
            'cloelib only supports z_pivot_IA = 0'
        )

    def preprocess_cosmo_pars(self):
        """Translates the SB parameter basis to the cloelib one"""
        # start with SB params
        self.cosmo_dict = deepcopy(self.cfg['cosmology'])

        # add H0
        self.cosmo_dict['H0'] = self.cosmo_dict['h'] * 100

        # Go from (Omega_m0, Omega_b0) to (Omega_cdm0, Omega_b0)
        self.cosmo_dict['Omega_nu0'] = cosmo_lib.get_omega_nu0(
            m_nu=self.cosmo_dict['m_nu'],
            h=self.cosmo_dict['h'],
            n_eff=self.cosmo_dict['Neff'],
            neutrino_mass_fac=94.07,
        )
        self.cosmo_dict['Omega_cdm0'] = (
            self.cosmo_dict['Omega_m0']
            - self.cosmo_dict['Omega_b0']
            - self.cosmo_dict['Omega_nu0']
        )

        # nonlinear model
        self.cosmo_dict['nonlinear_model'] = self.cfg['extra_parameters']['camb'][
            'halofit_version'
        ]
        self.cosmo_dict['log10TAGN'] = self.cfg['extra_parameters']['camb'][
            'HMCode_logT_AGN'
        ]

    def set_nuisance_dict_pos(self, mag_bias_table):
        """Translates the SB parameter basis to the cloelib one, but for the
        nuisance parameters"""

        # galaxy bias
        if self.cfg['C_ell']['which_gal_bias'] == 'FS2_polynomial_fit':
            self.galaxy_bias_model = 'poly'
        else:
            raise NotImplementedError(
                f'Galaxy bias model {self.cfg["C_ell"]["which_gal_bias"]} '
                'not implemented.'
            )
        # now set the polynomial coefficients
        self.nuisance_dict_pos = {
            f'b1_photo_poly{i}': self.cfg['C_ell']['galaxy_bias_fit_coeff'][i]
            for i in range(self.poly_order)
        }

        # magnification bias
        if self.cfg['C_ell']['has_magnification_bias']:
            # poly mag bias not implemented in cloelib at the moment
            if self.cfg['C_ell']['which_mag_bias'] == 'FS2_polynomial_fit':
                raise NotImplementedError(
                    'Polynomial magnification bias model is not implemented in cloelib. '
                    'Please set `which_mag_bias`: "from_input" and pass an input file with '
                    'a constant magnification bias value in each bin'
                )
            # to pass per-bin values, use the 'from_input' option (it's a
            # bit of a workaround)
            elif self.cfg['C_ell']['which_mag_bias'] == 'from_input':
                raise NotImplementedError('This is yet to be tested!')
                for zi in range(self.zbins):
                    np.testing.assert_allclose(
                        mag_bias_table[0, zi],
                        mag_bias_table[:, zi],
                        atol=0,
                        rtol=1e-5,
                        err_msg=f'Magnification bias table for bin {zi + 1} '
                        'is not constant over the redshift range',
                    )
            else:
                raise NotImplementedError(
                    f'Magnification bias model {self.cfg["C_ell"]["which_mag_bias"]} '
                    'not implemented.'
                )

        else:
            # if no magnification bias is required, set the coefficients to zero
            for zi in range(self.zbins):
                self.nuisance_dict_pos[f'magnification_bias_{zi + 1}'] = 0.0

        # dz GC
        for zi in range(self.zbins):
            self.nuisance_dict_pos[f'dz_pos_{zi + 1}'] = 0
            # if self.cfg['nz']['shift_nz']:
            # self.nuisance_dict_pos[f'dz_pos_{zi + 1}'] = self.cfg['nz']['dzGC'][zi]
            # else:
            # self.nuisance_dict_pos[f'dz_pos_{zi + 1}'] = 0

    def set_nuisance_dict_she(self):
        """Translates the SB parameter basis to the cloelib one, but for the
        nuisance parameters"""
        # IA
        self.nuisance_dict_she = deepcopy(self.cfg['intrinsic_alignment'])

        for zi in range(self.zbins):
            # multiplicative shear bias
            self.nuisance_dict_she[f'multiplicative_bias_{zi + 1}'] = (
                self.cfg['C_ell']['mult_shear_bias'][zi]
            )  # fmt: skip

            # dz WL
            self.nuisance_dict_she[f'dz_shear_{zi + 1}'] = 0
            # if self.cfg['nz']['shift_nz']:
            #     self.nuisance_dict_she[f'dz_shear_{zi + 1}'] = self.cfg['nz']['dzWL'][
            #         zi
            #     ]
            # else:
            #     self.nuisance_dict_she[f'dz_shear_{zi + 1}'] = 0

    def set_background(self):
        # The arguments of the Background functions follow the cosmology.API
        self.background = CAMBBackground(
            H0=self.cosmo_dict['H0'],
            Omega_cdm0=self.cosmo_dict['Omega_cdm0'],
            Omega_b0=self.cosmo_dict['Omega_b0'],
            w0=self.cosmo_dict['w0'],
            wa=self.cosmo_dict['wa'],
            Omega_k0=self.cosmo_dict['Omega_k0'],
            ns=self.cosmo_dict['ns'],
            As=self.cosmo_dict['As'],
            mnu=self.cosmo_dict['m_nu'],
            gamma_MG=self.cosmo_dict['gamma_MG'],
        )

    def set_perturbations(self):
        self.perturbations = CAMBNonLinearPerturbations(
            background=self.background,
            redshifts=self.z_grid_pk,
            nonlinear_model=self.cosmo_dict['nonlinear_model'],
            HMCode_logT_AGN=self.cosmo_dict['log10TAGN'],
        )

    def nuis_sanity_check(self):
        # TODO? I already check things like the len(dz) == zbins...
        pass

    def _resample_dndz(self, nz_in, z_nz_in, z_nz_out):
        nz_out = np.zeros([self.zbins, len(z_nz_out)])  # shape: (zbins, nz_grid)
        for zi in range(self.zbins):
            nz_out[zi, :] = np.interp(x=z_nz_out, xp=z_nz_in, fp=nz_in[:, zi])
        return nz_out

    def set_nz(self, z_nz, nz_pos_sb, nz_she_sb):
        # shape: (nz_grid, zbins)
        assert nz_pos_sb.shape[1] == self.zbins, (
            f'Number of bins in nz_pos ({nz_pos_sb.shape[1]}) does not match '
            f'zbins ({self.zbins}).'
        )
        assert nz_she_sb.shape[1] == self.zbins, (
            f'Number of bins in nz_she ({nz_she_sb.shape[1]}) does not match '
            f'zbins ({self.zbins}).'
        )

        # TODO normalize or check normalization
        # Function to resample the normalized dndz

        self.nz_pos = self._resample_dndz(
            nz_in=nz_pos_sb, z_nz_in=z_nz, z_nz_out=self.z_grid_pk
        )
        self.nz_she = self._resample_dndz(
            nz_in=nz_she_sb, z_nz_in=z_nz, z_nz_out=self.z_grid_pk
        )

    def set_pos_tracer(self, mag_bias_table=None):
        self.tracer_pos = PositionsTracer(
            perturbations=self.perturbations,
            dndz=self.nz_pos,
            z=self.z_grid_pk,
            galaxy_bias_model=self.galaxy_bias_model,
            nuisance_params=self.nuisance_dict_pos,
        )

        # Remove galaxy bias to have "clean" wf_delta kernel.
        # This is achieved by setting bg = cons = 1, via the 0-th order coefficient
        _nuisance_dict_pos_nogb = deepcopy(self.nuisance_dict_pos)
        _nuisance_dict_pos_nogb['b1_photo_poly0'] = 1
        for zi in range(1, self.poly_order):
            _nuisance_dict_pos_nogb[f'b1_photo_poly{zi}'] = 0.0
        self.tracer_pos_nogb = PositionsTracer(
            perturbations=self.perturbations,
            dndz=self.nz_pos,
            z=self.z_grid_pk,
            galaxy_bias_model='poly',
            nuisance_params=_nuisance_dict_pos_nogb,
        )

    def set_she_tracer(self):
        self.tracer_she = ShearTracer(
            perturbations=self.perturbations,
            dndz=self.nz_she,
            z=self.z_grid_pk,
            nuisance_params=self.nuisance_dict_she,
        )

    def set_wf_arrays(self, znew=None):
        """Set the window function arrays for positions and shear tracers."""
        # Positions tracer
        self.wf_delta = self.tracer_pos_nogb.get_window_positions(self.z_grid_pk).T
        self.wf_mu = self.tracer_pos_nogb.get_window_magnification(self.z_grid_pk).T
        self.wf_galaxy = self.tracer_pos_nogb.get_window(self.z_grid_pk).T

        # Shear tracer
        self.wf_gamma = self.tracer_she.get_window_lensing(self.z_grid_pk).T
        self.wf_ia = self.tracer_she.get_window_IA(self.z_grid_pk).T
        self.wf_lensing = self.tracer_she.get_window(self.z_grid_pk).T

        # Interpolate the window functions to the requested redshift
        if znew is not None:
            kw = dict(x=self.z_grid_pk, axis=0)
            self.wf_delta = CubicSpline(y=self.wf_delta, **kw)(znew)
            self.wf_mu = CubicSpline(y=self.wf_mu, **kw)(znew)
            self.wf_galaxy = CubicSpline(y=self.wf_galaxy, **kw)(znew)
            self.wf_gamma = CubicSpline(y=self.wf_gamma, **kw)(znew)
            self.wf_ia = CubicSpline(y=self.wf_ia, **kw)(znew)
            self.wf_lensing = CubicSpline(y=self.wf_lensing, **kw)(znew)

    def get_cl_3d(self, ells, tracer_a, tracer_b):
        twopoint_obj = AngularTwoPoint(tracer_a, tracer_b)
        return twopoint_obj.get_Cl(ells=ells, nl=0, ks=self.perturbations.k)
