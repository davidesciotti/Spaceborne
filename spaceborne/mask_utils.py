import os

import healpy as hp
import numpy as np

from spaceborne import cosmo_lib


def get_mask_cl(mask: np.ndarray) -> tuple:
    cl_mask = hp.anafast(mask)
    ell_mask = np.arange(len(cl_mask))
    fsky_mask = np.mean(mask**2)  # TODO 2 different masks
    return ell_mask, cl_mask, fsky_mask


def find_lmax(ell, cl_mask, var_tol=0.05, debug=False):
    """Auxiliary routine to search the best lmax for all later sums on multipoles.

    Computes the smallest lmax so that we reach convergence of the variance
    ..math ::
        var = \sum_\ell  \\frac{(2\ell + 1)}{4\pi} C_\ell^{mask}

    Parameters
    ----------
    ell : array_like
        Full vector of multipoles. As large as possible of shape (nell,)
    cl_mask : array_like
        power spectrum of the mask at the supplied multipoles of shape (nell,).
    var_tol : float, default 0.05
         Float that drives the target precision for the sum over angular multipoles.
         Default is 5%. Lowering it means increasing the number of multipoles
         thus increasing computational time.

    Returns
    -------
    float
        lmax
    """

    assert ell.ndim == 1, 'ell must be a 1-dimensional array'
    assert cl_mask.ndim == 1, 'cl_mask must be a 1-dimensional array'
    assert len(ell) == len(cl_mask), 'ell and cl_mask must have the same size'
    lmaxofcl = ell.max()
    summand = (2 * ell + 1) / (4 * np.pi) * cl_mask
    var_target = np.sum(summand)
    # Initialisation
    lmax = 0
    var_est = np.sum(summand[: (lmax + 1)])
    while abs(var_est - var_target) / var_target > var_tol and lmax < lmaxofcl:
        lmax = lmax + 1
        var_est = np.sum(summand[: (lmax + 1)])
        if debug:
            print(
                'In lmax search',
                lmax,
                abs(var_est - var_target) / var_target,
                var_target,
                var_est,
            )
    lmax = min(lmax, lmaxofcl)  # make sure we didnt overshoot at the last iteration
    return lmax


def generate_polar_cap_func(area_deg2, nside):
    fsky_expected = cosmo_lib.deg2_to_fsky(area_deg2)
    print(f'Generating a polar cap mask with area {area_deg2} deg^2 and nside {nside}')

    # Convert the area to radians squared for the angular radius calculation
    area_rad2 = area_deg2 * (np.pi / 180) ** 2

    # The area of a cap is given by A = 2*pi*(1 - cos(theta)),
    # so solving for theta gives the angular radius of the cap
    theta_cap_rad = np.arccos(1 - area_rad2 / (2 * np.pi))

    # Convert the angular radius to degrees for visualization
    theta_cap_deg = np.degrees(theta_cap_rad)
    print(f'Angular radius of the cap: {theta_cap_deg:.4f} deg')

    # Vector pointing to the North Pole (θ=0, φ can be anything since
    # θ=0 defines the pole)
    vec = hp.ang2vec(0, 0)
    pixels_in_cap = hp.query_disc(nside, vec, theta_cap_rad)

    # Set the pixels within the cap to 1
    mask = np.zeros(hp.nside2npix(nside))
    mask[pixels_in_cap] = 1

    # Calculate the actual sky fraction of the generated mask
    # fsky_actual = np.sum(mask) / len(mask)
    # print(f'Measured f_sky from the mask: {fsky_actual:.4f}')

    return mask


class Mask:
    def __init__(self, mask_cfg):
        self.load_mask = mask_cfg['load_mask']
        self.mask_path = mask_cfg['mask_path']
        self.nside = mask_cfg['nside']
        self.survey_area_deg2 = mask_cfg['survey_area_deg2']
        self.apodize = mask_cfg['apodize']
        self.aposize = float(mask_cfg['aposize'])
        self.generate_polar_cap = mask_cfg['generate_polar_cap']

    def load_mask_func(self):
        if not os.path.exists(self.mask_path):
            raise FileNotFoundError(f'{self.mask_path} does not exist.')

        print(f'Loading mask file from {self.mask_path}')
        if self.mask_path.endswith('.fits'):
            self.mask = hp.read_map(self.mask_path)
        elif self.mask_path.endswith('.npy'):
            self.mask = np.load(self.mask_path)

    def process(self):
        # 1. load or generate mask

        # check they are not both True
        assert self.load_mask or self.generate_polar_cap, (
            'Please choose whether to load or generate the mask, not both.'
        )

        if self.load_mask:
            self.load_mask_func()
            self.nside_mask = hp.get_nside(self.mask)

        elif self.generate_polar_cap:
            self.mask = generate_polar_cap_func(self.survey_area_deg2, self.nside)

        if self.load_mask and self.nside is not None and self.nside != self.nside_mask:
            print(
                f'Changing mask resolution from nside = '
                f'{self.nside_mask} to nside = {self.nside}'
            )
            self.mask = hp.ud_grade(map_in=self.mask, nside_out=self.nside)

        # 2. apodize
        if hasattr(self, 'mask') and self.apodize:
            print(f'Apodizing mask with aposize = {self.aposize} deg')
            import pymaster as nmt

            # Ensure the mask is float64 before apodization
            self.mask = self.mask.astype('float64', copy=False)
            self.mask = nmt.mask_apodization(self.mask, aposize=self.aposize)

        # 3. get mask spectrum and fsky
        self.ell_mask, self.cl_mask, self.fsky = get_mask_cl(self.mask)
        # normalization has been checked from
        # https://github.com/tilmantroester/KiDS-1000xtSZ/blob/master/scripts/compute_SSC_mask_power.py
        # and is the same as CSST paper https://zenodo.org/records/7813033
        self.cl_mask_norm = (
            self.cl_mask * (2 * self.ell_mask + 1) / (4 * np.pi * self.fsky) ** 2
        )

        # else:
        #     print(
        #         'No mask provided or requested. The covariance terms will be '
        #         'rescaled by 1/fsky'
        #     )
        #     self.ell_mask = None
        #     self.cl_mask = None
        #     self.fsky = self.survey_area_deg2 / constants.DEG2_IN_SPHERE

        print(f'fsky = {self.fsky:.4f}')
