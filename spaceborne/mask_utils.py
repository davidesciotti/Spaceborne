import os

import healpy as hp
import numpy as np

from spaceborne import cosmo_lib
from spaceborne import constants


def get_mask_cl(mask: np.ndarray) -> tuple:
    cl_mask = hp.anafast(mask)
    ell_mask = np.arange(len(cl_mask))
    fsky_mask = np.mean(mask**2)  # TODO 2 different masks
    return ell_mask, cl_mask, fsky_mask


def generate_polar_cap_func(area_deg2, nside):
    _fsky_expected = cosmo_lib.deg2_to_fsky(area_deg2)
    print(
        f'\nGenerating a polar cap mask with area {area_deg2} deg^2 and nside {nside}'
    )

    # Convert the area to radians squared for the angular radius calculation
    area_rad2 = area_deg2 * (np.pi / 180) ** 2

    # The area of a cap is given by A = 2*pi*(1 - cos(theta)),
    # so solving for theta gives the angular radius of the cap
    theta_cap_rad = np.arccos(1 - area_rad2 / (2 * np.pi))

    # Convert the angular radius to degrees for visualization
    theta_cap_deg = np.degrees(theta_cap_rad)
    print(f'Angular radius of the cap: {theta_cap_deg:.4f} deg')

    # Vector pointing to the North Pole (θ=0); φ can take any value
    vec = hp.ang2vec(0, 0)
    pixels_in_cap = hp.query_disc(nside, vec, theta_cap_rad)

    # Set the pixels within the cap to 1
    mask = np.zeros(hp.nside2npix(nside))
    mask[pixels_in_cap] = 1

    # Calculate the actual sky fraction of the generated mask
    # fsky_actual = np.sum(mask) / len(mask)
    # print(f'Measured fsky from the mask: {fsky_actual:.4f}')

    return mask


def _read_masking_map(path, nside, *, nest=False):
    """
    Read a HEALPix map in "partial" format from *path* and return it at
    resolution *nside*.

    The returned NSIDE cannot be larger than the NSIDE of the stored
    map.

    If *nest* is true, returns the map in NESTED ordering.
    """
    import fitsio

    data, header = fitsio.read(path, header=True)
    nside_in = header['NSIDE']
    fact = (nside_in // nside) ** 2
    if fact == 0:
        raise ValueError(f'requested NSIDE={nside} greater than map NSIDE={nside_in}')
    out = np.zeros(12 * nside**2)
    ipix, wht = data['PIXEL'], data['WEIGHT']
    order = header['ORDERING']
    if order == 'RING':
        ipix = hp.ring2nest(nside, ipix)
    elif order != 'NESTED':
        raise ValueError(f'unknown pixel ordering {order} in map')
    ipix = ipix // fact
    if not nest:
        ipix = hp.nest2ring(nside, ipix)
    np.add.at(out, ipix, wht / fact)
    return out


class Mask:
    def __init__(self, mask_cfg):
        self.load_mask = mask_cfg['load_mask']
        self.mask_filename = mask_cfg['mask_filename']
        self.nside = mask_cfg['nside']
        self.desired_survey_area_deg2 = mask_cfg['survey_area_deg2']
        self.apodize = mask_cfg['apodize']
        self.aposize = float(mask_cfg['aposize'])
        self.generate_polar_cap = mask_cfg['generate_polar_cap']

    def load_mask_func(self):
        if not os.path.exists(self.mask_filename):
            raise FileNotFoundError(f'{self.mask_filename} does not exist.')

        print(f'\nLoading mask file from {self.mask_filename}\n')

        if self.mask_filename.endswith('.fits') or self.mask_filename.endswith('.fits.gz'):
            try:
                # function provided by VMPZ team to read very high resolution map
                # and downgrade it on the fly
                self.mask = _read_masking_map(self.mask_filename, self.nside)
            except ValueError as ve:
                self.mask = hp.read_map(self.mask_filename)
                print(
                    f'ValueError raised: {ve}, \n'
                    'falling back on hp.read_map to read input map'
                )

        elif self.mask_filename.endswith('.npy'):
            self.mask = np.load(self.mask_filename)

        else:
            raise ValueError(
                f'Unsupported file format for mask file: {self.mask_filename}'
                'Supported formats are .fits, .fits.gz and .npy'
            )

    def process(self):
        # 1. load or generate mask

        # check that one and only one of load_mask and generate_polar_cap is True
        assert self.load_mask != self.generate_polar_cap, (
            'Please choose whether to load OR generate the mask, not neither or both.'
        )

        if self.load_mask:
            self.load_mask_func()
            self.nside_mask = hp.get_nside(self.mask)

        elif self.generate_polar_cap:
            self.mask = generate_polar_cap_func(
                self.desired_survey_area_deg2, self.nside
            )

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

        # 3. get mask spectrum and fsky (the latter is from the healpix mask!!)
        self.ell_mask, self.cl_mask, self.fsky = get_mask_cl(self.mask)
        self.cl_mask_norm = (
            self.cl_mask * (2 * self.ell_mask + 1) / (4 * np.pi * self.fsky) ** 2
        )

        # 4. finally, set survey area in steradians and other useful quantities
        self.survey_area_deg2 = self.fsky * constants.DEG2_IN_SPHERE
        self.survey_area_sr = self.survey_area_deg2 * constants.DEG2_TO_SR

        # else:
        #     print(
        #         'No mask provided or requested. The covariance terms will be '
        #         'rescaled by 1/fsky'
        #     )
        #     self.ell_mask = None
        #     self.cl_mask = None
        #     self.fsky = self.survey_area_deg2 / constants.DEG2_IN_SPHERE

        print(f'fsky = {self.fsky:.4f}')
        print(f'survey_area_sr = {self.survey_area_sr:.4f}')
        print(f'survey_area_deg2 = {self.survey_area_deg2:.4f}\n')
