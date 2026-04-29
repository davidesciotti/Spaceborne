import os

import healpy as hp
import numpy as np

from spaceborne import constants, cosmo_lib, io_handler


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


def up_downgrade_map(map_in, nside_out):
    """Very simple wrapper, basically gets nside_in and prints a message
    if up/downgrading is needed"""

    nside_in = hp.get_nside(map_in)
    if nside_out is not None and nside_out != nside_in:
        print(f'Changing map resolution from nside = {nside_in} to nside = {nside_out}')
        map_out = hp.ud_grade(map_in=map_in, nside_out=nside_out)
        return map_out
    else:
        return map_in


class Mask:
    def __init__(self, mask_cfg):
        self.use_polar_cap = mask_cfg['use_polar_cap']
        self.use_footprint = mask_cfg['use_footprint']
        self.use_weight_maps = mask_cfg['use_weight_maps']

        self.footprint_ll_filename = mask_cfg['footprint_LL_filename']
        self.footprint_gg_filename = mask_cfg['footprint_GG_filename']
        self.weight_maps_ll_filename = mask_cfg['weight_maps_LL_filename']
        self.weight_maps_gg_filename = mask_cfg['weight_maps_GG_filename']

        self.nside_cfg = mask_cfg['nside']
        self.desired_survey_area_deg2 = mask_cfg['survey_area_deg2']
        self.apodize = mask_cfg['apodize']
        self.aposize = float(mask_cfg['aposize'])

    def process(self):

        # ! 1. load footprint/weight maps or generate polar cap
        if self.use_footprint:
            # load
            self.footprint_ll = io_handler.load_footprint(
                path=self.footprint_ll_filename, nside=self.nside_cfg
            )
            self.footprint_gg = io_handler.load_footprint(
                path=self.footprint_gg_filename, nside=self.nside_cfg
            )
            # get nside and up/downgrade if needed
            self.footprint_ll = up_downgrade_map(self.footprint_ll, self.nside_cfg)
            self.footprint_gg = up_downgrade_map(self.footprint_gg, self.nside_cfg)

        elif self.use_polar_cap:
            self.footprint_ll = generate_polar_cap_func(
                self.desired_survey_area_deg2, self.nside_cfg
            )
            self.footprint_gg = generate_polar_cap_func(
                self.desired_survey_area_deg2, self.nside_cfg
            )

        if self.use_weight_maps:
            # load
            self.weight_maps_ll = io_handler.load_weight_map_fits(
                self.weight_maps_ll_filename
            )
            self.weight_maps_gg = io_handler.load_weight_map_fits(
                self.weight_maps_gg_filename
            )
            import ipdb; ipdb.set_trace()
            # get nside and up/downgrade if needed
            for zi in range(self.weight_maps_ll.shape[0]):
                self.weight_maps_ll[zi] = up_downgrade_map(
                    self.weight_maps_ll[zi], self.nside_cfg
                )
            for zi in range(self.weight_maps_gg.shape[1]):
                self.weight_maps_gg[zi] = up_downgrade_map(
                    self.weight_maps_gg[zi], self.nside_cfg
                )
            # create corresponding footprints
            support_ll = self.weight_maps_ll > 0
            if not np.all(support_ll == support_ll[0]):
                raise ValueError('LL weight map support is not the same across bins')
            self.footprint_ll = support_ll[0].astype(float)

            support_gg = self.weight_maps_gg > 0
            if not np.all(support_gg == support_gg[0]):
                raise ValueError('GG weight map support is not the same across bins')
            self.footprint_gg = support_gg[0].astype(float)

            self.footprint_ll = np.ones_like(self.weight_maps_ll[0])
            self.footprint_gg = np.ones_like(self.weight_maps_gg[0])
            self.footprint_ll[self.weight_maps_ll[0] == 0] = 0
            self.footprint_gg[self.weight_maps_gg[0] == 0] = 0

        # ! 2. apodize
        if self.apodize:
            print(f'Apodizing footprints with aposize = {self.aposize} deg')
            import pymaster as nmt

            # Ensure the mask is float64 before apodization
            self.footprint_ll = self.footprint_ll.astype('float64', copy=False)
            self.footprint_ll = nmt.mask_apodization(
                self.footprint_ll, aposize=self.aposize
            )
            self.footprint_gg = self.footprint_gg.astype('float64', copy=False)
            self.footprint_gg = nmt.mask_apodization(
                self.footprint_gg, aposize=self.aposize
            )

        # ! 3. get mask spectrum and fsky (the latter is from the healpix mask!!)
        self.ell_mask, self.cl_mask, self.fsky = get_mask_cl(self.footprint_ll)
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

    def plot_maps(self):

        hp.mollview(
            self.footprint_ll, cmap='inferno_r', title='Footprint LL - Mollweide view'
        )
        hp.mollview(
            self.footprint_gg, cmap='inferno_r', title='Footprint GG - Mollweide view'
        )

        if self.use_weight_maps:
            for zi in range(self.weight_maps_ll.shape[0]):
                hp.mollview(
                    self.weight_maps_ll[zi],
                    cmap='inferno_r',
                    title=f'Weight map LL, zi={zi} - Mollweide view',
                )
                hp.mollview(
                    self.weight_maps_gg[zi],
                    cmap='inferno_r',
                    title=f'Weight map GG, zi={zi} - Mollweide view',
                )
