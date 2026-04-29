import os

import healpy as hp
import numpy as np

from spaceborne import constants, cosmo_lib, io_handler


def combined_fsky(map1, map2):
    """Combine two masks (e.g. footprint and weight map) by multiplying them
    and compute the resulting fsky."""
    fsky_combined = np.mean(map1 * map2)
    return fsky_combined


def get_maps_cl(map1: np.ndarray, map2: np.ndarray) -> tuple:
    cl = hp.anafast(map1, map2)
    ells = np.arange(len(cl))
    return ells, cl


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
    def __init__(self, mask_cfg, probe):
        self.probe = probe
        self.geometry = mask_cfg[probe]['geometry']

        self.use_weight_maps = (
            False if mask_cfg[probe]['weight_maps_filename'] is None else True
        )

        self.footprint_filename = mask_cfg[probe]['footprint_filename']
        self.weight_maps_filename = mask_cfg[probe]['weight_maps_filename']
        
        self.footprint = None
        self.weight_maps = None

        self.nside_cfg = mask_cfg['nside']
        self.desired_survey_area_deg2 = mask_cfg['survey_area_deg2']
        self.apodize = mask_cfg['apodize']
        self.aposize = float(mask_cfg['aposize'])

    def load(self):

        # ! 1. load footprint/weight maps or generate polar cap
        if self.geometry == 'footprint_file':
            # load
            self.footprint = io_handler.load_footprint(
                path=self.footprint_filename, nside=self.nside_cfg
            )
            # get nside and up/downgrade if needed
            self.footprint = up_downgrade_map(self.footprint, self.nside_cfg)

        elif self.geometry == 'polar_cap':
            self.footprint = generate_polar_cap_func(
                self.desired_survey_area_deg2, self.nside_cfg
            )

        if self.use_weight_maps:
            # load
            self.weight_maps = io_handler.load_weight_map_fits(
                self.weight_maps_filename
            )
            # get nside and up/downgrade if needed
            for zi in range(self.weight_maps.shape[0]):
                self.weight_maps[zi] = up_downgrade_map(
                    self.weight_maps[zi], self.nside_cfg
                )
            # TODO this is a very approximate approach!!
            # self.footprint = np.sum(self.weight_maps, axis=0)
            # self.footprint[self.footprint > 0] = 1

    def apodize_func(self):
        # ! 2. apodize
        if not self.apodize:
            return

        print(f'Apodizing footprints with aposize = {self.aposize} deg')
        import pymaster as nmt

        # Ensure the mask is float64 before apodization
        self.footprint = self.footprint.astype('float64', copy=False)
        self.footprint = nmt.mask_apodization(self.footprint, aposize=self.aposize)

        if self.use_weight_maps:
            for zi in range(self.weight_maps.shape[0]):
                self.weight_maps[zi] = self.weight_maps[zi].astype(
                    'float64', copy=False
                )
                self.weight_maps[zi] = nmt.mask_apodization(
                    self.weight_maps[zi], aposize=self.aposize
                )

    def get_cls_fsky(self):
        """get footprint angular power spectrum and effective fsky"""
        self.ells_footprint, self.cl_footprint = get_maps_cl(
            self.footprint, self.footprint
        )
        self.fsky_footprint = combined_fsky(self.footprint, self.footprint)

        self.cl_footprint_norm = (
            self.cl_footprint
            * (2 * self.ells_footprint + 1)
            / (4 * np.pi * self.fsky_footprint) ** 2
        )

        # 4. finally, set survey area in steradians and other useful quantities
        self.survey_area_deg2 = self.fsky_footprint * constants.DEG2_IN_SPHERE
        self.survey_area_sr = self.survey_area_deg2 * constants.DEG2_TO_SR

        # else:
        #     print(
        #         'No mask provided or requested. The covariance terms will be '
        #         'rescaled by 1/fsky'
        #     )
        #     self.ell_mask = None
        #     self.cl_mask = None
        #     self.fsky = self.survey_area_deg2 / constants.DEG2_IN_SPHERE

        # TODO prolly this should be done for all probe and bin combinations...
        if self.use_weight_maps:
            self.ells_weight_maps = []
            self.cl_weight_maps = []
            self.fsky_weight_maps = []
            for zi in range(self.weight_maps.shape[0]):
                ells, cl = get_maps_cl(self.weight_maps[zi], self.weight_maps[zi])
                _fsky = combined_fsky(self.weight_maps[zi], self.weight_maps[zi])
                self.ells_weight_maps.append(ells)
                self.cl_weight_maps.append(cl)
                self.fsky_weight_maps.append(_fsky)

        print(f'fsky = {self.fsky_footprint:.4f}')
        print(f'survey_area_sr = {self.survey_area_sr:.4f}')
        print(f'survey_area_deg2 = {self.survey_area_deg2:.4f}\n')

    def process(self):
        self.load()
        self.apodize_func()
        self.get_cls_fsky()

    def plot_maps(self):

        hp.mollview(
            self.footprint,
            cmap='inferno_r',
            title=f'Footprint {self.probe} - Mollweide view',
        )
        if self.use_weight_maps:
            for zi in range(self.weight_maps.shape[0]):
                hp.mollview(
                    self.weight_maps[zi],
                    cmap='inferno_r',
                    title=f'Weight map {self.probe}, zi={zi} - Mollweide view',
                )
